"""Group DRO from scratch: group weights, group-balanced batches, and the training loop.

Idea: ordinary training (ERM) minimises the average loss, so a small or hard group can stay
badly served. Group DRO keeps one weight per group, q, and raises the weight of whichever
groups currently have high loss, so the model is pushed to serve the worst group.

Each training step:
  1. Draw a batch containing rows from every group (`GroupBalancedBatches`).
  2. Compute each group's mean loss on that batch (`group_mean_losses`).
  3. Update q: multiply each q_g by exp(step_size * loss_g), then rescale q to sum to 1.
  4. Take a gradient step on the weighted loss, sum over g of q_g * loss_g.

With step_size = 0, q stays uniform, and this is plain training on group-balanced batches.
CODE_WALKTHROUGH.md explains this file line by line.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from fin_inclusion.models.pytorch_mlp import POSITIVE_LABEL, PyTorchMLP


def group_mean_losses(row_losses: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Mean loss per group for one batch. Stays differentiable, so gradients flow through it."""
    totals = torch.zeros(n_groups, dtype=row_losses.dtype, device=row_losses.device).scatter_add(
        0, groups, row_losses
    )
    counts = torch.zeros(n_groups, dtype=row_losses.dtype, device=row_losses.device).scatter_add(
        0, groups, torch.ones_like(row_losses)
    )
    if (counts == 0).any():
        # A missing group would silently get loss 0 and lose weight for the wrong reason.
        raise ValueError(f"Every group must appear in every batch; rows per group were {counts.tolist()}")
    return totals / counts


class GroupWeights:
    """The group weights q, and the exponentiated-gradient update that moves them."""

    def __init__(self, group_sizes: Sequence[int], step_size: float, adjustment_c: float = 0.0):
        if step_size < 0:
            raise ValueError("step_size must be non-negative")
        n_groups = len(group_sizes)
        self.step_size = step_size
        self.q = torch.full((n_groups,), 1.0 / n_groups, dtype=torch.float32)
        self._log_q = torch.log(self.q)
        # C / sqrt(n_g): a fixed extra loss for each group, larger for smaller groups, so a small
        # group is treated as if its loss might really be somewhat higher than measured.
        sizes = torch.tensor(list(group_sizes), dtype=torch.float32)
        self.adjustments = adjustment_c / torch.sqrt(sizes)

    def update(self, group_losses: torch.Tensor) -> torch.Tensor:
        """One exponentiated-gradient step on q. q must not carry gradients into the model."""
        if self.step_size == 0:
            return self.q  # Returning early keeps q exactly uniform, with no rounding drift.
        losses = group_losses.detach().to(self.q.device)
        # The same update as q_g * exp(step * loss_g) followed by rescaling, done in log space.
        # Multiplying directly overflows to NaN at high step size and loss (a unit test caught this).
        self._log_q = torch.log_softmax(self._log_q + self.step_size * (losses + self.adjustments), dim=0)
        self.q = torch.exp(self._log_q)
        return self.q

    def weighted_loss(self, group_losses: torch.Tensor) -> torch.Tensor:
        """The objective the model minimises: sum over g of q_g * (loss_g + adjustment_g)."""
        return (self.q * (group_losses + self.adjustments)).sum()


class GroupBalancedBatches:
    """Index batches with the same number of rows from every group.

    Small groups are revisited more often than large ones: each group's rows are shuffled and
    served in order, and reshuffled when used up. One epoch has as many steps as an ordinary
    shuffled epoch over all rows, so the two kinds of training take the same number of updates.
    """

    def __init__(self, groups: torch.Tensor, batch_size: int, generator: torch.Generator):
        self.n_groups = int(groups.max().item()) + 1
        self.rows_per_group = batch_size // self.n_groups
        if self.rows_per_group < 1:
            raise ValueError(f"batch_size {batch_size} is too small for {self.n_groups} groups")
        self.steps_per_epoch = math.ceil(len(groups) / batch_size)
        self.generator = generator
        self.members = [torch.nonzero(groups == g).squeeze(1) for g in range(self.n_groups)]
        if any(len(m) == 0 for m in self.members):
            raise ValueError("Group ids must be contiguous, 0 to n_groups - 1, with no empty group")
        self.queues = [self._shuffled(m) for m in self.members]
        self.cursors = [0] * self.n_groups

    def _shuffled(self, members: torch.Tensor) -> torch.Tensor:
        return members[torch.randperm(len(members), generator=self.generator)]

    def _take(self, g: int) -> torch.Tensor:
        taken = []
        needed = self.rows_per_group
        while needed > 0:
            available = len(self.queues[g]) - self.cursors[g]
            if available == 0:
                self.queues[g] = self._shuffled(self.members[g])
                self.cursors[g] = 0
                continue
            chunk = min(needed, available)
            taken.append(self.queues[g][self.cursors[g] : self.cursors[g] + chunk])
            self.cursors[g] += chunk
            needed -= chunk
        return torch.cat(taken)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _ in range(self.steps_per_epoch):
            yield torch.cat([self._take(g) for g in range(self.n_groups)])


@dataclass
class DROHistory:
    """What happened during training, saved so q can be inspected rather than assumed."""

    q_by_epoch: list[list[float]] = field(default_factory=list)
    group_loss_by_epoch: list[list[float]] = field(default_factory=list)


def train_group_dro(
    net: nn.Module,
    inputs: Sequence[torch.Tensor],
    y: torch.Tensor,
    sample_weight: torch.Tensor,
    groups: torch.Tensor,
    *,
    step_size: float,
    adjustment_c: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 0,
    forward: Callable[..., torch.Tensor] | None = None,
) -> DROHistory:
    """Trains `net` in place with Group DRO. Works for any net whose output is one logit per row.

    `sample_weight` carries the class-imbalance weights, so a Group DRO run handles imbalance the
    same way as an ERM run. The caller seeds torch before building `net`, as PyTorchMLP.fit does.
    """
    n_groups = int(groups.max().item()) + 1
    group_sizes = [int((groups == g).sum().item()) for g in range(n_groups)]
    weights = GroupWeights(group_sizes, step_size=step_size, adjustment_c=adjustment_c)
    batches = GroupBalancedBatches(groups, batch_size, torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    call = forward if forward is not None else net
    history = DROHistory()

    net.train()
    for _ in range(epochs):
        epoch_losses = torch.zeros(n_groups)
        for index in batches:
            optimizer.zero_grad()
            logits = call(*[tensor[index] for tensor in inputs])
            row_losses = loss_fn(logits, y[index]) * sample_weight[index]
            group_losses = group_mean_losses(row_losses, groups[index], n_groups)
            weights.update(group_losses)  # q moves first; the model then steps on the new q.
            weights.weighted_loss(group_losses).backward()
            optimizer.step()
            epoch_losses += group_losses.detach()
        history.q_by_epoch.append(weights.q.tolist())
        history.group_loss_by_epoch.append((epoch_losses / batches.steps_per_epoch).tolist())
    return history


class GroupDROMLP(PyTorchMLP):
    """The baseline MLP architecture, trained with Group DRO instead of ERM.

    Inherits the network, input handling and prediction, so any difference from the ERM MLP
    comes from the training objective and sampling, not the model.
    """

    def __init__(self, *args, step_size: float = 0.01, adjustment_c: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size = step_size
        self.adjustment_c = adjustment_c
        self.history_: DROHistory | None = None

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> "GroupDROMLP":
        if groups is None:
            raise ValueError("GroupDROMLP.fit needs groups: one integer group id per row")
        torch.manual_seed(self.seed)
        self._net = self._build_net()
        cat_x, num_x = self._to_tensors(X)
        y_binary = torch.tensor((y == POSITIVE_LABEL).to_numpy(dtype="float32"), dtype=torch.float32)
        weights = (
            torch.tensor(sample_weight, dtype=torch.float32)
            if sample_weight is not None
            else torch.ones_like(y_binary)
        )
        self.history_ = train_group_dro(
            self._net,
            (cat_x, num_x),
            y_binary,
            weights,
            torch.as_tensor(groups, dtype=torch.long),
            step_size=self.step_size,
            adjustment_c=self.adjustment_c,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        return self
