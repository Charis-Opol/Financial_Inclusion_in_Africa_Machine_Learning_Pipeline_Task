"""Group DRO unit tests. The first three are the ones the brief requires."""
import math

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from group_dro import GroupBalancedBatches, GroupWeights, group_mean_losses, train_group_dro


def _small_net(seed: int, n_features: int = 3) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, 1), nn.Flatten(0))


def _toy_data(seed: int = 0, n: int = 240):
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, generator=generator)
    y = (X[:, 0] > 0).float()
    groups = torch.arange(n) % 3
    return X, y, torch.ones(n), groups


def test_step_size_zero_matches_erm_with_uniform_group_weights():
    """Required. With step size 0, Group DRO is plain training on the uniform group-weighted loss."""
    X, y, weight, groups = _toy_data()
    seed, lr, epochs, batch_size = 3, 0.01, 4, 30

    dro_net = _small_net(seed)
    train_group_dro(dro_net, (X,), y, weight, groups, step_size=0.0, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)

    # An independent ERM loop: the same batches and optimiser, with fixed uniform weights and no q.
    erm_net = _small_net(seed)
    optimizer = torch.optim.Adam(erm_net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    batches = GroupBalancedBatches(groups, batch_size, torch.Generator().manual_seed(seed))
    uniform = torch.full((3,), 1.0 / 3.0)
    for _ in range(epochs):
        for index in batches:
            optimizer.zero_grad()
            losses = group_mean_losses(loss_fn(erm_net(X[index]), y[index]) * weight[index], groups[index], 3)
            (uniform * losses).sum().backward()
            optimizer.step()

    for dro_param, erm_param in zip(dro_net.parameters(), erm_net.parameters()):
        assert torch.equal(dro_param, erm_param)


def test_group_weights_stay_non_negative_and_sum_to_one():
    """Required. Checked during real training and under extreme losses."""
    X, y, weight, groups = _toy_data()
    history = train_group_dro(
        _small_net(0), (X,), y, weight, groups, step_size=1.0, adjustment_c=2.0, lr=0.01, epochs=5, batch_size=30, seed=0
    )
    for q in history.q_by_epoch:
        assert min(q) >= 0
        assert math.isclose(sum(q), 1.0, abs_tol=1e-5)

    weights = GroupWeights([100, 100, 10], step_size=5.0)
    for losses in ([100.0, 0.0, 0.0], [0.0, 50.0, 1e-6], [3.0, 3.0, 3.0]):
        q = weights.update(torch.tensor(losses))
        assert (q >= 0).all()
        assert math.isclose(q.sum().item(), 1.0, abs_tol=1e-5)


def _minority_group_data(seed: int, n_major: int = 900, n_minor: int = 100):
    """In the majority group the label follows x0. In the minority group it follows x1 instead."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_major + n_minor, 2)).astype(np.float32)
    y = np.concatenate([X[:n_major, 0] > 0, X[n_major:, 1] > 0]).astype(np.float32)
    groups = np.concatenate([np.zeros(n_major), np.ones(n_minor)]).astype(np.int64)
    # Group membership is visible only weakly, through this noisy feature.
    marker = (groups + 0.5 * rng.normal(size=len(groups))).astype(np.float32)
    return np.column_stack([X, marker]), y, groups


def _train_plain(net, X, y, lr, epochs, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(net(X[index]), y[index]).backward()
            optimizer.step()


def test_group_dro_raises_minority_performance_over_erm():
    """Required. Averaged over seeds, so one lucky initialisation cannot pass it."""
    lr, epochs, batch_size = 0.005, 30, 64
    erm_auc, dro_auc = [], []
    for seed in range(5):
        X, y, groups = _minority_group_data(seed)
        X_test, y_test, groups_test = _minority_group_data(seed + 100)
        X, y, groups = torch.tensor(X), torch.tensor(y), torch.tensor(groups)
        X_test, minority = torch.tensor(X_test), groups_test == 1

        erm = _small_net(seed)
        _train_plain(erm, X, y, lr, epochs, batch_size, seed)
        dro = _small_net(seed)
        train_group_dro(
            dro, (X,), y, torch.ones_like(y), groups, step_size=0.1, lr=lr, epochs=epochs, batch_size=batch_size, seed=seed
        )
        with torch.no_grad():
            erm_auc.append(roc_auc_score(y_test[minority], erm(X_test[minority]).numpy()))
            dro_auc.append(roc_auc_score(y_test[minority], dro(X_test[minority]).numpy()))
    assert np.mean(dro_auc) > np.mean(erm_auc) + 0.05, f"ERM {erm_auc}, DRO {dro_auc}"


def test_update_matches_the_formula_by_hand():
    weights = GroupWeights([4, 4], step_size=0.5)
    q = weights.update(torch.tensor([1.0, 3.0]))
    raw = np.array([0.5 * math.exp(0.5), 0.5 * math.exp(1.5)])
    assert np.allclose(q.numpy(), raw / raw.sum(), atol=1e-6)


def test_higher_loss_group_gains_weight():
    weights = GroupWeights([10, 10, 10], step_size=0.1)
    q = weights.update(torch.tensor([0.2, 0.9, 0.5]))
    assert q[1] > q[2] > q[0]


def test_adjustment_is_c_over_root_size_and_favours_small_groups():
    weights = GroupWeights([100, 4], step_size=1.0, adjustment_c=2.0)
    assert np.allclose(weights.adjustments.numpy(), [0.2, 1.0])
    q = weights.update(torch.tensor([1.0, 1.0]))
    assert q[1] > q[0]


def test_q_does_not_carry_gradients_into_the_model():
    weights = GroupWeights([10, 10], step_size=1.0)
    losses = torch.tensor([1.0, 2.0], requires_grad=True)
    assert not weights.update(losses).requires_grad
    weights.weighted_loss(losses).backward()
    assert np.allclose(losses.grad.numpy(), weights.q.numpy())


def test_every_batch_contains_every_group_equally():
    groups = torch.cat([torch.zeros(1000), torch.ones(50), torch.full((10,), 2)]).long()
    batches = GroupBalancedBatches(groups, 30, torch.Generator().manual_seed(0))
    assert batches.steps_per_epoch == math.ceil(1060 / 30)
    for index in batches:
        counts = torch.bincount(groups[index], minlength=3)
        assert counts.tolist() == [10, 10, 10]


def test_each_small_group_row_is_used_before_any_repeats():
    groups = torch.cat([torch.zeros(200), torch.ones(25)]).long()
    batches = GroupBalancedBatches(groups, 10, torch.Generator().manual_seed(0))
    taken = [batches._take(1).tolist() for _ in range(5)]
    assert len({row for chunk in taken for row in chunk}) == 25


def test_missing_group_raises_rather_than_passing_silently():
    with pytest.raises(ValueError, match="Every group"):
        group_mean_losses(torch.ones(4), torch.tensor([0, 0, 1, 1]), 3)


def test_batch_too_small_for_the_groups_raises():
    with pytest.raises(ValueError, match="too small"):
        GroupBalancedBatches(torch.tensor([0, 1, 2, 3]), 3, torch.Generator())
