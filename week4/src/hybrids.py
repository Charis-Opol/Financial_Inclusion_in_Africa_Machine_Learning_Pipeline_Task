"""H2: a small head trained on TabPFN embeddings, with ERM, reweighting or Group DRO.

Embeddings for one outer fold are made once and saved, so the three heads share exactly the same
representation, and head tuning does not rerun TabPFN.
  - Outer training rows get out-of-fold embeddings: each row is embedded by a model that did not
    have it in its context (`out_of_fold_embeddings`).
  - Outer validation rows are embedded with a context drawn from the whole outer training fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fin_inclusion.data.loader import TARGET_COLUMN
from group_dro import DROHistory, train_group_dro
from harness import Prepared, RunConfig, country_label_key, prepare_matrix
from tabpfn_tools import embed_new_rows, make_v2, out_of_fold_embeddings


class EmbeddingHead(nn.Module):
    """One hidden layer on top of an embedding. Small on purpose: the representation is TabPFN's."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def embedding_cache_path(config: RunConfig, design: str, seed: int, fold_name: str) -> Path:
    # The estimator count is in the path, so embeddings made with different counts never mix.
    return config.cache_dir / "h2_embeddings" / f"est{config.h2_estimators}" / design / f"seed{seed}" / f"{fold_name}.npz"


def fold_embeddings(
    train: pd.DataFrame, val: pd.DataFrame, use_country: bool, config: RunConfig, seed: int, design: str, fold_name: str
) -> dict[int, int] | dict:
    """Loads one outer fold's embeddings, making and saving them first if needed."""
    path = embedding_cache_path(config, design, seed, fold_name)
    if not path.exists():
        prepared = prepare_matrix(train, val, use_country)
        strata = country_label_key(train)
        categorical = list(range(len(prepared.categorical_columns)))

        def make_model(lists):
            return make_v2(
                device=config.device, seed=seed, n_estimators=config.h2_estimators,
                context_lists=lists, categorical_indices=categorical,
            )

        shared = dict(n_estimators=config.h2_estimators, max_rows=config.context_rows, seed=seed)
        train_vectors = out_of_fold_embeddings(prepared.X_train, prepared.y_binary, strata, make_model, **shared)
        val_vectors = embed_new_rows(
            prepared.X_train, prepared.y_binary, prepared.X_val, strata, make_model, **shared
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary,
            row_ids=np.concatenate([train.index.to_numpy(), val.index.to_numpy()]),
            vectors=np.vstack([train_vectors, val_vectors]),
        )
        temporary.replace(path)
    stored = np.load(path)
    return {
        "positions": {int(row): i for i, row in enumerate(stored["row_ids"])},
        "vectors": stored["vectors"],
    }


def prepare_embeddings(train: pd.DataFrame, val: pd.DataFrame, context: dict) -> Prepared:
    """Looks rows up by id, and standardises using the training part of this split only."""
    positions, vectors = context["positions"], context["vectors"]
    X_train = vectors[[positions[int(row)] for row in train.index]]
    X_val = vectors[[positions[int(row)] for row in val.index]]
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    return Prepared(
        X_train=((X_train - mean) / std).astype(np.float32),
        X_val=((X_val - mean) / std).astype(np.float32),
        y_train=train[TARGET_COLUMN],
        country_train=train["country"],
    )


def fit_head(
    prepared: Prepared,
    sample_weight: np.ndarray,
    params: dict,
    epochs: int,
    seed: int,
    groups: np.ndarray | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, DROHistory | None]:
    """Trains a head and scores validation rows. `groups` switches from plain training to Group DRO."""
    torch.manual_seed(seed)
    X = torch.tensor(prepared.X_train, dtype=torch.float32)
    y = torch.tensor(prepared.y_binary, dtype=torch.float32)
    weights = torch.tensor(sample_weight, dtype=torch.float32)
    net = EmbeddingHead(X.shape[1], params["hidden_dim"], params["dropout"])
    history = None
    if groups is not None:
        history = train_group_dro(
            net, (X,), y, weights, torch.as_tensor(groups, dtype=torch.long),
            step_size=params["step_size"], adjustment_c=params["adjustment_c"],
            lr=params["lr"], weight_decay=params["weight_decay"],
            epochs=epochs, batch_size=batch_size, seed=seed,
        )
    else:
        # The same loop as the baseline PyTorchMLP.fit, for a like-for-like ERM head.
        loader = DataLoader(
            TensorDataset(X, y, weights), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        net.train()
        for _ in range(epochs):
            for x_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                (loss_fn(net(x_batch), y_batch) * w_batch).mean().backward()
                optimizer.step()
    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(torch.tensor(prepared.X_val, dtype=torch.float32))).numpy()
    return scores, history
