"""PyTorch MLP with categorical embeddings -- the deep learning variant (README §5).

`hidden_dims` controls depth directly (2 entries = Phase 3's 2-layer
baseline; 3 entries = Phase 4's depth-ablation arm) so the 3-layer variant
needs no new class, per Open/Closed.

Consumes the PyTorch preprocessing branch's output as-is (`preprocessing/encoders.py`):
categorical columns already integer-indexed (0 = unknown, per Phase 2.5),
numeric columns already scaled (Phase 2.6).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fin_inclusion.models.base_model import BaseModel

POSITIVE_LABEL = "Yes"


class _MLPNet(nn.Module):
    def __init__(
        self,
        categorical_columns: list[str],
        vocab_sizes: dict[str, int],
        n_numeric: int,
        hidden_dims: tuple[int, ...],
        embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.categorical_columns = categorical_columns
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_sizes[col], embedding_dim) for col in categorical_columns]
        )

        input_dim = len(categorical_columns) * embedding_dim + n_numeric
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        parts = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        if num_x.shape[1] > 0:
            parts.append(num_x)
        x = torch.cat(parts, dim=1)
        return self.mlp(x).squeeze(-1)


class PyTorchMLP(BaseModel):
    def __init__(
        self,
        categorical_columns: list[str],
        vocab_sizes: dict[str, int],
        numeric_columns: list[str],
        hidden_dims: tuple[int, ...] = (64, 32),
        embedding_dim: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 256,
        seed: int = 42,
    ):
        self.categorical_columns = categorical_columns
        self.vocab_sizes = vocab_sizes
        self.numeric_columns = numeric_columns
        self.hidden_dims = tuple(hidden_dims)
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._net: _MLPNet | None = None

    def _build_net(self) -> _MLPNet:
        return _MLPNet(
            categorical_columns=self.categorical_columns,
            vocab_sizes=self.vocab_sizes,
            n_numeric=len(self.numeric_columns),
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        )

    def _to_tensors(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        cat_x = torch.tensor(X[self.categorical_columns].to_numpy(dtype="int64"), dtype=torch.long)
        num_x = torch.tensor(X[self.numeric_columns].to_numpy(dtype="float32"), dtype=torch.float32)
        return cat_x, num_x

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "PyTorchMLP":
        torch.manual_seed(self.seed)
        self._net = self._build_net()

        cat_x, num_x = self._to_tensors(X)
        y_binary = torch.tensor((y == POSITIVE_LABEL).to_numpy(dtype="float32"), dtype=torch.float32)
        weights = (
            torch.tensor(sample_weight, dtype=torch.float32)
            if sample_weight is not None
            else torch.ones_like(y_binary)
        )

        loader = DataLoader(
            TensorDataset(cat_x, num_x, y_binary, weights),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self._net.train()
        for _ in range(self.epochs):
            for cat_batch, num_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                logits = self._net(cat_batch, num_batch)
                loss = (loss_fn(logits, y_batch) * w_batch).mean()
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._net is not None, "call fit() before predict_proba()"
        cat_x, num_x = self._to_tensors(X)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(cat_x, num_x)
        return torch.sigmoid(logits).numpy()

    def get_params(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def save(self, path: Path) -> None:
        assert self._net is not None, "call fit() before save()"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "categorical_columns": self.categorical_columns,
                "vocab_sizes": self.vocab_sizes,
                "numeric_columns": self.numeric_columns,
                "hidden_dims": self.hidden_dims,
                "embedding_dim": self.embedding_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PyTorchMLP":
        checkpoint = torch.load(path, weights_only=False)
        instance = cls(
            categorical_columns=checkpoint["categorical_columns"],
            vocab_sizes=checkpoint["vocab_sizes"],
            numeric_columns=checkpoint["numeric_columns"],
            hidden_dims=checkpoint["hidden_dims"],
            embedding_dim=checkpoint["embedding_dim"],
            dropout=checkpoint["dropout"],
            lr=checkpoint["lr"],
            epochs=checkpoint["epochs"],
            batch_size=checkpoint["batch_size"],
            seed=checkpoint["seed"],
        )
        instance._net = instance._build_net()
        instance._net.load_state_dict(checkpoint["state_dict"])
        return instance
