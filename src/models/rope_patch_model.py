"""
A small, custom Transformer for time series: patches the input window
(PatchTST-style), applies a SINGLE self-attention layer using RoPE for
positional information -- NO LSTM anywhere, unlike our other models.

This is a from-scratch build, not a library, since no verified pip-installable
RoPE time series model exists yet (checked pytorch-forecasting and
neuralforecast -- neither has one). Reuses WindowedFeaturizer unchanged,
since it needs the exact same scaled-window shape LSTM used -- good
validation that the Featurizer/Model split was designed correctly.

Direct multi-output, same strategy as LSTMModel: one forward pass predicts
the whole horizon, horizon fixed at training time.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .base import ForecastModel


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = positions[:, None] * theta[None, :]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class RoPEPatchNet(nn.Module):
    def __init__(self, patch_len: int, num_patches: int, d_model: int, horizon: int):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = num_patches
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(num_patches * d_model, horizon)

    def forward(self, x):
        batch_size = x.shape[0]
        patches = x.reshape(batch_size, self.num_patches, self.patch_len)
        embedded = self.patch_embed(patches)

        positions = torch.arange(self.num_patches, dtype=torch.float32, device=x.device)
        q = apply_rope(self.q_proj(embedded), positions)
        k = apply_rope(self.k_proj(embedded), positions)
        v = self.v_proj(embedded)

        scores = q @ k.transpose(-2, -1) / (self.d_model ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ v

        x1 = self.norm1(embedded + attended)
        x2 = self.norm2(x1 + self.ffn(x1))

        flattened = x2.reshape(batch_size, -1)
        return self.output_head(flattened)


class RoPEPatchModel(ForecastModel):
    def __init__(self, patch_len: int = 4, d_model: int = 16, max_epochs: int = 150,
                 learning_rate: float = 0.02, patience: int = 8, min_delta: float = 1e-3):
        self.patch_len = patch_len
        self.d_model = d_model
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        self._net = None

    def fit(self, features: dict) -> "RoPEPatchModel":
        window_size = features["window_size"]
        if window_size % self.patch_len != 0:
            raise ValueError(f"window_size ({window_size}) must be divisible by patch_len ({self.patch_len})")
        num_patches = window_size // self.patch_len

        self._horizon = features["horizon"]
        self._window_size = window_size
        self._history_tail = features["history_tail"]
        self._last_date = features["last_date"]
        self._freq = features["freq"]

        X = torch.tensor(features["X"])
        y = torch.tensor(features["y"])

        self._net = RoPEPatchNet(patch_len=self.patch_len, num_patches=num_patches, d_model=self.d_model, horizon=self._horizon)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_loss, epochs_without_improvement = float("inf"), 0
        self._net.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            loss = loss_fn(self._net(X), y)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if best_loss - current_loss > self.min_delta:
                best_loss, epochs_without_improvement = current_loss, 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._net is None:
            raise RuntimeError("Call fit() before predict()")
        if horizon != self._horizon:
            raise ValueError(f"Trained for horizon={self._horizon} (direct multi-output), cannot predict horizon={horizon}")

        self._net.eval()
        with torch.no_grad():
            x = torch.tensor(self._history_tail, dtype=torch.float32).reshape(1, self._window_size)
            scaled_forecast = self._net(x).squeeze(0).numpy()

        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        return pd.Series(scaled_forecast, index=future_dates)