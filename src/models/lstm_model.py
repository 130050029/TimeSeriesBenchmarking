"""
LSTM, direct multi-output. One forward pass predicts the WHOLE horizon --
no recursive feedback loop, unlike XGBoostModel. Trade-off: horizon is
fixed at training time (baked into the final Linear layer's output size).
predict() will raise if asked for a different horizon than it was trained on.

Uses early stopping with a HARD epoch ceiling (not just patience) -- on
constrained hardware, average-case speed isn't enough, worst-case matters.
Also supports save()/load() checkpointing, since training has real cost
here unlike the other models in this project (SARIMA/ETS/XGBoost fit in
milliseconds, so checkpointing them saves little).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .base import ForecastModel


class LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (final_hidden, _) = self.lstm(x)
        final_hidden = final_hidden.squeeze(0)
        return self.output_layer(final_hidden)


class LSTMModel(ForecastModel):
    def __init__(self, hidden_size: int = 16, max_epochs: int = 150, learning_rate: float = 0.02,
                 patience: int = 8, min_delta: float = 1e-3):
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        self._net = None

    def fit(self, features: dict) -> "LSTMModel":
        self._horizon = features["horizon"]
        self._window_size = features["window_size"]
        self._history_tail = features["history_tail"]
        self._last_date = features["last_date"]
        self._freq = features["freq"]

        X = torch.tensor(features["X"]).unsqueeze(-1)
        y = torch.tensor(features["y"])

        self._net = LSTMNet(hidden_size=self.hidden_size, horizon=self._horizon)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        epochs_without_improvement = 0

        self._net.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            predictions = self._net(X)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if best_loss - current_loss > self.min_delta:
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        return self

    def save(self, path: str) -> None:
        if self._net is None:
            raise RuntimeError("Nothing to save -- call fit() first")
        torch.save({
            "state_dict": self._net.state_dict(),
            "hidden_size": self.hidden_size,
            "horizon": self._horizon,
            "window_size": self._window_size,
            "history_tail": self._history_tail,
            "last_date": self._last_date,
            "freq": self._freq,
        }, path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        checkpoint = torch.load(path, weights_only=False)
        model = cls(hidden_size=checkpoint["hidden_size"])
        model._horizon = checkpoint["horizon"]
        model._window_size = checkpoint["window_size"]
        model._history_tail = checkpoint["history_tail"]
        model._last_date = checkpoint["last_date"]
        model._freq = checkpoint["freq"]
        model._net = LSTMNet(hidden_size=checkpoint["hidden_size"], horizon=checkpoint["horizon"])
        model._net.load_state_dict(checkpoint["state_dict"])
        model._net.eval()
        return model

    def predict(self, horizon: int) -> pd.Series:
        if self._net is None:
            raise RuntimeError("Call fit() before predict()")
        if horizon != self._horizon:
            raise ValueError(
                f"This model was trained for horizon={self._horizon} (direct multi-output, "
                f"fixed at training time) -- cannot predict horizon={horizon}."
            )

        self._net.eval()
        with torch.no_grad():
            x = torch.tensor(self._history_tail, dtype=torch.float32).reshape(1, self._window_size, 1)
            scaled_forecast = self._net(x).squeeze(0).numpy()

        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        return pd.Series(scaled_forecast, index=future_dates)