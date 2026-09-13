"""
Featurizer for LSTM: sliding-window sequences (NOT flat lag vectors) plus
scaling. Neural nets are scale-sensitive (unlike trees), so this featurizer
also normalizes -- fit on training data's mean/std, apply the same
transform to any future data, and inverse_transform undoes it on output.

Built for DIRECT multi-output forecasting: each training example's target
is the FULL horizon-length vector of future values, not a single next
step -- so the model predicts the whole horizon in one forward pass, with
no recursive feedback loop (and no compounding-error risk from that, unlike
the XGBoost model). The real trade-off, as discussed: horizon is baked in
at training time, not flexible per predict() call like SARIMA/ETS/XGBoost.
"""

import numpy as np
import pandas as pd
from .base import Featurizer


class WindowedFeaturizer(Featurizer):
    def __init__(self, window_size: int = 12, horizon: int = 12):
        self.window_size = window_size
        self.horizon = horizon
        self._mean = None
        self._std = None

    def fit_transform(self, raw: pd.Series) -> dict:
        self._mean = raw.mean()
        self._std = raw.std()
        scaled = (raw.values - self._mean) / self._std

        X, y = [], []
        for i in range(self.window_size, len(scaled) - self.horizon + 1):
            X.append(scaled[i - self.window_size:i])
            y.append(scaled[i:i + self.horizon])

        return {
            "X": np.array(X, dtype=np.float32),
            "y": np.array(y, dtype=np.float32),
            "history_tail": scaled[-self.window_size:],
            "last_date": raw.index[-1],
            "freq": raw.index.freq,
            "window_size": self.window_size,
            "horizon": self.horizon,
        }

    def inverse_transform(self, model_output: pd.Series) -> pd.Series:
        if self._mean is None:
            raise RuntimeError("fit_transform must be called before inverse_transform")
        real_scale = model_output.values * self._std + self._mean
        return pd.Series(real_scale, index=model_output.index)