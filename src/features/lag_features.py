"""
Tabular lag-feature featurizer for XGBoost. The one genuinely tricky part:
multi-step forecasting requires feeding each prediction back in as a new
lag for the next step (recursive/autoregressive rollout). To avoid
duplicating "how do I turn history into a feature row" logic between
training (here) and forecasting (models/xgboost_model.py), that logic is
a single shared pure function both places call.
"""

import numpy as np
import pandas as pd
from .base import Featurizer


def build_lag_feature_row(recent_values: np.ndarray, month: int) -> np.ndarray:
    """
    recent_values: the n_lags most recent values, ordered OLDEST to NEWEST.
    Returns [most-recent-lag, ..., oldest-lag, month] -- this exact ordering
    must match between training and forecasting, or predictions will be
    silently wrong (features fed in a different order than the model learned).
    """
    return np.append(recent_values[::-1], month)


class LagFeatureFeaturizer(Featurizer):
    def __init__(self, n_lags: int = 13):
        self.n_lags = n_lags

    def fit_transform(self, raw: pd.Series) -> dict:
        values = raw.values
        X, y = [], []
        for i in range(self.n_lags, len(values)):
            row = build_lag_feature_row(values[i - self.n_lags:i], month=raw.index[i].month)
            X.append(row)
            y.append(values[i])

        return {
            "X": np.array(X),
            "y": np.array(y),
            "history_tail": values[-self.n_lags:],
            "last_date": raw.index[-1],
            "freq": raw.index.freq,
            "n_lags": self.n_lags,
        }

    def inverse_transform(self, model_output: pd.Series) -> pd.Series:
        return model_output.copy()