"""
XGBoost, paired with LagFeatureFeaturizer. predict() does RECURSIVE
multi-step forecasting: each prediction is appended to the rolling history
and used to build the next step's lag features -- errors can compound
step over step, which is a real, known limitation of this approach
(different from SARIMA/ETS, which forecast the whole horizon analytically).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .base import ForecastModel
from src.features.lag_features import build_lag_feature_row


class XGBoostModel(ForecastModel):
    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model = None

    def fit(self, features: dict) -> "XGBoostModel":
        self._model = XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate
        )
        self._model.fit(features["X"], features["y"])
        self._history_tail = list(features["history_tail"])
        self._last_date = features["last_date"]
        self._freq = features["freq"]
        self._n_lags = features["n_lags"]
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")

        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        history = list(self._history_tail)
        preds = []
        for date in future_dates:
            recent = np.array(history[-self._n_lags:])
            row = build_lag_feature_row(recent, month=date.month).reshape(1, -1)
            pred = self._model.predict(row)[0]
            preds.append(pred)
            history.append(pred)
        return pd.Series(preds, index=future_dates)