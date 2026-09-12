"""
VAR (Vector AutoRegression) -- unlike every other model so far, this one
genuinely operates on multiple series at once. fit()/predict() work with
DataFrames, not Series -- ForecastModel's type hints are loose (Any) so
this is a valid, intentional variation, not a hack.
"""

import pandas as pd
from statsmodels.tsa.api import VAR
from .base import ForecastModel


class VARModel(ForecastModel):
    def __init__(self, maxlags: int = 4):
        self.maxlags = maxlags
        self._fit_result = None
        self._n_lags_used = None
        self._last_values = None

    def fit(self, features: pd.DataFrame) -> "VARModel":
        model = VAR(features)
        selected = model.select_order(maxlags=self.maxlags)  # AIC-based lag selection, not a fixed guess
        self._n_lags_used = max(selected.aic, 1)
        self._fit_result = model.fit(self._n_lags_used)
        self._last_values = features.values[-self._n_lags_used:]
        self._columns = features.columns
        self._last_index = features.index[-1]
        self._freq = features.index.freq
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self._fit_result is None:
            raise RuntimeError("Call fit() before predict()")
        forecast_values = self._fit_result.forecast(self._last_values, steps=horizon)
        future_index = pd.date_range(self._last_index, periods=horizon + 1, freq=self._freq)[1:]
        return pd.DataFrame(forecast_values, index=future_index, columns=self._columns)