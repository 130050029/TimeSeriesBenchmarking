"""
SARIMA, wrapped behind ForecastModel. Pair this with a Featurizer that does
ONLY log transform (diff_lags=[]) -- differencing is handled internally by
SARIMAX via the (d, D) terms in order/seasonal_order. Feeding it an already-
differenced series here would silently double-difference.
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import ForecastModel


class SARIMAModel(ForecastModel):
    def __init__(self, order: tuple, seasonal_order: tuple = (0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fit_result = None

    def fit(self, features: pd.Series) -> "SARIMAModel":
        model = SARIMAX(
            features, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        self._fit_result = model.fit(disp=False)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._fit_result is None:
            raise RuntimeError("Call fit() before predict()")
        return self._fit_result.forecast(steps=horizon)