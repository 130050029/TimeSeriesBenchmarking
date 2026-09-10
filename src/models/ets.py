"""
ETS / Holt-Winters exponential smoothing. Pair with RawPassthroughFeaturizer --
unlike SARIMA, this model handles trend and seasonality natively (via
trend="mul"/seasonal="mul" or "add") and doesn't need log/diff preprocessing.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import ForecastModel


class ETSModel(ForecastModel):
    def __init__(self, trend: str = "add", seasonal: str = "add", seasonal_periods: int = 12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fit_result = None

    def fit(self, features: pd.Series) -> "ETSModel":
        model = ExponentialSmoothing(
            features, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods,
        )
        self._fit_result = model.fit()
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._fit_result is None:
            raise RuntimeError("Call fit() before predict()")
        return self._fit_result.forecast(horizon)