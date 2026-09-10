"""
Naive baseline -- repeats the last observed value for every future step.
This isn't a throwaway: every real forecasting benchmark needs a trivial
floor to compare against. If a fancy model can't beat this, that's a real
finding, not an embarrassment to hide.
"""

import pandas as pd
from .base import ForecastModel


class NaiveLastValueModel(ForecastModel):
    def fit(self, features: pd.Series) -> "NaiveLastValueModel":
        if len(features) == 0:
            raise ValueError("Cannot fit on empty series")
        self._last_value = features.iloc[-1]
        self._last_date = features.index[-1]
        self._freq = features.index.freq
        return self

    def predict(self, horizon: int) -> pd.Series:
        future_index = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        return pd.Series([self._last_value] * horizon, index=future_index)