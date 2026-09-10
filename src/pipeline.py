"""
Composes a Featurizer + a ForecastModel into one callable unit -- the
sklearn Pipeline idiom, applied to time series. This is what evaluation
code calls; it never touches Featurizer or Model directly.
"""

import pandas as pd
from src.features.base import Featurizer
from src.models.base import ForecastModel


class ForecastPipeline:
    def __init__(self, featurizer: Featurizer, model: ForecastModel):
        self.featurizer = featurizer
        self.model = model

    def fit(self, raw: pd.Series) -> "ForecastPipeline":
        features = self.featurizer.fit_transform(raw)
        self.model.fit(features)
        return self

    def predict(self, horizon: int) -> pd.Series:
        raw_output = self.model.predict(horizon)
        return self.featurizer.inverse_transform(raw_output)