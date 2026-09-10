"""Wraps TransformPipeline behind the Featurizer interface -- for any model
that needs a stationary univariate series (SARIMA today)."""

import pandas as pd
from .base import Featurizer
from src.preprocessing.transforms import TransformPipeline


class UnivariateStationaryFeaturizer(Featurizer):
    def __init__(self, log: bool = False, diff_lags: list[int] | None = None):
        self.pipeline = TransformPipeline(log=log, diff_lags=diff_lags)
        self._fitted = False

    def fit_transform(self, raw: pd.Series) -> pd.Series:
        result = self.pipeline.fit_transform(raw)
        self._fitted = True
        return result

    def inverse_transform(self, model_output: pd.Series) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("fit_transform must be called before inverse_transform")
        return self.pipeline.inverse_transform(model_output)