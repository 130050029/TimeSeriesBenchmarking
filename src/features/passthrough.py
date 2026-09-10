"""No-op featurizer -- for models (ETS) that want the raw series untouched."""

import pandas as pd
from .base import Featurizer


class RawPassthroughFeaturizer(Featurizer):
    def fit_transform(self, raw: pd.Series) -> pd.Series:
        return raw.copy()

    def inverse_transform(self, model_output: pd.Series) -> pd.Series:
        return model_output.copy()