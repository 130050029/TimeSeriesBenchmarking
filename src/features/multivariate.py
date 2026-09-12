"""
Featurizer for multiple related series at once (VAR needs this -- it's
the one real deviation from the rest of this project, which has assumed
a single pd.Series everywhere else).
"""

import pandas as pd
from .base import Featurizer


class MultivariateDifferenceFeaturizer(Featurizer):
    """
    Ordinary (lag=1) differencing, applied independently to every column.
    Kept deliberately simple -- no log transform, no seasonal differencing --
    since quarterly macro data doesn't have the strong seasonal pattern
    AirPassengers did. Extend this if a future multivariate dataset needs more.
    """

    def fit_transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        self._base = raw.copy()  # needed to invert later
        return raw.diff().dropna()

    def inverse_transform(self, model_output: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "_base"):
            raise RuntimeError("fit_transform must be called before inverse_transform")
        full_index = self._base.index.union(model_output.index)
        result = pd.DataFrame(index=full_index, columns=self._base.columns, dtype=float)
        result.loc[self._base.index] = self._base
        for t in model_output.index:
            pos = full_index.get_loc(t)
            t_prev = full_index[pos - 1]
            result.loc[t] = result.loc[t_prev] + model_output.loc[t]
        return result.loc[model_output.index]