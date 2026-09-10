"""
Reversible transforms for making a series stationary: log transform and
differencing (ordinary + seasonal). Every transform here has a matching
inverse, so a model can be fit on the transformed scale and forecasts
converted back to the real scale.

Inversion is done by index lookup (not positional seed-counting) --
an earlier positional version of this had a subtle off-by-N bug when
chaining multiple differencing steps, because each step shifts *where*
the correct seed value lives. Index-based lookup can't drift like that.
"""

import numpy as np
import pandas as pd


def log_transform(series: pd.Series) -> pd.Series:
    if (series <= 0).any():
        raise ValueError("log_transform requires strictly positive values")
    return np.log(series)


def inverse_log_transform(series: pd.Series) -> pd.Series:
    return np.exp(series)


def difference(series: pd.Series, lag: int = 1) -> pd.Series:
    """y_t = x_t - x_{t-lag}. Drops the first `lag` rows (no prior value to subtract)."""
    return series.diff(lag).dropna()


def inverse_difference(diffed: pd.Series, base: pd.Series, lag: int = 1) -> pd.Series:
    """
    Reconstruct the pre-differencing scale from a differenced series.

    base: the series BEFORE this differencing step was applied (same scale
          as the desired output). Must contain at least `lag` real values
          immediately preceding diffed's first index entry -- typically,
          just pass the full series you differenced from, no truncation needed.
    """
    full_index = base.index.union(diffed.index)
    result = pd.Series(index=full_index, dtype=float)
    result.loc[base.index] = base.values
    for t in diffed.index:
        pos = full_index.get_loc(t)
        if pos < lag:
            raise ValueError(f"Not enough history in `base` to invert at {t} (need {lag} prior points)")
        t_lag = full_index[pos - lag]
        result.loc[t] = result.loc[t_lag] + diffed.loc[t]
    return result.loc[diffed.index]


class TransformPipeline:
    """
    Chains log transform + one or more differencing steps (e.g. ordinary
    diff then seasonal diff), and remembers everything needed to invert
    back to the original scale -- so callers never manually track seeds.

    Example (log, then diff(1), then seasonal diff(12)):
        pipeline = TransformPipeline(log=True, diff_lags=[1, 12])
        stationary = pipeline.fit_transform(raw_series)
        # ... fit a model on `stationary`, get a forecast on that scale ...
        real_scale_forecast = pipeline.inverse_transform(forecast)
    """

    def __init__(self, log: bool = False, diff_lags: list[int] | None = None):
        self.log = log
        self.diff_lags = diff_lags or []
        self._stage_series: list[pd.Series] = []  # series BEFORE each diff step, for inversion

    def fit_transform(self, series: pd.Series) -> pd.Series:
        current = log_transform(series) if self.log else series.copy()
        self._stage_series = []
        for lag in self.diff_lags:
            self._stage_series.append(current)  # remember pre-diff state at this stage
            current = difference(current, lag)
        return current

    def inverse_transform(self, transformed: pd.Series) -> pd.Series:
        if not self.diff_lags and not self.log:
            return transformed.copy()
        current = transformed
        for lag, base in zip(reversed(self.diff_lags), reversed(self._stage_series)):
            current = inverse_difference(current, base, lag)
        return inverse_log_transform(current) if self.log else current