"""
Stationarity diagnostics -- wraps the Augmented Dickey-Fuller test with
a clear, opinionated result object instead of a raw tuple of numbers.
"""

from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller
import pandas as pd


@dataclass
class StationarityResult:
    p_value: float
    test_statistic: float
    is_stationary: bool  # p_value < alpha

    def __repr__(self) -> str:
        verdict = "STATIONARY" if self.is_stationary else "NOT stationary"
        return f"<{verdict}: ADF p-value={self.p_value:.4f}, statistic={self.test_statistic:.3f}>"


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> StationarityResult:
    """
    Runs the Augmented Dickey-Fuller test.
    Convention: p_value < alpha (default 0.05) => reject the "has a unit
    root" null hypothesis => treat the series as stationary.
    """
    clean = series.dropna()
    if len(clean) < 10:
        raise ValueError("Need at least 10 non-null points to run ADF meaningfully")

    result = adfuller(clean, result_object=True)  # explicit: statsmodels is changing the default return shape in a future release
    return StationarityResult(p_value=result.pvalue, test_statistic=result.statistic, is_stationary=result.pvalue < alpha)