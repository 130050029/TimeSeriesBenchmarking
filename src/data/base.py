"""
Base interface every data source must implement.

Design goal: src/models and src/evaluation should never know or care
whether data came from a local CSV, Databricks, or Snowflake. They only
ever call `load()` and get back a pandas Series/DataFrame with a
DatetimeIndex. Swapping sources should be a one-line config change.
"""

from abc import ABC, abstractmethod
import pandas as pd


class DataSource(ABC):
    """Every concrete data source (LocalCSV, Databricks, Snowflake) subclasses this."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Return a DataFrame with:
          - a DatetimeIndex (parsed, sorted ascending)
          - one or more numeric columns (the series to forecast)

        Implementations should NOT do any modeling-specific preprocessing
        here (no differencing, no log transform) -- that belongs in
        src/preprocessing. This layer's only job is: get the raw data,
        in a consistent shape, regardless of where it came from.
        """
        raise NotImplementedError

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared sanity checks every source should run on its output."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{self.__class__.__name__}: index must be DatetimeIndex, got {type(df.index)}")
        if df.isnull().all().any():
            raise ValueError(f"{self.__class__.__name__}: one or more columns are entirely null")
        return df.sort_index()