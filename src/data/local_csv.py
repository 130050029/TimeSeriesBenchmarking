"""
Fully working local data source. This is what tests and local dev run
against -- no network, no credentials, deterministic.
"""

import pandas as pd
from pathlib import Path
from .base import DataSource


class LocalCSVSource(DataSource):
    def __init__(self, filepath: str, date_col: str, value_cols: list[str], freq: str | None = None):
        """
        filepath:   path to the CSV, e.g. "data/local/airpassengers.csv"
        date_col:   name of the column holding dates, e.g. "Month"
        value_cols: which column(s) to keep as the series to forecast
        freq:       optional pandas frequency string ("MS" for month-start) --
                    if given, we reindex to enforce a clean, gap-free calendar.
        """
        self.filepath = Path(filepath)
        self.date_col = date_col
        self.value_cols = value_cols
        self.freq = freq

    def load(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(f"No CSV at {self.filepath} -- check the path in your config.")

        df = pd.read_csv(self.filepath, parse_dates=[self.date_col])
        df = df.set_index(self.date_col)[self.value_cols]

        if self.freq:
            df = df.asfreq(self.freq)  # will introduce NaN rows if the source has gaps -- surfaces data quality issues early

        return self.validate(df)