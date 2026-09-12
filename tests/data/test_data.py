import pandas as pd
import pytest
from src.data.local_csv import LocalCSVSource


def test_local_csv_loads_correctly(airpassengers_csv):
    source = LocalCSVSource(
        filepath=airpassengers_csv,
        date_col="Month",
        value_cols=["Passengers"],
        freq="MS",
    )
    df = source.load()

    assert isinstance(df.index, pd.DatetimeIndex)
    assert "Passengers" in df.columns
    assert len(df) == 144  # known length of AirPassengers dataset
    assert df["Passengers"].iloc[0] == 112  # first known value, catches silent parsing bugs


def test_local_csv_missing_file_raises():
    source = LocalCSVSource(filepath="data/local/does_not_exist.csv", date_col="Month", value_cols=["Passengers"])
    with pytest.raises(FileNotFoundError):
        source.load()