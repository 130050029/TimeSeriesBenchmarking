import numpy as np
import pandas as pd
import pytest
from src.features.windowed import WindowedFeaturizer


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_windowed_shapes(raw_series):
    featurizer = WindowedFeaturizer(window_size=12, horizon=12)
    features = featurizer.fit_transform(raw_series)
    n_expected = len(raw_series) - 12 - 12 + 1
    assert features["X"].shape == (n_expected, 12)
    assert features["y"].shape == (n_expected, 12)
    assert len(features["history_tail"]) == 12


def test_windowed_scaling_round_trip(raw_series):
    featurizer = WindowedFeaturizer(window_size=12, horizon=12)
    featurizer.fit_transform(raw_series)
    fake_scaled_output = pd.Series([0.0, 1.0, -1.0], index=pd.date_range("2000-01-01", periods=3, freq="MS"))
    recovered = featurizer.inverse_transform(fake_scaled_output)
    assert abs(recovered.iloc[0] - raw_series.mean()) < 1e-6


def test_windowed_inverse_before_fit_raises():
    featurizer = WindowedFeaturizer()
    with pytest.raises(RuntimeError):
        featurizer.inverse_transform(pd.Series([1, 2, 3]))