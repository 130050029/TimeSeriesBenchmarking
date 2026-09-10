import pandas as pd
import pytest
from src.models.naive import NaiveLastValueModel


def test_naive_model_repeats_last_value():
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    series = pd.Series(range(10), index=dates)  # values 0..9, last value = 9

    model = NaiveLastValueModel().fit(series)
    forecast = model.predict(horizon=3)

    assert len(forecast) == 3
    assert (forecast.values == 9).all()
    # forecast dates should continue the same monthly cadence right after training ends
    # (10 months starting Jan 2020 -> last training month is Oct 2020 -> forecast starts Nov 2020)
    expected_dates = pd.date_range("2020-11-01", periods=3, freq="MS")
    assert list(forecast.index) == list(expected_dates)


def test_naive_model_rejects_empty_series():
    with pytest.raises(ValueError):
        NaiveLastValueModel().fit(pd.Series([], dtype=float))