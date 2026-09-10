import numpy as np
import pandas as pd
import pytest
from src.preprocessing.transforms import (
    log_transform, inverse_log_transform, difference, inverse_difference, TransformPipeline
)
from src.preprocessing.stationarity import check_stationarity


@pytest.fixture
def log_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv)
    return np.log(df["Passengers"])


def test_log_transform_round_trip(log_series):
    recovered = inverse_log_transform(log_series)
    original = np.exp(log_series)  # sanity: compare against a fresh exp, not circular
    assert np.allclose(recovered, original)


def test_log_transform_rejects_non_positive():
    with pytest.raises(ValueError):
        log_transform(pd.Series([1, 2, -1, 4]))


def test_single_difference_round_trip(log_series):
    diffed = difference(log_series, lag=1)
    recovered = inverse_difference(diffed, base=log_series, lag=1)
    assert np.allclose(recovered.values, log_series.loc[recovered.index].values)


def test_chained_difference_round_trip(log_series):
    """
    This is the exact case that caught a real seed-alignment bug during
    development -- chaining ordinary diff(1) then seasonal diff(12) and
    inverting both. Guards against regressing to the buggy positional version.
    """
    pipeline = TransformPipeline(log=False, diff_lags=[1, 12])
    stationary = pipeline.fit_transform(log_series)
    recovered = pipeline.inverse_transform(stationary)
    assert np.allclose(recovered.values, log_series.loc[recovered.index].values, atol=1e-8)


def test_full_pipeline_log_and_seasonal_diff(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv)
    raw = df["Passengers"]

    pipeline = TransformPipeline(log=True, diff_lags=[1, 12])
    stationary = pipeline.fit_transform(raw)
    recovered = pipeline.inverse_transform(stationary)

    assert np.allclose(recovered.values, raw.loc[recovered.index].values, atol=1e-6)


def test_stationarity_confirms_raw_series_is_not_stationary(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv)
    result = check_stationarity(df["Passengers"])
    assert result.is_stationary is False
    assert result.p_value > 0.9  # matches what we found manually earlier in this project


def test_stationarity_confirms_fully_transformed_series_is_stationary(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv)
    pipeline = TransformPipeline(log=True, diff_lags=[1, 12])
    stationary = pipeline.fit_transform(df["Passengers"])
    result = check_stationarity(stationary)
    assert result.is_stationary is True