import numpy as np
import pandas as pd
import pytest
from src.features.stationary import UnivariateStationaryFeaturizer
from src.features.passthrough import RawPassthroughFeaturizer


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_stationary_featurizer_round_trip(raw_series):
    featurizer = UnivariateStationaryFeaturizer(log=True, diff_lags=[1, 12])
    transformed = featurizer.fit_transform(raw_series)
    recovered = featurizer.inverse_transform(transformed)
    assert np.allclose(recovered.values, raw_series.loc[recovered.index].values, atol=1e-6)


def test_stationary_featurizer_inverse_before_fit_raises():
    featurizer = UnivariateStationaryFeaturizer(log=True, diff_lags=[1, 12])
    with pytest.raises(RuntimeError):
        featurizer.inverse_transform(pd.Series([1, 2, 3]))


def test_passthrough_featurizer_is_identity(raw_series):
    featurizer = RawPassthroughFeaturizer()
    transformed = featurizer.fit_transform(raw_series)
    recovered = featurizer.inverse_transform(transformed)
    assert transformed.equals(raw_series)
    assert recovered.equals(raw_series)


def test_passthrough_featurizer_returns_copies_not_references(raw_series):
    """Mutating the output shouldn't corrupt the caller's original series."""
    featurizer = RawPassthroughFeaturizer()
    transformed = featurizer.fit_transform(raw_series)
    transformed.iloc[0] = -999
    assert raw_series.iloc[0] != -999