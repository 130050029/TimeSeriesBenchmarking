import numpy as np
import pandas as pd
import pytest
from src.features.multivariate import MultivariateDifferenceFeaturizer


@pytest.fixture
def two_column_series():
    dates = pd.date_range("2020-01-01", periods=20, freq="QS")
    return pd.DataFrame({
        "a": np.linspace(10, 30, 20) + np.random.RandomState(0).normal(0, 0.5, 20),
        "b": np.linspace(100, 50, 20) + np.random.RandomState(1).normal(0, 0.5, 20),
    }, index=dates)


def test_multivariate_round_trip(two_column_series):
    featurizer = MultivariateDifferenceFeaturizer()
    transformed = featurizer.fit_transform(two_column_series)
    recovered = featurizer.inverse_transform(transformed)
    assert np.allclose(recovered.values, two_column_series.loc[recovered.index].values, atol=1e-8)


def test_multivariate_inverse_before_fit_raises():
    featurizer = MultivariateDifferenceFeaturizer()
    with pytest.raises(RuntimeError):
        featurizer.inverse_transform(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_multivariate_preserves_column_order(two_column_series):
    featurizer = MultivariateDifferenceFeaturizer()
    transformed = featurizer.fit_transform(two_column_series)
    assert list(transformed.columns) == ["a", "b"]