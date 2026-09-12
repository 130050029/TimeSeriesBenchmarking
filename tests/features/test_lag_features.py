import numpy as np
import pandas as pd
import pytest
from src.features.lag_features import LagFeatureFeaturizer, build_lag_feature_row


def test_build_lag_feature_row_ordering():
    recent = np.array([10, 20, 30])
    row = build_lag_feature_row(recent, month=6)
    assert list(row) == [30, 20, 10, 6]


def test_lag_featurizer_output_shapes(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    featurizer = LagFeatureFeaturizer(n_lags=13)
    features = featurizer.fit_transform(df)

    assert features["X"].shape == (len(df) - 13, 14)
    assert features["y"].shape == (len(df) - 13,)
    assert len(features["history_tail"]) == 13


def test_lag_featurizer_inverse_is_identity():
    featurizer = LagFeatureFeaturizer()
    output = pd.Series([1.0, 2.0, 3.0])
    recovered = featurizer.inverse_transform(output)
    assert recovered.equals(output)