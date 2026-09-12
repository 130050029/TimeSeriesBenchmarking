import pandas as pd
import pytest
from src.features.lag_features import LagFeatureFeaturizer
from src.models.xgboost_model import XGBoostModel
from src.pipeline import ForecastPipeline


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_xgboost_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        XGBoostModel().predict(horizon=3)


def test_xgboost_pipeline_produces_valid_forecast(raw_series):
    train = raw_series[:-12]
    pipeline = ForecastPipeline(LagFeatureFeaturizer(n_lags=13), XGBoostModel(n_estimators=200, max_depth=3, learning_rate=0.05))
    pipeline.fit(train)
    forecast = pipeline.predict(horizon=12)

    assert len(forecast) == 12
    assert (forecast.values > 0).all()
    assert forecast.index[0] > train.index[-1]


def test_xgboost_pipeline_beats_naive_on_holdout(raw_series):
    from src.features.passthrough import RawPassthroughFeaturizer
    from src.models.naive import NaiveLastValueModel

    train, test = raw_series[:-12], raw_series[-12:]

    xgb_pipeline = ForecastPipeline(
        LagFeatureFeaturizer(n_lags=13), XGBoostModel(n_estimators=200, max_depth=3, learning_rate=0.05)
    )
    xgb_pipeline.fit(train)
    xgb_forecast = xgb_pipeline.predict(horizon=12)

    naive_pipeline = ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel())
    naive_pipeline.fit(train)
    naive_forecast = naive_pipeline.predict(horizon=12)

    xgb_mape = (abs(xgb_forecast.values - test.values) / test.values).mean()
    naive_mape = (abs(naive_forecast.values - test.values) / test.values).mean()

    assert xgb_mape < naive_mape
    assert xgb_mape < 0.10  # ~6.5% expected; margin for version drift, since recursive rollout has some inherent variance