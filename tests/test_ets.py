import pandas as pd
import pytest
from src.features.passthrough import RawPassthroughFeaturizer
from src.models.ets import ETSModel
from src.pipeline import ForecastPipeline


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_ets_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ETSModel().predict(horizon=3)


def test_ets_pipeline_beats_naive_on_holdout(raw_series):
    from src.models.naive import NaiveLastValueModel

    train, test = raw_series[:-12], raw_series[-12:]

    ets_pipeline = ForecastPipeline(
        RawPassthroughFeaturizer(), ETSModel(trend="mul", seasonal="mul", seasonal_periods=12)
    )
    ets_pipeline.fit(train)
    ets_forecast = ets_pipeline.predict(horizon=12)

    naive_pipeline = ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel())
    naive_pipeline.fit(train)
    naive_forecast = naive_pipeline.predict(horizon=12)

    ets_mape = (abs(ets_forecast.values - test.values) / test.values).mean()
    naive_mape = (abs(naive_forecast.values - test.values) / test.values).mean()

    assert ets_mape < naive_mape
    assert ets_mape < 0.06  # ~4.3% expected; margin for version drift