import pandas as pd
import pytest
from src.features.stationary import UnivariateStationaryFeaturizer
from src.models.sarima import SARIMAModel
from src.pipeline import ForecastPipeline


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_sarima_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).predict(horizon=3)


def test_sarima_pipeline_beats_naive_on_holdout(raw_series):
    """
    The real benchmark question: does SARIMA actually beat the trivial
    naive baseline on held-out data? This is the test that matters most --
    a model earning its place in the comparison, not just running without errors.
    """
    from src.features.passthrough import RawPassthroughFeaturizer
    from src.models.naive import NaiveLastValueModel

    train, test = raw_series[:-12], raw_series[-12:]

    sarima_pipeline = ForecastPipeline(
        UnivariateStationaryFeaturizer(log=True, diff_lags=[]),
        SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
    )
    sarima_pipeline.fit(train)
    sarima_forecast = sarima_pipeline.predict(horizon=12)

    naive_pipeline = ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel())
    naive_pipeline.fit(train)
    naive_forecast = naive_pipeline.predict(horizon=12)

    sarima_mape = (abs(sarima_forecast.values - test.values) / test.values).mean()
    naive_mape = (abs(naive_forecast.values - test.values) / test.values).mean()

    assert sarima_mape < naive_mape
    assert sarima_mape < 0.05  # ~2.3% expected; generous margin against minor version drift