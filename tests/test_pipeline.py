import pandas as pd
import pytest
from src.features.stationary import UnivariateStationaryFeaturizer
from src.features.passthrough import RawPassthroughFeaturizer
from src.models.naive import NaiveLastValueModel
from src.pipeline import ForecastPipeline


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_pipeline_with_passthrough_featurizer(raw_series):
    """Naive model + no transform: forecast should exactly repeat the last training value."""
    pipeline = ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel())
    train = raw_series[:-12]
    pipeline.fit(train)
    forecast = pipeline.predict(horizon=12)

    assert len(forecast) == 12
    assert (forecast.values == train.iloc[-1]).all()
    assert forecast.index[0] > train.index[-1]  # forecast starts strictly after training ends


def test_pipeline_with_stationary_featurizer_produces_valid_forecast(raw_series):
    """
    Naive model doesn't need to be accurate on the transformed scale for this
    test -- we're only verifying the Featurizer<->Model<->Pipeline wiring holds:
    correct length, continuous dates, and a plausible real-scale (positive) output.
    """
    pipeline = ForecastPipeline(
        UnivariateStationaryFeaturizer(log=True, diff_lags=[1, 12]),
        NaiveLastValueModel(),
    )
    train = raw_series[:-12]
    pipeline.fit(train)
    forecast = pipeline.predict(horizon=12)

    assert len(forecast) == 12
    assert (forecast.values > 0).all()  # passenger counts can't be negative -- log/exp round-trip should preserve this
    assert forecast.index[0] > train.index[-1]