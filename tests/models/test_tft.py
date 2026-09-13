import pandas as pd
import pytest
from src.features.tft_features import TFTFeaturizer
from src.models.tft_model import TFTModel
from src.pipeline import ForecastPipeline


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_tft_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TFTModel().predict(horizon=12)


def test_tft_pipeline_produces_valid_forecast_shape(raw_series):
    train = raw_series[:-12]
    pipeline = ForecastPipeline(TFTFeaturizer(), TFTModel(hidden_size=8, limit_train_batches=3, max_epochs=1))
    pipeline.fit(train)
    forecast = pipeline.predict(horizon=12)

    assert len(forecast) == 12
    assert forecast.index[0] > train.index[-1]


def test_tft_predict_quantiles_shape(raw_series):
    train = raw_series[:-12]
    pipeline = ForecastPipeline(TFTFeaturizer(), TFTModel(hidden_size=8, limit_train_batches=3, max_epochs=1))
    pipeline.fit(train)
    quantiles = pipeline.model.predict_quantiles(12)

    assert len(quantiles) == 12
    assert "q0.5" in quantiles.columns