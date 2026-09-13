import pandas as pd
import pytest
from src.features.windowed import WindowedFeaturizer
from src.models.lstm_model import LSTMModel
from src.pipeline import ForecastPipeline


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_lstm_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LSTMModel().predict(horizon=12)


def test_lstm_wrong_horizon_raises(raw_series):
    pipeline = ForecastPipeline(WindowedFeaturizer(window_size=12, horizon=12), LSTMModel(max_epochs=500))
    pipeline.fit(raw_series[:-12])
    with pytest.raises(ValueError):
        pipeline.predict(horizon=6)


def test_lstm_pipeline_produces_valid_forecast(raw_series):
    train = raw_series[:-12]
    pipeline = ForecastPipeline(WindowedFeaturizer(window_size=12, horizon=12), LSTMModel(hidden_size=32, max_epochs=500, learning_rate=0.01))
    pipeline.fit(train)
    forecast = pipeline.predict(horizon=12)

    assert len(forecast) == 12
    assert forecast.index[0] > train.index[-1]
    assert forecast.min() > 0
    assert forecast.max() < train.max() * 3

def test_lstm_save_load_round_trip(raw_series, tmp_path):
    train = raw_series[:-12]
    featurizer = WindowedFeaturizer(window_size=12, horizon=12)
    features = featurizer.fit_transform(train)

    model = LSTMModel()
    model.fit(features)
    forecast_before = model.predict(12)

    checkpoint_path = str(tmp_path / "lstm.pt")
    model.save(checkpoint_path)

    loaded = LSTMModel.load(checkpoint_path)
    forecast_after = loaded.predict(12)

    assert (forecast_before.values == forecast_after.values).all()


def test_lstm_save_before_fit_raises():
    with pytest.raises(RuntimeError):
        LSTMModel().save("/tmp/should_not_be_created.pt")