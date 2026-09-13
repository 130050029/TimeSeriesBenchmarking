import pandas as pd
import pytest
from src.features.windowed import WindowedFeaturizer
from src.models.rope_patch_model import RoPEPatchModel
from src.pipeline import ForecastPipeline


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_rope_patch_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        RoPEPatchModel().predict(horizon=12)


def test_rope_patch_rejects_indivisible_window_size(raw_series):
    pipeline = ForecastPipeline(WindowedFeaturizer(window_size=10, horizon=12), RoPEPatchModel(patch_len=4))
    with pytest.raises(ValueError):
        pipeline.fit(raw_series[:-12])


def test_rope_patch_wrong_horizon_raises(raw_series):
    pipeline = ForecastPipeline(WindowedFeaturizer(window_size=12, horizon=12), RoPEPatchModel(max_epochs=50))
    pipeline.fit(raw_series[:-12])
    with pytest.raises(ValueError):
        pipeline.predict(horizon=6)


def test_rope_patch_pipeline_produces_valid_forecast(raw_series):
    train = raw_series[:-12]
    pipeline = ForecastPipeline(WindowedFeaturizer(window_size=12, horizon=12), RoPEPatchModel(patch_len=4, d_model=16, max_epochs=150))
    pipeline.fit(train)
    forecast = pipeline.predict(horizon=12)

    assert len(forecast) == 12
    assert forecast.index[0] > train.index[-1]
    assert forecast.min() > 0
    assert forecast.max() < train.max() * 3