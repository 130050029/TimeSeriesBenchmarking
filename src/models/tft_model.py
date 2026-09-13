"""
Temporal Fusion Transformer, via pytorch-forecasting. Deliberately tiny
and minimally trained (a few batches, not real epochs) -- inference-focused
per the project's compute constraints, not tuned for competitive accuracy.

predict() returns the MEDIAN (q=0.5) forecast, matching every other
model's plain point-forecast interface so this slots into eval.py
unchanged. Use predict_quantiles() for the full distribution TFT actually
produces -- that's the real point of this model, not the point forecast.
"""

import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from .base import ForecastModel


class TFTModel(ForecastModel):
    def __init__(self, hidden_size: int = 8, attention_head_size: int = 1,
                 max_encoder_length: int = 12, max_prediction_length: int = 12,
                 limit_train_batches: int = 3, max_epochs: int = 1):
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.limit_train_batches = limit_train_batches
        self.max_epochs = max_epochs
        self._tft = None

    def fit(self, features: dict) -> "TFTModel":
        self._df = features["df"]
        self._last_date = features["last_date"]
        self._freq = features["freq"]

        self._training_dataset = TimeSeriesDataSet(
            self._df,
            time_idx="time_idx",
            target="value",
            group_ids=["series_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["series_id"],
            time_varying_known_categoricals=["month"],
            time_varying_unknown_reals=["value"],
            target_normalizer=GroupNormalizer(groups=["series_id"]),
        )
        train_dataloader = self._training_dataset.to_dataloader(train=True, batch_size=16, num_workers=0)

        self._tft = TemporalFusionTransformer.from_dataset(
            self._training_dataset,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=0.1,
            loss=QuantileLoss(),
            log_interval=0,
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs, limit_train_batches=self.limit_train_batches,
            enable_progress_bar=False, enable_model_summary=False, logger=False,
        )
        trainer.fit(self._tft, train_dataloaders=train_dataloader)
        return self

    def _build_future_df(self, horizon: int) -> pd.DataFrame:
        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        future_df = pd.DataFrame({
            "value": [0.0] * horizon,
            "time_idx": range(len(self._df), len(self._df) + horizon),
            "series_id": "series_0",
            "month": [str(d.month) for d in future_dates],
        })
        future_df["month"] = future_df["month"].astype("category")
        return pd.concat([self._df, future_df], ignore_index=True)

    def predict_quantiles(self, horizon: int) -> pd.DataFrame:
        if self._tft is None:
            raise RuntimeError("Call fit() before predict()")
        full_df = self._build_future_df(horizon)
        predict_dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, full_df, predict=True, stop_randomization=True)
        predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        quantile_preds = self._tft.predict(predict_dataloader, mode="quantiles", return_x=False)
        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        quantiles = self._tft.loss.quantiles
        return pd.DataFrame(quantile_preds[0].numpy(), index=future_dates, columns=[f"q{q}" for q in quantiles])

    def predict(self, horizon: int) -> pd.Series:
        quantile_df = self.predict_quantiles(horizon)
        median_col = [c for c in quantile_df.columns if c == "q0.5"][0]
        return quantile_df[median_col].rename(None)