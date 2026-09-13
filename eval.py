"""
The real benchmark -- walk-forward validation across all models, multiple
folds. main.py is a quick single-holdout smoke test; this is what you'd
actually cite as "how good is each model."

Run from the repo root:
    python eval.py
"""

from src.data.local_csv import LocalCSVSource
from src.features.stationary import UnivariateStationaryFeaturizer
from src.features.passthrough import RawPassthroughFeaturizer
from src.features.lag_features import LagFeatureFeaturizer
from src.models.naive import NaiveLastValueModel
from src.models.sarima import SARIMAModel
from src.models.ets import ETSModel
from src.models.xgboost_model import XGBoostModel
from src.pipeline import ForecastPipeline
from src.evaluation.walk_forward import walk_forward_validate, summarize
from src.features.windowed import WindowedFeaturizer
from src.models.lstm_model import LSTMModel

def main():
    source = LocalCSVSource(
        filepath="data/local/airpassengers.csv", date_col="Month", value_cols=["Passengers"], freq="MS"
    )
    series = source.load()["Passengers"]

    factories = {
        "Naive": lambda: ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel()),
        "SARIMA(1,1,1)(1,1,1,12)": lambda: ForecastPipeline(
            UnivariateStationaryFeaturizer(log=True, diff_lags=[]),
            SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
        ),
        "ETS (Holt-Winters, mul)": lambda: ForecastPipeline(
            RawPassthroughFeaturizer(), ETSModel(trend="mul", seasonal="mul", seasonal_periods=12)
        ),
        "XGBoost (lag features)": lambda: ForecastPipeline(
            LagFeatureFeaturizer(n_lags=13), XGBoostModel(n_estimators=200, max_depth=3, learning_rate=0.05)
        ),
        "LSTM (direct multi-output)": lambda: ForecastPipeline(
            WindowedFeaturizer(window_size=12, horizon=12), LSTMModel(hidden_size=4)
        ),
    }

    horizon, n_folds = 12, 4
    print(f"Walk-forward validation: {n_folds} folds, {horizon}-month horizon each\n")
    results = walk_forward_validate(series, factories, horizon=horizon, n_folds=n_folds)

    print("Per-fold results:")
    print(results.pivot(index="fold_end", columns="model", values="mape").round(2).to_string())

    print(f"\nSummary across all {n_folds} folds (sorted best-first):")
    print(summarize(results).round(2).to_string())


if __name__ == "__main__":
    main()