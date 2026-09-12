"""
Human-facing entry point -- run this to see actual forecast results
printed to the terminal. This is NOT what pytest runs; pytest checks
correctness silently and reports pass/fail. This script is for you to
look at real numbers.

Run from the repo root:
    python main.py
"""

import pandas as pd
from src.data.local_csv import LocalCSVSource
from src.features.stationary import UnivariateStationaryFeaturizer
from src.features.passthrough import RawPassthroughFeaturizer
from src.models.naive import NaiveLastValueModel
from src.models.sarima import SARIMAModel
from src.pipeline import ForecastPipeline
from src.models.ets import ETSModel

from src.features.lag_features import LagFeatureFeaturizer
from src.models.xgboost_model import XGBoostModel

def mape(forecast: pd.Series, actual: pd.Series) -> float:
    return (abs(forecast.values - actual.values) / actual.values).mean() * 100


def main():
    print("Loading data...")
    source = LocalCSVSource(
        filepath="data/local/airpassengers.csv",
        date_col="Month",
        value_cols=["Passengers"],
        freq="MS",
    )
    df = source.load()
    series = df["Passengers"]

    holdout = 12
    train, test = series[:-holdout], series[-holdout:]
    print(f"Train: {len(train)} months, Test (holdout): {len(test)} months\n")

    pipelines = {
        "Naive (last value)": ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel()),
        "SARIMA(1,1,1)(1,1,1,12)": ForecastPipeline(
            UnivariateStationaryFeaturizer(log=True, diff_lags=[]),
            SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
        ),
        "ETS (Holt-Winters, mul)": ForecastPipeline(
            RawPassthroughFeaturizer(), ETSModel(trend="mul", seasonal="mul", seasonal_periods=12)
        ),
        "XGBoost (lag features)": ForecastPipeline(
            LagFeatureFeaturizer(n_lags=13), XGBoostModel(n_estimators=200, max_depth=3, learning_rate=0.05)
        ),
    }

    print(f"{'Model':<28} {'MAPE':>8}")
    print("-" * 38)
    results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(train)
        forecast = pipeline.predict(horizon=holdout)
        score = mape(forecast, test)
        results[name] = (forecast, score)
        print(f"{name:<28} {score:>7.2f}%")

    print("\nMonth-by-month, best model vs actual:")
    best_name = min(results, key=lambda k: results[k][1])
    best_forecast, _ = results[best_name]
    print(f"(best: {best_name})")
    print(f"{'Date':<12} {'Forecast':>10} {'Actual':>10}")
    for date, f, a in zip(test.index, best_forecast.values, test.values):
        print(f"{date.strftime('%Y-%m'):<12} {f:>10.0f} {a:>10.0f}")


if __name__ == "__main__":
    main()