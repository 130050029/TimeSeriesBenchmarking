"""
Rolling-origin walk-forward validation: instead of one fixed train/test
split, run multiple folds sliding backward through history, score each,
and aggregate. This is the real benchmark; main.py's single holdout was
only ever a smoke test.
"""

import pandas as pd
from typing import Callable
from src.pipeline import ForecastPipeline
from .metrics import mape


def walk_forward_validate(
    series: pd.Series,
    pipeline_factories: dict[str, Callable[[], ForecastPipeline]],
    horizon: int,
    n_folds: int,
    step: int | None = None,
) -> pd.DataFrame:
    """
    pipeline_factories: name -> zero-arg callable returning a FRESH
    ForecastPipeline. A factory (not a shared instance) is required so
    every fold gets a clean, unfitted pipeline -- this doesn't rely on
    any model happening to fully overwrite its state on each fit() call.

    Returns one row per (model, fold): model name, fold's cutoff date, MAPE.
    A model that raises during a fold gets mape=None for that fold rather
    than crashing the whole run -- one model's bug shouldn't block the rest.
    """
    step = step or horizon
    n = len(series)
    results = []

    for fold in range(n_folds):
        cutoff = n - horizon - fold * step
        if cutoff <= 0:
            break
        train = series.iloc[:cutoff]
        test = series.iloc[cutoff:cutoff + horizon]
        if len(test) < horizon:
            continue

        for name, factory in pipeline_factories.items():
            pipeline = factory()
            try:
                pipeline.fit(train)
                forecast = pipeline.predict(horizon)
                score = mape(forecast, test)
            except Exception as e:
                score = None
                print(f"[walk_forward] {name} failed on fold ending {train.index[-1].date()}: {e}")
            results.append({"model": name, "fold_end": train.index[-1], "mape": score})

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean/std/count MAPE across folds, sorted best-first."""
    return results.dropna(subset=["mape"]).groupby("model")["mape"].agg(["mean", "std", "count"]).sort_values("mean")