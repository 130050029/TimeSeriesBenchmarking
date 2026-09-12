import pandas as pd
import pytest
from src.features.passthrough import RawPassthroughFeaturizer
from src.models.naive import NaiveLastValueModel
from src.pipeline import ForecastPipeline
from src.evaluation.walk_forward import walk_forward_validate, summarize


@pytest.fixture
def raw_series(airpassengers_csv):
    df = pd.read_csv(airpassengers_csv, parse_dates=["Month"]).set_index("Month")["Passengers"]
    return df.asfreq("MS")


def test_walk_forward_produces_one_row_per_model_per_fold(raw_series):
    factories = {
        "naive_a": lambda: ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel()),
        "naive_b": lambda: ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel()),
    }
    results = walk_forward_validate(raw_series, factories, horizon=12, n_folds=3, step=12)
    assert len(results) == 2 * 3
    assert set(results["model"]) == {"naive_a", "naive_b"}


def test_walk_forward_folds_use_different_cutoffs(raw_series):
    factories = {"naive": lambda: ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel())}
    results = walk_forward_validate(raw_series, factories, horizon=12, n_folds=3, step=12)
    assert results["fold_end"].nunique() == 3


def test_walk_forward_handles_a_failing_model_without_crashing(raw_series):
    class BrokenModel:
        def fit(self, features):
            raise ValueError("intentionally broken for this test")
        def predict(self, horizon):
            pass

    factories = {
        "broken": lambda: ForecastPipeline(RawPassthroughFeaturizer(), BrokenModel()),
        "naive": lambda: ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel()),
    }
    results = walk_forward_validate(raw_series, factories, horizon=12, n_folds=2, step=12)

    broken_rows = results[results["model"] == "broken"]
    naive_rows = results[results["model"] == "naive"]
    assert broken_rows["mape"].isna().all()
    assert naive_rows["mape"].notna().all()


def test_summarize_aggregates_and_sorts_by_mean(raw_series):
    factories = {"naive": lambda: ForecastPipeline(RawPassthroughFeaturizer(), NaiveLastValueModel())}
    results = walk_forward_validate(raw_series, factories, horizon=12, n_folds=2, step=12)
    summary = summarize(results)
    assert "mean" in summary.columns and "std" in summary.columns and "count" in summary.columns
    assert summary.loc["naive", "count"] == 2