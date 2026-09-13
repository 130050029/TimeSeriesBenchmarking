"""
Featurizer for TFT: reshapes our plain pd.Series into the long-format
table pytorch-forecasting's TimeSeriesDataSet expects (time_idx, target,
group id, known-future covariates). The heavier machinery (TimeSeriesDataSet,
dataloaders, Trainer) lives in the model, not here -- pytorch-forecasting
bundles feature engineering and modeling more tightly than our other
libraries, so the usual clean split is less crisp for this one model.
"""

import pandas as pd
from .base import Featurizer


class TFTFeaturizer(Featurizer):
    def fit_transform(self, raw: pd.Series) -> dict:
        df = pd.DataFrame({"value": raw.values})
        df["time_idx"] = range(len(df))
        df["series_id"] = "series_0"
        df["month"] = raw.index.month.astype(str)
        df["month"] = df["month"].astype("category")

        return {
            "df": df,
            "last_date": raw.index[-1],
            "freq": raw.index.freq,
        }

    def inverse_transform(self, model_output: pd.Series) -> pd.Series:
        return model_output.copy()