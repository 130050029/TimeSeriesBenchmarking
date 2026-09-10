from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class Featurizer(ABC):
    """
    Converts raw data into whatever shape a model family needs, and knows
    how to convert that model's output back to a real-scale forecast.
    Models never see raw data directly, and never do scale conversion --
    that responsibility lives here, once, shared across any model that
    needs the same shape.
    """

    @abstractmethod
    def fit_transform(self, raw: pd.Series) -> Any:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, model_output: Any) -> pd.Series:
        raise NotImplementedError