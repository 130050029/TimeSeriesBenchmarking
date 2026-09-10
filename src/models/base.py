from abc import ABC, abstractmethod
from typing import Any


class ForecastModel(ABC):
    """
    A model operates entirely in whatever scale/shape its paired Featurizer
    produces. It never sees raw data and never converts scale -- ForecastPipeline
    handles wiring the two together. This interface has NO dependency on
    Featurizer -- they're independent; only ForecastPipeline composes them.
    """

    @abstractmethod
    def fit(self, features: Any) -> "ForecastModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> Any:
        """Returns predictions on the SAME scale/shape the features were in."""
        raise NotImplementedError