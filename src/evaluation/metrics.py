import pandas as pd


def mape(forecast: pd.Series, actual: pd.Series) -> float:
    return float((abs(forecast.values - actual.values) / abs(actual.values)).mean() * 100)