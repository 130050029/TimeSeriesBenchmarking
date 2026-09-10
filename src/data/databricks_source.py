"""
Databricks source. Uses the official databricks-sql-connector.

Requires env vars (never hardcode these):
  DATABRICKS_SERVER_HOSTNAME
  DATABRICKS_HTTP_PATH
  DATABRICKS_ACCESS_TOKEN

Get these free via Databricks Free Edition (community.databricks.com) --
Community Edition was retired Jan 2026 and replaced by this perpetually-free tier.
"""

import os
import pandas as pd
from .base import DataSource


class DatabricksSource(DataSource):
    def __init__(self, query: str, date_col: str):
        """
        query:    a SQL SELECT statement, e.g.
                  "SELECT ds, y FROM main.timeseries.demand ORDER BY ds"
        date_col: which returned column to set as the DatetimeIndex
        """
        self.query = query
        self.date_col = date_col

    def load(self) -> pd.DataFrame:
        try:
            from databricks import sql
        except ImportError as e:
            raise ImportError(
                "databricks-sql-connector not installed. "
                "Run: pip install databricks-sql-connector"
            ) from e

        required = ["DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_ACCESS_TOKEN"]
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            raise EnvironmentError(f"Missing required env vars for Databricks: {missing}")

        with sql.connect(
            server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_ACCESS_TOKEN"],
        ) as conn:
            df = pd.read_sql(self.query, conn)

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col)
        return self.validate(df)