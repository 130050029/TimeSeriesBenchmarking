"""
Snowflake source. Uses the official snowflake-connector-python.

Requires env vars:
  SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
  SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA

Note: Snowflake does NOT have a perpetual free tier (unlike Databricks Free
Edition) -- only a 30-day trial with credits. Budget for that when testing.
"""

import os
import pandas as pd
from .base import DataSource


class SnowflakeSource(DataSource):
    def __init__(self, query: str, date_col: str):
        self.query = query
        self.date_col = date_col

    def load(self) -> pd.DataFrame:
        try:
            import snowflake.connector
        except ImportError as e:
            raise ImportError(
                "snowflake-connector-python not installed. "
                "Run: pip install snowflake-connector-python"
            ) from e

        required = [
            "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
        ]
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            raise EnvironmentError(f"Missing required env vars for Snowflake: {missing}")

        conn = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
        )
        try:
            df = pd.read_sql(self.query, conn)
        finally:
            conn.close()

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col)
        return self.validate(df)