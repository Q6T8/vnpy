"""
Utility functions for Pandas-based implementations
"""

from datetime import datetime
from typing import Union

import pandas as pd
import numpy as np


class DataProxy:
    """Feature data proxy (Pandas version)"""

    def __init__(self, df: pd.DataFrame) -> None:
        """Constructor"""
        self.name: str = df.columns[-1]
        self.df: pd.DataFrame = df.rename(columns={self.name: "data"})

        # Note that for numerical expressions, variables should be placed before numbers. e.g. a * 2

    def result(self, s: pd.Series) -> "DataProxy":
        """Convert series data to feature object"""
        result: pd.DataFrame = self.df[["datetime", "vt_symbol"]].copy()
        result["data"] = s.values
        return DataProxy(result)

    def __add__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Addition operation"""
        if isinstance(other, DataProxy):
            s: pd.Series = self.df["data"] + other.df["data"]
        else:
            s = self.df["data"] + other
        return self.result(s)

    def __sub__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Subtraction operation"""
        if isinstance(other, DataProxy):
            s: pd.Series = self.df["data"] - other.df["data"]
        else:
            s = self.df["data"] - other
        return self.result(s)

    def __mul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Multiplication operation"""
        if isinstance(other, DataProxy):
            s: pd.Series = self.df["data"] * other.df["data"]
        else:
            s = self.df["data"] * other
        return self.result(s)

    def __rmul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Right multiplication operation"""
        if isinstance(other, DataProxy):
            s: pd.Series = self.df["data"] * other.df["data"]
        else:
            s = self.df["data"] * other
        return self.result(s)

    def __truediv__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Division operation"""
        if isinstance(other, DataProxy):
            s: pd.Series = self.df["data"] / other.df["data"]
        else:
            s = self.df["data"] / other
        return self.result(s)

    def __abs__(self) -> "DataProxy":
        """Get absolute value"""
        s: pd.Series = self.df["data"].abs()
        return self.result(s)

    def __gt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Greater than comparison"""
        if isinstance(other, DataProxy):
            # 如果任一值为 NaN，结果应为 NaN（与 Polars 行为一致）
            mask = self.df["data"].isna() | other.df["data"].isna()
            s: pd.Series = (self.df["data"] > other.df["data"]).astype(float)
            s[mask] = np.nan
        else:
            mask = self.df["data"].isna()
            s = (self.df["data"] > other).astype(float)
            s[mask] = np.nan
        return self.result(s)

    def __ge__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Greater than or equal comparison"""
        if isinstance(other, DataProxy):
            mask = self.df["data"].isna() | other.df["data"].isna()
            s: pd.Series = (self.df["data"] >= other.df["data"]).astype(float)
            s[mask] = np.nan
        else:
            mask = self.df["data"].isna()
            s = (self.df["data"] >= other).astype(float)
            s[mask] = np.nan
        return self.result(s)

    def __lt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Less than comparison"""
        if isinstance(other, DataProxy):
            mask = self.df["data"].isna() | other.df["data"].isna()
            s: pd.Series = (self.df["data"] < other.df["data"]).astype(float)
            s[mask] = np.nan
        else:
            mask = self.df["data"].isna()
            s = (self.df["data"] < other).astype(float)
            s[mask] = np.nan
        return self.result(s)

    def __le__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Less than or equal comparison"""
        if isinstance(other, DataProxy):
            mask = self.df["data"].isna() | other.df["data"].isna()
            s: pd.Series = (self.df["data"] <= other.df["data"]).astype(float)
            s[mask] = np.nan
        else:
            mask = self.df["data"].isna()
            s = (self.df["data"] <= other).astype(float)
            s[mask] = np.nan
        return self.result(s)

    def __eq__(self, other: Union["DataProxy", int, float]) -> "DataProxy":  # type: ignore
        """Equal comparison"""
        if isinstance(other, DataProxy):
            mask = self.df["data"].isna() | other.df["data"].isna()
            s = (self.df["data"] == other.df["data"]).astype(float)
            s[mask] = np.nan
        else:
            mask = self.df["data"].isna()
            s = (self.df["data"] == other).astype(float)
            s[mask] = np.nan
        return self.result(s)


def calculate_by_expression(df: pd.DataFrame, expression: str) -> pd.DataFrame:
    """Execute calculation based on expression (Pandas version)"""
    # Import operators locally to avoid polluting global namespace
    from .ts_function import (  # noqa
        ts_delay,
        ts_min,
        ts_max,
        ts_argmax,
        ts_argmin,
        ts_rank,
        ts_sum,
        ts_mean,
        ts_std,
        ts_slope,
        ts_quantile,
        ts_rsquare,
        ts_resi,
        ts_corr,
        ts_less,
        ts_greater,
        ts_log,
        ts_abs,
    )

    # Import Pandas-based cross-section and technical analysis functions
    from .cs_function import (  # noqa
        cs_rank,
        cs_mean,
        cs_std,
    )
    from .ta_function import (  # noqa
        ta_rsi,
        ta_atr,
    )

    # Extract feature objects to local space
    d: dict = locals()

    for column in df.columns:
        # Filter index columns
        if column in {"datetime", "vt_symbol"}:
            continue

        # Cache feature df
        column_df = df[["datetime", "vt_symbol", column]].copy()
        d[column] = DataProxy(column_df)

    # Use eval to execute calculation
    other: DataProxy = eval(expression, {}, d)

    # Return result DataFrame
    return other.df
