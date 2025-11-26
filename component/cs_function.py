"""
Cross Section Operators (Pandas version)
"""

import pandas as pd

from .utility import DataProxy


def cs_rank(feature: DataProxy) -> DataProxy:
    """Perform cross-sectional ranking"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df.groupby("datetime")["data"].rank()
    return DataProxy(df)


def cs_mean(feature: DataProxy) -> DataProxy:
    """Calculate cross-sectional mean"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df.groupby("datetime")["data"].transform("mean")
    return DataProxy(df)


def cs_std(feature: DataProxy) -> DataProxy:
    """Calculate cross-sectional standard deviation"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df.groupby("datetime")["data"].transform("std")
    return DataProxy(df)
