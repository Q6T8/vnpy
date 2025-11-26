"""
Technical Analysis Operators (Pandas version)
"""

import talib
import pandas as pd
import numpy as np

from .utility import DataProxy


def to_pd_series(feature: DataProxy) -> pd.Series:
    """Convert to pandas.Series data structure"""
    series: pd.Series = feature.df.set_index(["datetime", "vt_symbol"])["data"]
    return series


def ta_rsi(close: DataProxy, window: int) -> DataProxy:
    """Calculate RSI indicator by contract"""
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.RSI(close_, timeperiod=window)   # type: ignore

    df: pd.DataFrame = result.reset_index()
    # result 是 Series，reset_index 后列名是 Series 的名称或 0
    if "data" in df.columns:
        pass  # 已经是 data
    elif 0 in df.columns:
        df = df.rename(columns={0: "data"})
    else:
        # 如果 Series 有名称，使用该名称
        df = df.rename(columns={result.name: "data"}) if result.name else df.rename(columns={df.columns[-1]: "data"})
    return DataProxy(df)


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """Calculate ATR indicator by contract"""
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.ATR(high_, low_, close_, timeperiod=window)   # type: ignore

    df: pd.DataFrame = result.reset_index()
    # result 是 Series，reset_index 后列名是 Series 的名称或 0
    if "data" in df.columns:
        pass  # 已经是 data
    elif 0 in df.columns:
        df = df.rename(columns={0: "data"})
    else:
        # 如果 Series 有名称，使用该名称
        df = df.rename(columns={result.name: "data"}) if result.name else df.rename(columns={df.columns[-1]: "data"})
    return DataProxy(df)

