"""
Time Series Operators (Pandas version)
"""

from typing import cast

from scipy import stats
import pandas as pd
import numpy as np

from .utility import DataProxy


def ts_delay(feature: DataProxy, window: int) -> DataProxy:
    """Get the value from a fixed time in the past"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df.groupby("vt_symbol")["data"].shift(window)
    return DataProxy(df)


def ts_min(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the minimum value over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).min().reset_index(0, drop=True)
    return DataProxy(df)


def ts_max(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the maximum value over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).max().reset_index(0, drop=True)
    return DataProxy(df)


def ts_argmax(feature: DataProxy, window: int) -> DataProxy:
    """Return the index of the maximum value over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def argmax_func(s: pd.Series) -> int:
        if len(s) == 0:
            return np.nan
        max_idx = s.values.argmax()
        return cast(int, max_idx + 1)
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        argmax_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_argmin(feature: DataProxy, window: int) -> DataProxy:
    """Return the index of the minimum value over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def argmin_func(s: pd.Series) -> int:
        if len(s) == 0:
            return np.nan
        min_idx = s.values.argmin()
        return cast(int, min_idx + 1)
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        argmin_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_rank(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the percentile rank of the current value within the window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def rank_func(s: pd.Series) -> float:
        if len(s) == 0:
            return np.nan
        return stats.percentileofscore(s.values, s.iloc[-1]) / 100
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        rank_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_sum(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the sum over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).sum().reset_index(0, drop=True)
    return DataProxy(df)


def ts_mean(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the mean over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    # 确保数据类型为数值类型（处理布尔类型）
    data_series = feature.df["data"].astype(float)
    
    def mean_func(s: pd.Series) -> float:
        return float(np.nanmean(s.values))
    
    df["data"] = data_series.groupby(feature.df["vt_symbol"]).rolling(window, min_periods=window).apply(
        mean_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_std(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the standard deviation over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def std_func(s: pd.Series) -> float:
        return float(np.nanstd(s.values, ddof=0))
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        std_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_slope(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the slope of linear regression over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def slope_func(s: pd.Series) -> float:
        if len(s) < 2:
            return np.nan
        x = np.arange(len(s))
        y = s.values
        return float(np.polyfit(x, y, 1)[0])
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        slope_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_quantile(feature: DataProxy, window: int, quantile: float) -> DataProxy:
    """Calculate the quantile value over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def quantile_func(s: pd.Series) -> float:
        return float(s.quantile(q=quantile, interpolation="linear"))
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        quantile_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_rsquare(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the R-squared value of linear regression over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def rsquare_func(s: pd.Series) -> float:
        if len(s) < 2:
            return np.nan
        if s.std() == 0:
            return np.nan
        x = np.arange(len(s))
        y = s.values
        return float(stats.linregress(x, y).rvalue ** 2)
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        rsquare_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_resi(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the residual of linear regression over a rolling window"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    
    def resi_func(s: pd.Series) -> float:
        if len(s) < 2:
            return np.nan
        x = np.arange(len(s))
        y = s.values
        coefficients = np.polyfit(x, y, 1)
        predictions = coefficients[0] * x + coefficients[1]
        resi = y - predictions
        return float(resi[-1])
    
    df["data"] = feature.df.groupby("vt_symbol")["data"].rolling(window, min_periods=window).apply(
        resi_func, raw=False
    ).reset_index(0, drop=True)
    return DataProxy(df)


def ts_corr(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    """Calculate the correlation between two features over a rolling window"""
    df_merged: pd.DataFrame = feature1.df.merge(
        feature2.df.rename(columns={"data": "data_right"}),
        on=["datetime", "vt_symbol"],
        how="inner"
    )
    
    df: pd.DataFrame = df_merged[["datetime", "vt_symbol"]].copy()
    
    # Calculate rolling correlation for each group
    corr_list = []
    for vt_symbol, group in df_merged.groupby("vt_symbol"):
        corr_series = group["data"].rolling(window, min_periods=window).corr(group["data_right"])
        corr_list.append(corr_series)
    
    # 合并所有组的结果
    corr_series = pd.concat(corr_list)
    corr_series = corr_series.sort_index()
    
    # 确保索引对齐
    df = df.reset_index(drop=True)
    df["data"] = corr_series.values
    
    # Replace infinite values with NaN
    df["data"] = df["data"].replace([np.inf, -np.inf], np.nan)
    
    return DataProxy(df)


def ts_less(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    """Return the minimum value between two features"""
    if isinstance(feature2, DataProxy):
        df_merged: pd.DataFrame = feature1.df.merge(
            feature2.df.rename(columns={"data": "data_right"}),
            on=["datetime", "vt_symbol"],
            how="inner"
        )
    else:
        df_merged = feature1.df.copy()
        df_merged["data_right"] = feature2
    
    df: pd.DataFrame = df_merged[["datetime", "vt_symbol"]].copy()
    df["data"] = df_merged[["data", "data_right"]].min(axis=1)
    
    return DataProxy(df)


def ts_greater(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    """Return the maximum value between two features"""
    if isinstance(feature2, DataProxy):
        df_merged: pd.DataFrame = feature1.df.merge(
            feature2.df.rename(columns={"data": "data_right"}),
            on=["datetime", "vt_symbol"],
            how="inner"
        )
    else:
        df_merged = feature1.df.copy()
        df_merged["data_right"] = feature2
    
    df: pd.DataFrame = df_merged[["datetime", "vt_symbol"]].copy()
    df["data"] = df_merged[["data", "data_right"]].max(axis=1)
    
    return DataProxy(df)


def ts_log(feature: DataProxy) -> DataProxy:
    """Calculate the natural logarithm of the feature"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = np.log(feature.df["data"])
    return DataProxy(df)


def ts_abs(feature: DataProxy) -> DataProxy:
    """Calculate the absolute value of the feature"""
    df: pd.DataFrame = feature.df[["datetime", "vt_symbol"]].copy()
    df["data"] = feature.df["data"].abs()
    return DataProxy(df)

