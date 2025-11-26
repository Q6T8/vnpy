import pandas as pd
import numpy as np
import talib as ta
from typing import Union, cast
from scipy import stats


class Alpha158:
    """158 basic factors from Qlib (Pandas-based implementation)"""

    def __init__(self, df: pd.DataFrame) -> None:
        """Constructor"""
        self.df: pd.DataFrame = df
        self.factor_expressions: dict[str, str] = {}

        # Candlestick pattern factors
        self.add_factor("kmid", "(close - open) / open")
        self.add_factor("klen", "(high - low) / open")
        self.add_factor("kmid_2", "(close - open) / (high - low + 1e-12)")
        self.add_factor("kup", "(high - ts_greater(open, close)) / open")
        self.add_factor(
            "kup_2", "(high - ts_greater(open, close)) / (high - low + 1e-12)"
        )
        self.add_factor("klow", "(ts_less(open, close) - low) / open")
        self.add_factor(
            "klow_2", "((ts_less(open, close) - low) / (high - low + 1e-12))"
        )
        self.add_factor("ksft", "(close * 2 - high - low) / open")
        self.add_factor("ksft_2", "(close * 2 - high - low) / (high - low + 1e-12)")

        # Price change factors
        for field in ["open", "high", "low", "vwap"]:
            self.add_factor(f"{field}_0", f"{field} / close")

        # Time series factors
        windows: list[int] = [5, 10, 20, 30, 60]

        for w in windows:
            self.add_factor(f"roc_{w}", f"ts_delay(close, {w}) / close")

        for w in windows:
            self.add_factor(f"ma_{w}", f"ts_mean(close, {w}) / close")

        for w in windows:
            self.add_factor(f"std_{w}", f"ts_std(close, {w}) / close")

        for w in windows:
            self.add_factor(f"beta_{w}", f"ts_slope(close, {w}) / close")

        for w in windows:
            self.add_factor(f"rsqr_{w}", f"ts_rsquare(close, {w})")

        for w in windows:
            self.add_factor(f"resi_{w}", f"ts_resi(close, {w}) / close")

        for w in windows:
            self.add_factor(f"max_{w}", f"ts_max(high, {w}) / close")

        for w in windows:
            self.add_factor(f"min_{w}", f"ts_min(low, {w}) / close")

        for w in windows:
            self.add_factor(f"qtlu_{w}", f"ts_quantile(close, {w}, 0.8) / close")

        for w in windows:
            self.add_factor(f"qtld_{w}", f"ts_quantile(close, {w}, 0.2) / close")

        for w in windows:
            self.add_factor(f"rank_{w}", f"ts_rank(close, {w})")

        for w in windows:
            self.add_factor(
                f"rsv_{w}",
                f"(close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + 1e-12)",
            )

        for w in windows:
            self.add_factor(f"imax_{w}", f"ts_argmax(high, {w}) / {w}")

        for w in windows:
            self.add_factor(f"imin_{w}", f"ts_argmin(low, {w}) / {w}")

        for w in windows:
            self.add_factor(
                f"imxd_{w}", f"(ts_argmax(high, {w}) - ts_argmin(low, {w})) / {w}"
            )

        for w in windows:
            self.add_factor(f"corr_{w}", f"ts_corr(close, ts_log(volume + 1), {w})")

        for w in windows:
            self.add_factor(
                f"cord_{w}",
                f"ts_corr(close / ts_delay(close, 1), ts_log(volume / ts_delay(volume, 1) + 1), {w})",
            )

        for w in windows:
            self.add_factor(f"cntp_{w}", f"ts_mean(close > ts_delay(close, 1), {w})")

        for w in windows:
            self.add_factor(f"cntn_{w}", f"ts_mean(close < ts_delay(close, 1), {w})")

        for w in windows:
            self.add_factor(
                f"cntd_{w}",
                f"ts_mean(close > ts_delay(close, 1), {w}) - ts_mean(close < ts_delay(close, 1), {w})",
            )

        for w in windows:
            self.add_factor(
                f"sump_{w}",
                f"ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)",
            )

        for w in windows:
            self.add_factor(
                f"sumn_{w}",
                f"ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w}) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)",
            )

        for w in windows:
            self.add_factor(
                f"sumd_{w}",
                f"(ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) - ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w})) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)",
            )

        for w in windows:
            self.add_factor(f"vma_{w}", f"ts_mean(volume, {w}) / (volume + 1e-12)")

        for w in windows:
            self.add_factor(f"vstd_{w}", f"ts_std(volume, {w}) / (volume + 1e-12)")

        for w in windows:
            self.add_factor(
                f"wvma_{w}",
                f"ts_std(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) / (ts_mean(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) + 1e-12)",
            )

        for w in windows:
            self.add_factor(
                f"vsump_{w}",
                f"ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)",
            )

        for w in windows:
            self.add_factor(
                f"vsumn_{w}",
                f"ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w}) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)",
            )

        for w in windows:
            self.add_factor(
                f"vsumd_{w}",
                f"(ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) - ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w})) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)",
            )

    def add_factor(self, name: str, expression: str) -> None:
        """Add a factor expression"""
        self.factor_expressions[name] = expression

    def calculate_factor(self, name: str) -> pd.DataFrame:
        """Calculate a single factor"""
        if name not in self.factor_expressions:
            raise ValueError(f"Factor '{name}' not found")
        expression = self.factor_expressions[name]
        return calculate(self.df, expression)

    def calculate_all_factors(self) -> pd.DataFrame:
        """Calculate all factors"""
        result_df = self.df.copy()

        for name, expression in self.factor_expressions.items():
            factor_df = calculate(self.df, expression)
            result_df = result_df.merge(
                factor_df.rename(columns={"data": name}),
                on=["datetime", "vt_symbol"],
                how="left",
            )

        return result_df


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


def ts_delay(feature: DataProxy, window: int) -> DataProxy:
    """Get the value from a fixed time in the past"""
    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = feature.df.groupby("vt_symbol")["data"].shift(window)
    return DataProxy(result)


def ts_min(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the minimum value over a rolling window"""
    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .min()
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_max(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the maximum value over a rolling window"""
    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .max()
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_argmax(feature: DataProxy, window: int) -> DataProxy:
    """Return the index of the maximum value over a rolling window"""

    def argmax_func(s: pd.Series) -> int:
        if len(s) == 0:
            return np.nan
        max_idx = s.values.argmax()
        return cast(int, max_idx + 1)

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(argmax_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_argmin(feature: DataProxy, window: int) -> DataProxy:
    """Return the index of the minimum value over a rolling window"""

    def argmin_func(s: pd.Series) -> int:
        if len(s) == 0:
            return np.nan
        min_idx = s.values.argmin()
        return cast(int, min_idx + 1)

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(argmin_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_rank(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the percentile rank of the current value within the window"""

    def rank_func(s: pd.Series) -> float:
        if len(s) == 0:
            return np.nan
        return stats.percentileofscore(s.values, s.iloc[-1]) / 100

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(rank_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_sum(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the sum over a rolling window"""
    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .sum()
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_mean(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the mean over a rolling window"""

    def mean_func(s: pd.Series) -> float:
        return float(np.nanmean(s.values))

    # 确保数据类型为数值类型（处理布尔类型）
    data_series = feature.df["data"].astype(float)
    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        data_series.groupby(feature.df["vt_symbol"])
        .rolling(window, min_periods=window)
        .apply(mean_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_std(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the standard deviation over a rolling window"""

    def std_func(s: pd.Series) -> float:
        return float(np.nanstd(s.values, ddof=0))

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(std_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_slope(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the slope of linear regression over a rolling window"""

    def slope_func(s: pd.Series) -> float:
        if len(s) < 2:
            return np.nan
        x = np.arange(len(s))
        y = s.values
        return float(np.polyfit(x, y, 1)[0])

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(slope_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_quantile(feature: DataProxy, window: int, quantile: float) -> DataProxy:
    """Calculate the quantile value over a rolling window"""

    def quantile_func(s: pd.Series) -> float:
        return float(s.quantile(q=quantile, interpolation="linear"))

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(quantile_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_rsquare(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the R-squared value of linear regression over a rolling window"""

    def rsquare_func(s: pd.Series) -> float:
        if len(s) < 2:
            return np.nan
        if s.std() == 0:
            return np.nan
        x = np.arange(len(s))
        y = s.values
        return float(stats.linregress(x, y).rvalue ** 2)

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(rsquare_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_resi(feature: DataProxy, window: int) -> DataProxy:
    """Calculate the residual of linear regression over a rolling window"""

    def resi_func(s: pd.Series) -> float:
        if len(s) < 2:
            return np.nan
        x = np.arange(len(s))
        y = s.values
        coefficients = np.polyfit(x, y, 1)
        predictions = coefficients[0] * x + coefficients[1]
        resi = y - predictions
        return float(resi[-1])

    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = (
        feature.df.groupby("vt_symbol")["data"]
        .rolling(window, min_periods=window)
        .apply(resi_func, raw=False)
        .reset_index(0, drop=True)
    )
    return DataProxy(result)


def ts_corr(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    """Calculate the correlation between two features over a rolling window"""
    df_merged = feature1.df.merge(
        feature2.df.rename(columns={"data": "data_right"}),
        on=["datetime", "vt_symbol"],
        how="inner",
    )

    result = df_merged[["datetime", "vt_symbol"]].copy()

    # Calculate rolling correlation for each group
    corr_list = []
    for _vt_symbol, group in df_merged.groupby("vt_symbol"):
        corr_series = (
            group["data"].rolling(window, min_periods=window).corr(group["data_right"])
        )
        corr_list.append(corr_series)

    # 合并所有组的结果
    corr_series = pd.concat(corr_list).sort_index()

    # 确保索引对齐并替换无穷值
    result = result.reset_index(drop=True)
    result["data"] = corr_series.values
    result["data"] = result["data"].replace([np.inf, -np.inf], np.nan)

    return DataProxy(result)


def ts_less(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    """Return the minimum value between two features"""
    if isinstance(feature2, DataProxy):
        df_merged = feature1.df.merge(
            feature2.df.rename(columns={"data": "data_right"}),
            on=["datetime", "vt_symbol"],
            how="inner",
        )
        result = df_merged[["datetime", "vt_symbol"]].copy()
        result["data"] = df_merged[["data", "data_right"]].min(axis=1)
    else:
        result = feature1.df[["datetime", "vt_symbol"]].copy()
        result["data"] = feature1.df["data"].clip(upper=feature2)

    return DataProxy(result)


def ts_greater(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    """Return the maximum value between two features"""
    if isinstance(feature2, DataProxy):
        df_merged = feature1.df.merge(
            feature2.df.rename(columns={"data": "data_right"}),
            on=["datetime", "vt_symbol"],
            how="inner",
        )
        result = df_merged[["datetime", "vt_symbol"]].copy()
        result["data"] = df_merged[["data", "data_right"]].max(axis=1)
    else:
        result = feature1.df[["datetime", "vt_symbol"]].copy()
        result["data"] = feature1.df["data"].clip(lower=feature2)

    return DataProxy(result)


def ts_log(feature: DataProxy) -> DataProxy:
    """Calculate the natural logarithm of the feature"""
    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = np.log(feature.df["data"])
    return DataProxy(result)


def ts_abs(feature: DataProxy) -> DataProxy:
    """Calculate the absolute value of the feature"""
    result = feature.df[["datetime", "vt_symbol"]].copy()
    result["data"] = feature.df["data"].abs()
    return DataProxy(result)


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


def to_pd_series(feature: DataProxy) -> pd.Series:
    """Convert to pandas.Series data structure"""
    series: pd.Series = feature.df.set_index(["datetime", "vt_symbol"])["data"]
    return series


def ta_rsi(close: DataProxy, window: int) -> DataProxy:
    """Calculate RSI indicator by contract"""
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = ta.RSI(close_, timeperiod=window)  # type: ignore

    df: pd.DataFrame = result.reset_index()
    # result 是 Series，reset_index 后列名是 Series 的名称或 0
    if "data" in df.columns:
        pass  # 已经是 data
    elif 0 in df.columns:
        df = df.rename(columns={0: "data"})
    else:
        # 如果 Series 有名称，使用该名称
        df = (
            df.rename(columns={result.name: "data"})
            if result.name
            else df.rename(columns={df.columns[-1]: "data"})
        )
    return DataProxy(df)


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """Calculate ATR indicator by contract"""
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = ta.ATR(high_, low_, close_, timeperiod=window)  # type: ignore

    df: pd.DataFrame = result.reset_index()
    # result 是 Series，reset_index 后列名是 Series 的名称或 0
    if "data" in df.columns:
        pass  # 已经是 data
    elif 0 in df.columns:
        df = df.rename(columns={0: "data"})
    else:
        # 如果 Series 有名称，使用该名称
        df = (
            df.rename(columns={result.name: "data"})
            if result.name
            else df.rename(columns={df.columns[-1]: "data"})
        )
    return DataProxy(df)


def calculate(df: pd.DataFrame, expression: str) -> pd.DataFrame:
    """Execute calculation based on expression"""
    # Create execution context with all available functions
    exec_context: dict = {
        "DataProxy": DataProxy,
        "ts_delay": ts_delay,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "ts_argmax": ts_argmax,
        "ts_argmin": ts_argmin,
        "ts_rank": ts_rank,
        "ts_sum": ts_sum,
        "ts_mean": ts_mean,
        "ts_std": ts_std,
        "ts_slope": ts_slope,
        "ts_quantile": ts_quantile,
        "ts_rsquare": ts_rsquare,
        "ts_resi": ts_resi,
        "ts_corr": ts_corr,
        "ts_less": ts_less,
        "ts_greater": ts_greater,
        "ts_log": ts_log,
        "ts_abs": ts_abs,
        "cs_rank": cs_rank,
        "cs_mean": cs_mean,
        "cs_std": cs_std,
        "ta_rsi": ta_rsi,
        "ta_atr": ta_atr,
    }

    # Add column data as DataProxy objects
    for column in df.columns:
        # Filter index columns
        if column in {"datetime", "vt_symbol"}:
            continue

        # Cache feature df
        column_df = df[["datetime", "vt_symbol", column]].copy()
        exec_context[column] = DataProxy(column_df)

    # Use eval to execute calculation
    other: DataProxy = eval(expression, exec_context, {})

    # Return result DataFrame
    return other.df
