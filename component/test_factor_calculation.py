"""
测试因子表达式计算
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .utility import DataProxy, calculate_by_expression
from .ts_function import (
    ts_delay,
    ts_min,
    ts_max,
    ts_mean,
    ts_sum,
    ts_std,
    ts_rank,
    ts_log,
    ts_abs,
    ts_greater,
    ts_less
)
from .alpha_158 import Alpha158


def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
    symbols = ["AAPL", "MSFT"]
    
    data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            base_price = 100.0 if symbol == "AAPL" else 200.0
            data.append({
                "datetime": date,
                "vt_symbol": symbol,
                "open": base_price + np.random.randn() * 2,
                "high": base_price + abs(np.random.randn()) * 3 + 1,
                "low": base_price - abs(np.random.randn()) * 3 - 1,
                "close": base_price + np.random.randn() * 2,
                "volume": 1000000 + np.random.randint(-100000, 100000),
                "turnover": 100000000 + np.random.randint(-10000000, 10000000),
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(["datetime", "vt_symbol"]).reset_index(drop=True)
    
    # 计算 vwap
    df["vwap"] = df["turnover"] / df["volume"]
    
    return df


def test_data_proxy_operations():
    """测试 DataProxy 基本操作"""
    print("=" * 60)
    print("测试 DataProxy 基本操作")
    print("=" * 60)
    
    df = create_sample_data()
    close_proxy = DataProxy(df[["datetime", "vt_symbol", "close"]])
    open_proxy = DataProxy(df[["datetime", "vt_symbol", "open"]])
    
    # 测试加法
    result = close_proxy + open_proxy
    print(f"✓ 加法操作: close + open")
    print(f"  结果形状: {result.df.shape}")
    print(f"  前5行:\n{result.df.head()}\n")
    
    # 测试减法
    result = close_proxy - open_proxy
    print(f"✓ 减法操作: close - open")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试乘法
    result = close_proxy * 2
    print(f"✓ 乘法操作: close * 2")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试除法
    result = close_proxy / open_proxy
    print(f"✓ 除法操作: close / open")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试绝对值
    result = ts_abs(close_proxy - open_proxy)
    print(f"✓ 绝对值操作: abs(close - open)")
    print(f"  结果形状: {result.df.shape}\n")


def test_ts_functions():
    """测试时间序列函数"""
    print("=" * 60)
    print("测试时间序列函数")
    print("=" * 60)
    
    df = create_sample_data()
    close_proxy = DataProxy(df[["datetime", "vt_symbol", "close"]])
    
    # 测试 ts_delay
    result = ts_delay(close_proxy, 1)
    print(f"✓ ts_delay(close, 1)")
    print(f"  结果形状: {result.df.shape}")
    print(f"  前5行:\n{result.df.head()}\n")
    
    # 测试 ts_mean
    result = ts_mean(close_proxy, 5)
    print(f"✓ ts_mean(close, 5)")
    print(f"  结果形状: {result.df.shape}")
    print(f"  前5行:\n{result.df.head()}\n")
    
    # 测试 ts_min
    result = ts_min(close_proxy, 5)
    print(f"✓ ts_min(close, 5)")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试 ts_max
    result = ts_max(close_proxy, 5)
    print(f"✓ ts_max(close, 5)")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试 ts_sum
    result = ts_sum(close_proxy, 5)
    print(f"✓ ts_sum(close, 5)")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试 ts_std
    result = ts_std(close_proxy, 5)
    print(f"✓ ts_std(close, 5)")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试 ts_rank
    result = ts_rank(close_proxy, 5)
    print(f"✓ ts_rank(close, 5)")
    print(f"  结果形状: {result.df.shape}\n")
    
    # 测试 ts_log
    volume_proxy = DataProxy(df[["datetime", "vt_symbol", "volume"]])
    result = ts_log(volume_proxy)
    print(f"✓ ts_log(volume)")
    print(f"  结果形状: {result.df.shape}\n")


def test_expression_calculation():
    """测试表达式计算"""
    print("=" * 60)
    print("测试表达式计算")
    print("=" * 60)
    
    df = create_sample_data()
    
    # 测试简单表达式
    expression = "(close - open) / open"
    result = calculate_by_expression(df, expression)
    print(f"✓ 表达式: {expression}")
    print(f"  结果形状: {result.shape}")
    print(f"  前5行:\n{result.head()}\n")
    
    # 测试带时间序列函数的表达式
    expression = "ts_mean(close, 5) / close"
    result = calculate_by_expression(df, expression)
    print(f"✓ 表达式: {expression}")
    print(f"  结果形状: {result.shape}")
    print(f"  前5行:\n{result.head()}\n")
    
    # 测试复杂表达式
    expression = "(close - ts_delay(close, 1)) / ts_delay(close, 1)"
    result = calculate_by_expression(df, expression)
    print(f"✓ 表达式: {expression}")
    print(f"  结果形状: {result.shape}")
    print(f"  前5行:\n{result.head()}\n")
    
    # 测试 ts_greater
    expression = "ts_greater(close, open)"
    result = calculate_by_expression(df, expression)
    print(f"✓ 表达式: {expression}")
    print(f"  结果形状: {result.shape}")
    print(f"  前5行:\n{result.head()}\n")
    
    # 测试 ts_less
    expression = "ts_less(close, open)"
    result = calculate_by_expression(df, expression)
    print(f"✓ 表达式: {expression}")
    print(f"  结果形状: {result.shape}\n")


def test_alpha158_features():
    """测试 Alpha158 特征计算"""
    print("=" * 60)
    print("测试 Alpha158 特征计算")
    print("=" * 60)
    
    df = create_sample_data()
    alpha158 = Alpha158(df)
    
    print(f"✓ Alpha158 初始化成功")
    print(f"  特征数量: {len(alpha158.feature_expressions)}")
    print(f"  标签表达式: {alpha158.label_expression}\n")
    
    # 测试计算单个特征
    try:
        result = alpha158.calculate_feature("kmid")
        print(f"✓ 计算特征 'kmid': (close - open) / open")
        print(f"  结果形状: {result.shape}")
        print(f"  前5行:\n{result.head()}\n")
    except Exception as e:
        print(f"✗ 计算特征 'kmid' 失败: {e}\n")
    
    # 测试计算所有特征（只计算前几个特征以节省时间）
    print("计算前5个特征...")
    try:
        result_df = df.copy()
        feature_names = list(alpha158.feature_expressions.keys())[:5]
        
        for name in feature_names:
            feature_df = alpha158.calculate_feature(name)
            result_df = result_df.merge(
                feature_df.rename(columns={"data": name}),
                on=["datetime", "vt_symbol"],
                how="left"
            )
        
        print(f"✓ 成功计算 {len(feature_names)} 个特征")
        print(f"  结果形状: {result_df.shape}")
        print(f"  特征列: {feature_names}")
        print(f"  前3行:\n{result_df[['datetime', 'vt_symbol'] + feature_names].head(3)}\n")
    except Exception as e:
        print(f"✗ 计算特征失败: {e}\n")
        import traceback
        traceback.print_exc()


def test_complex_expressions():
    """测试复杂表达式"""
    print("=" * 60)
    print("测试复杂表达式")
    print("=" * 60)
    
    df = create_sample_data()
    
    # 测试 Alpha158 中的一些典型表达式
    expressions = [
        "(close - open) / open",  # kmid
        "(high - low) / open",  # klen
        "ts_delay(close, 5) / close",  # roc_5
        "ts_mean(close, 5) / close",  # ma_5
        "ts_std(close, 5) / close",  # std_5
        "ts_max(high, 5) / close",  # max_5
        "ts_min(low, 5) / close",  # min_5
    ]
    
    for expr in expressions:
        try:
            result = calculate_by_expression(df, expr)
            print(f"✓ {expr}")
            print(f"  结果形状: {result.shape}, 非空值: {result['data'].notna().sum()}")
        except Exception as e:
            print(f"✗ {expr}")
            print(f"  错误: {e}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试因子表达式计算")
    print("=" * 60 + "\n")
    
    try:
        test_data_proxy_operations()
        test_ts_functions()
        test_expression_calculation()
        test_complex_expressions()
        test_alpha158_features()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

