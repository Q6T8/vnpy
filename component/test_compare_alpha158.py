"""
比较 component 和 vnpy 中 Alpha158 实现的因子计算结果
"""

import pandas as pd
import polars as pl
import numpy as np

# Component 版本（Pandas）
from component.factors import Alpha158 as ComponentAlpha158
from component.factors import calculate as component_calculate

# Vnpy 版本（Polars）
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158 as VnpyAlpha158


def create_test_data(
    n_days: int = 100, n_symbols: int = 3
) -> tuple[pd.DataFrame, pl.DataFrame]:
    """创建测试数据，返回 pandas 和 polars 两个版本"""
    np.random.seed(42)  # 固定随机种子，确保可重复

    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    symbols = [f"SYMBOL_{i}" for i in range(n_symbols)]

    data = []
    for symbol in symbols:
        base_price = 100.0 + hash(symbol) % 50
        for i, date in enumerate(dates):
            trend = i * 0.1  # 轻微趋势
            noise = np.random.randn() * 2
            close = base_price + trend + noise

            data.append(
                {
                    "datetime": date,
                    "vt_symbol": symbol,
                    "open": close + np.random.randn() * 1,
                    "high": close + abs(np.random.randn()) * 2 + 1,
                    "low": close - abs(np.random.randn()) * 2 - 1,
                    "close": close,
                    "volume": 1000000 + np.random.randint(-200000, 200000),
                    "turnover": 100000000 + np.random.randint(-20000000, 20000000),
                }
            )

    # 创建 pandas DataFrame
    df_pd = pd.DataFrame(data)
    df_pd = df_pd.sort_values(["datetime", "vt_symbol"]).reset_index(drop=True)
    df_pd["vwap"] = df_pd["turnover"] / df_pd["volume"]

    # 转换为 polars DataFrame
    df_pl = pl.from_pandas(df_pd)

    return df_pd, df_pl


def compare_feature(
    factor_name: str, expression: str, df_pd: pd.DataFrame, df_pl: pl.DataFrame
) -> dict:
    """比较单个因子的计算结果"""
    result = {
        "feature": factor_name,
        "expression": expression,
        "status": "unknown",
        "error": None,
        "component_shape": None,
        "vnpy_shape": None,
        "match_count": 0,
        "total_count": 0,
        "max_diff": None,
        "mean_diff": None,
        "nan_match": True,
    }

    try:
        # Component 版本计算
        try:
            component_result = component_calculate(df_pd, expression)
            component_values = component_result["data"].values
            component_error = None
        except Exception as e:
            component_error = str(e)
            component_result = None
            component_values = None

        # Vnpy 版本计算
        try:
            from vnpy.alpha.dataset.utility import (
                calculate_by_expression as vnpy_calculate,
            )

            vnpy_result = vnpy_calculate(df_pl, expression)
            vnpy_values = vnpy_result["data"].to_numpy()
            vnpy_error = None
        except Exception as e:
            vnpy_error = str(e)
            vnpy_result = None
            vnpy_values = None

        # 如果两个版本都出错，记录错误
        if component_error and vnpy_error:
            result["status"] = "both_error"
            result["error"] = (
                f"Component: {component_error[:100]}; Vnpy: {vnpy_error[:100]}"
            )
            return result
        elif component_error:
            result["status"] = "component_error"
            result["error"] = f"Component error: {component_error[:100]}"
            return result
        elif vnpy_error:
            result["status"] = "vnpy_error"
            result["error"] = f"Vnpy error: {vnpy_error[:100]}"
            return result

        result["component_shape"] = component_result.shape
        result["vnpy_shape"] = vnpy_result.shape

        # 确保长度一致
        min_len = min(len(component_values), len(vnpy_values))
        component_values = component_values[:min_len]
        vnpy_values = vnpy_values[:min_len]

        # 比较 NaN 位置
        component_nan = pd.isna(component_values)
        vnpy_nan = pd.isna(vnpy_values)
        nan_match = (component_nan == vnpy_nan).all()
        result["nan_match"] = bool(nan_match)

        # 比较非 NaN 值
        both_valid = ~component_nan & ~vnpy_nan
        result["total_count"] = len(component_values)
        result["match_count"] = int(both_valid.sum())

        if both_valid.sum() > 0:
            valid_component = component_values[both_valid]
            valid_vnpy = vnpy_values[both_valid]

            diff = np.abs(valid_component - valid_vnpy)
            result["max_diff"] = float(np.max(diff))
            result["mean_diff"] = float(np.mean(diff))

            # 记录偏差较大的样本
            LARGE_DIFF_THRESHOLD = 0.01
            large_diff_mask = diff > LARGE_DIFF_THRESHOLD
            if np.any(large_diff_mask):
                indices = np.where(both_valid)[0][large_diff_mask][:5]
                result["large_diff_indices"] = indices.tolist()
                result["large_diff_component"] = valid_component[large_diff_mask][
                    :5
                ].tolist()
                result["large_diff_vnpy"] = valid_vnpy[large_diff_mask][:5].tolist()

            # 使用相对误差判断是否匹配（考虑浮点误差）
            relative_diff = diff / (np.abs(valid_vnpy) + 1e-10)
            max_relative_diff = np.max(relative_diff)

            if max_relative_diff < 1e-5:  # 相对误差小于 1e-5 认为匹配
                result["status"] = "match"
            elif max_relative_diff < 1e-3:  # 相对误差小于 1e-3 认为接近
                result["status"] = "close"
            else:
                result["status"] = "diff"
        else:
            result["status"] = "all_nan"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback

        result["traceback"] = traceback.format_exc()

    return result


def test_all_features():
    """测试所有因子"""
    print("=" * 80)
    print("比较 Component (Pandas) 和 Vnpy (Polars) Alpha158 实现")
    print("=" * 80)

    # 创建测试数据
    print("\n创建测试数据...")
    df_pd, df_pl = create_test_data(n_days=100, n_symbols=3)
    print(f"✓ 数据创建成功: {df_pd.shape[0]} 行, {df_pd['vt_symbol'].nunique()} 个标的")

    # 初始化两个版本
    print("\n初始化 Alpha158...")
    component_alpha = ComponentAlpha158(df_pd)
    vnpy_alpha = VnpyAlpha158(
        df_pl,
        train_period=("2024-01-01", "2024-12-31"),
        valid_period=("2024-01-01", "2024-12-31"),
        test_period=("2024-01-01", "2024-12-31"),
    )

    print(f"✓ Component 版本: {len(component_alpha.factor_expressions)} 个因子")
    print(f"✓ Vnpy 版本: {len(vnpy_alpha.feature_expressions)} 个因子")

    # 确保因子列表一致
    component_factors = set(component_alpha.factor_expressions.keys())
    vnpy_factors = set(vnpy_alpha.feature_expressions.keys())

    if component_factors != vnpy_factors:
        print(f"\n⚠️  因子列表不一致:")
        only_component = component_factors - vnpy_factors
        only_vnpy = vnpy_factors - component_factors
        if only_component:
            print(f"  仅在 Component: {only_component}")
        if only_vnpy:
            print(f"  仅在 Vnpy: {only_vnpy}")

    # 测试所有因子
    print(f"\n开始比较 {len(component_alpha.factor_expressions)} 个因子...")
    print("-" * 80)

    results = []
    match_count = 0
    close_count = 0
    diff_count = 0
    error_count = 0
    all_nan_count = 0

    # 按字母顺序排序，便于查看
    sorted_factors = sorted(component_alpha.factor_expressions.items())

    for i, (factor_name, expression) in enumerate(sorted_factors, 1):
        result = compare_feature(factor_name, expression, df_pd, df_pl)
        results.append(result)

        status_icon = {
            "match": "✓",
            "close": "~",
            "diff": "✗",
            "error": "!",
            "all_nan": "N",
        }.get(result["status"], "?")

        status_text = {
            "match": "匹配",
            "close": "接近",
            "diff": "差异",
            "error": "错误",
            "all_nan": "全NaN",
        }.get(result["status"], "未知")

        if result["status"] == "match":
            match_count += 1
        elif result["status"] == "close":
            close_count += 1
        elif result["status"] == "diff":
            diff_count += 1
        elif result["status"] == "error":
            error_count += 1
        elif result["status"] == "all_nan":
            all_nan_count += 1

        # 显示进度和结果
        if result["status"] != "match" or i % 10 == 0:
            print(
                f"{status_icon} [{i:3d}/{len(sorted_factors)}] {factor_name:20s} - {status_text}",
                end="",
            )
            if result["max_diff"] is not None:
                print(
                    f" (最大差异: {result['max_diff']:.2e}, 平均差异: {result['mean_diff']:.2e})"
                )
                if result.get("large_diff_indices"):
                    print(
                        f"    偏差>0.01样本 idx={result['large_diff_indices']} comp={result['large_diff_component']} vnpy={result['large_diff_vnpy']}"
                    )
            elif result["error"]:
                print(f" - {result['error'][:50]}")
            else:
                print()

    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    print(f"总因子数: {len(results)}")
    print(f"✓ 完全匹配: {match_count} ({match_count / len(results) * 100:.1f}%)")
    print(f"~ 接近匹配: {close_count} ({close_count / len(results) * 100:.1f}%)")
    print(f"✗ 存在差异: {diff_count} ({diff_count / len(results) * 100:.1f}%)")
    print(f"! 计算错误: {error_count} ({error_count / len(results) * 100:.1f}%)")
    print(f"N 全为NaN:  {all_nan_count} ({all_nan_count / len(results) * 100:.1f}%)")

    # 显示有差异的因子详情
    if diff_count > 0 or error_count > 0:
        print("\n" + "=" * 80)
        print("差异详情")
        print("=" * 80)

        for result in results:
            if result["status"] in ["diff", "error"]:
                print(f"\n因子: {result['feature']}")
                print(f"  表达式: {result['expression']}")
                if result["status"] == "error":
                    print(f"  错误: {result['error']}")
                else:
                    print(f"  Component 形状: {result['component_shape']}")
                    print(f"  Vnpy 形状: {result['vnpy_shape']}")
                    print(
                        f"  有效值数量: {result['match_count']}/{result['total_count']}"
                    )
                    if result.get("max_diff") is not None:
                        print(f"  最大差异: {result['max_diff']:.6e}")
                        print(f"  平均差异: {result['mean_diff']:.6e}")
                    print(f"  NaN 位置匹配: {result['nan_match']}")

    # 显示接近匹配的因子（可能需要进一步检查）
    if close_count > 0:
        print("\n" + "=" * 80)
        print("接近匹配的因子（可能需要进一步检查）")
        print("=" * 80)

        for result in results:
            if result["status"] == "close":
                print(
                    f"{result['feature']:20s} - 最大差异: {result['max_diff']:.6e}, 平均差异: {result['mean_diff']:.6e}"
                )

    return results


def test_sample_features():
    """测试部分代表性因子（快速测试）"""
    print("=" * 80)
    print("快速测试：比较代表性因子")
    print("=" * 80)

    df_pd, df_pl = create_test_data(n_days=100, n_symbols=2)

    # 选择一些代表性因子
    sample_features = [
        "kmid",  # 简单计算
        "klen",  # 简单计算
        "ma_5",  # 时间序列均值
        "std_5",  # 时间序列标准差
        "roc_5",  # 延迟计算
        "max_5",  # 滚动最大值
        "min_5",  # 滚动最小值
        "rank_5",  # 排名
        "corr_5",  # 相关性
    ]

    component_alpha = ComponentAlpha158(df_pd)
    
    print(f"\n测试 {len(sample_features)} 个代表性因子...")
    print("-" * 80)
    
    for factor_name in sample_features:
        if factor_name not in component_alpha.factor_expressions:
            print(f"✗ {factor_name} - 因子不存在")
            continue
        
        expression = component_alpha.factor_expressions[factor_name]
        result = compare_feature(factor_name, expression, df_pd, df_pl)

        status_icon = {
            "match": "✓",
            "close": "~",
            "diff": "✗",
            "error": "!",
            "all_nan": "N",
        }.get(result["status"], "?")

        print(f"{status_icon} {factor_name:20s}", end="")
        if result["max_diff"] is not None:
            print(
                f" - 最大差异: {result['max_diff']:.6e}, 平均差异: {result['mean_diff']:.6e}"
            )
        elif result["error"]:
            print(f" - 错误: {result['error'][:50]}")
        else:
            print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_sample_features()
    else:
        test_all_features()
