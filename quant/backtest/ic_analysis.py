"""ic_analysis —— 因子 IC / Rank IC 分析。

IC（Information Coefficient）回答：
    t 日因子分数的横截面排序，能否预测未来 h 日收益的横截面排序？

严谨性：
- 因子值使用 t 日及以前信息。
- 未来收益使用 t -> t+h 的收益，只用于评价，不用于构建当期持仓。
- 默认 Spearman Rank IC，减少异常值影响。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """计算未来 horizon 日收益：close[t+h] / close[t] - 1。"""
    if horizon <= 0:
        raise ValueError("horizon 必须为正整数")
    return close.shift(-horizon) / close - 1.0


def daily_ic(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    method: str = "spearman",
    min_count: int = 5,
) -> pd.Series:
    """逐日计算横截面 IC。

    参数:
        factor: 因子分数面板（日期 × 股票，值越高越好）。
        fwd_ret: 未来收益面板，与 factor 同形状。
        method: "spearman"(Rank IC) 或 "pearson"。
        min_count: 当日有效股票数少于该值时，不计算 IC。

    返回:
        Series，index=日期，value=当日 IC。
    """
    factor, fwd_ret = factor.align(fwd_ret, join="inner", axis=None)
    values = []
    dates = []

    for date in factor.index:
        x = factor.loc[date]
        y = fwd_ret.loc[date]
        valid = pd.concat([x, y], axis=1).dropna()
        if len(valid) < min_count:
            continue
        if method == "spearman":
            # 避免依赖 scipy：Spearman = rank 后的 Pearson 相关。
            x_rank = valid.iloc[:, 0].rank(method="average")
            y_rank = valid.iloc[:, 1].rank(method="average")
            ic = x_rank.corr(y_rank, method="pearson")
        else:
            ic = valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method)
        if pd.notna(ic):
            values.append(ic)
            dates.append(date)

    return pd.Series(values, index=dates, name=f"{method}_ic")


def ic_summary(ic: pd.Series) -> dict:
    """汇总 IC 统计量。"""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {
            "n": 0,
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "positive_rate": np.nan,
            "t_stat": np.nan,
        }

    mean_ic = ic.mean()
    std_ic = ic.std()
    icir = mean_ic / std_ic if std_ic and not np.isnan(std_ic) else np.nan
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic and not np.isnan(std_ic) else np.nan
    return {
        "n": int(n),
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "icir": float(icir),
        "positive_rate": float((ic > 0).mean()),
        "t_stat": float(t_stat),
    }


def cumulative_ic(ic: pd.Series) -> pd.Series:
    """累计 IC，用于观察稳定性趋势。"""
    return ic.dropna().cumsum()
