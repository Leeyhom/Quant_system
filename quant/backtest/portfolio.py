"""portfolio —— 横截面因子选股组合回测。

核心流程（见 docs/06）：
- 因子打分：每个再平衡日，用上一交易日的因子值对股票排序。
- 选 top-N：持有分数最高的 N 只股票。
- 等权配仓：每只 1/N。
- 定期再平衡：每隔 rebalance_every 个交易日调仓。
- 扣换手成本：权重变化的绝对和 × 成本率。

组合基准：等权持有所有可交易股票（equal-weight buy & hold）。
"""
from __future__ import annotations

import pandas as pd


def _top_n_equal_weight(scores: pd.Series, top_n: int) -> pd.Series:
    """根据一日因子分数选 top-N 并等权；缺失值不参与排序。"""
    valid = scores.dropna()
    weights = pd.Series(0.0, index=scores.index)
    if valid.empty:
        return weights
    selected = valid.sort_values(ascending=False).head(top_n).index
    weights.loc[selected] = 1.0 / len(selected)
    return weights


def run_factor_portfolio(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    top_n: int = 3,
    rebalance_every: int = 20,
    cost_rate: float = 0.001,
    first_rebalance: bool = False,
) -> pd.DataFrame:
    """运行横截面因子组合回测。

    参数:
        close: 收盘价面板（日期 × 股票）。
        factor: 因子分数面板（同形状，越高越好）。
        top_n: 每次选分数最高的 N 只。
        rebalance_every: 每隔多少个交易日再平衡一次。
        cost_rate: 单位换手成本率（手续费+滑点的组合层近似）。
        first_rebalance: 是否在第 1 个可交易日即调仓。用于样本外验证时，
            可把 train 最后一日作为第 0 行预热数据，test 第一天用上一日因子建仓；
            默认 False，保持旧 demo 行为不变。

    返回:
        DataFrame：port_ret, cost, turnover, equity, benchmark。
    """
    close = close.sort_index()
    factor = factor.reindex_like(close)
    ret = close.pct_change().fillna(0.0)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current_w = pd.Series(0.0, index=close.columns)
    turnover = pd.Series(0.0, index=close.index)

    for i, date in enumerate(close.index):
        # i==0 没有上一交易日因子，保持空仓。
        # 默认从第 rebalance_every 天开始调仓；若 first_rebalance=True，
        # 则在 i==1 时用第0行因子立即建仓，适合样本外验证的「预热行」场景。
        should_rebalance = i > 0 and (
            ((i - 1) % rebalance_every == 0) if first_rebalance else (i % rebalance_every == 0)
        )
        if should_rebalance:
            # 用上一交易日因子选今天起的新持仓，防未来函数。
            target_w = _top_n_equal_weight(factor.iloc[i - 1], top_n)
            turnover.loc[date] = (target_w - current_w).abs().sum()
            current_w = target_w
        weights.loc[date] = current_w

    # 今日收益由昨日收盘到今日收盘，持仓按今日权重近似；首日空仓。
    port_ret_before_cost = (weights * ret).sum(axis=1)
    cost = turnover * cost_rate
    port_ret = port_ret_before_cost - cost

    # 基准：等权持有所有当日有价格的股票（不扣成本，作为朴素对照）
    available = close.notna().astype(float)
    bench_w = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    bench_ret = (bench_w * ret).sum(axis=1)

    out = pd.DataFrame(index=close.index)
    out["port_ret"] = port_ret
    out["cost"] = cost
    out["turnover"] = turnover
    out["equity"] = (1.0 + port_ret).cumprod()
    out["benchmark_ret"] = bench_ret
    out["benchmark"] = (1.0 + bench_ret).cumprod()
    return out
