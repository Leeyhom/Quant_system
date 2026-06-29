"""portfolio_validation —— 因子组合的样本外与滚动 walk-forward 验证。

专业原则（见 docs/07）：
- 候选组合只能在 train 段比较、择优。
- test 段只接受 train 选出的候选，不再调参。
- 滚动窗口把这个过程重复多次，看样本外表现是否稳定。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.backtest.metrics import summary
from quant.backtest.portfolio import run_factor_portfolio


@dataclass(frozen=True)
class PortfolioCandidate:
    """一个可验证的因子组合候选。"""

    name: str
    factor: pd.DataFrame
    top_n: int
    rebalance_every: int


def evaluate_candidate(
    close: pd.DataFrame,
    candidate: PortfolioCandidate,
    cost_rate: float = 0.001,
    first_rebalance: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """在给定区间上评估一个候选，返回回测明细与绩效。"""
    factor = candidate.factor.reindex_like(close)
    bt = run_factor_portfolio(
        close,
        factor,
        top_n=candidate.top_n,
        rebalance_every=candidate.rebalance_every,
        cost_rate=cost_rate,
        first_rebalance=first_rebalance,
    )
    metrics = summary(bt["equity"], bt["port_ret"])
    return bt, metrics


def select_best_candidate(
    close_train: pd.DataFrame,
    candidates: list[PortfolioCandidate],
    metric: str = "sharpe",
    cost_rate: float = 0.001,
) -> tuple[PortfolioCandidate, dict]:
    """只在 train 段选择最优候选。"""
    best_candidate = None
    best_metrics = None
    for c in candidates:
        _, m = evaluate_candidate(close_train, c, cost_rate=cost_rate)
        if best_metrics is None or m[metric] > best_metrics[metric]:
            best_candidate = c
            best_metrics = m
    if best_candidate is None or best_metrics is None:
        raise ValueError("候选列表不能为空")
    return best_candidate, best_metrics


def train_test_validate(
    close: pd.DataFrame,
    candidates: list[PortfolioCandidate],
    train_ratio: float = 0.7,
    metric: str = "sharpe",
    cost_rate: float = 0.001,
) -> dict:
    """单次 train/test 验证。"""
    cut = int(len(close) * train_ratio)
    train = close.iloc[:cut]
    # test 多带一行 train 末日作为预热：test首日可用上一日因子建仓。
    test = close.iloc[cut - 1:]

    best, train_m = select_best_candidate(train, candidates, metric=metric, cost_rate=cost_rate)
    test_bt, test_m = evaluate_candidate(test, best, cost_rate=cost_rate, first_rebalance=True)
    # 去掉预热行，只统计真正样本外。
    test_bt = test_bt.iloc[1:].copy()
    test_bt["equity"] = (1.0 + test_bt["port_ret"]).cumprod()
    test_bt["benchmark"] = (1.0 + test_bt["benchmark_ret"]).cumprod()
    test_m = summary(test_bt["equity"], test_bt["port_ret"])

    return {
        "best": best,
        "n_train": len(train),
        "n_test": len(test_bt),
        "train_metrics": train_m,
        "test_metrics": test_m,
        "test_bt": test_bt,
    }


def rolling_walk_forward(
    close: pd.DataFrame,
    candidates: list[PortfolioCandidate],
    train_size: int = 240,
    test_size: int = 60,
    step: int = 60,
    metric: str = "sharpe",
    cost_rate: float = 0.001,
) -> dict:
    """滚动 walk-forward 验证。

    返回：
        periods: 每个窗口的 train/test 指标与所选候选
        oos: 拼接后的样本外净值/基准净值
    """
    periods = []
    oos_parts = []

    start = 0
    while start + train_size + test_size <= len(close):
        train_start = start
        train_end = start + train_size
        test_end = train_end + test_size

        train = close.iloc[train_start:train_end]
        # 带 train 最后一行作为预热行
        test = close.iloc[train_end - 1:test_end]

        best, train_m = select_best_candidate(train, candidates, metric=metric, cost_rate=cost_rate)
        test_bt, _ = evaluate_candidate(test, best, cost_rate=cost_rate, first_rebalance=True)
        test_bt = test_bt.iloc[1:].copy()
        test_m = summary((1.0 + test_bt["port_ret"]).cumprod(), test_bt["port_ret"])
        bench_m = summary(
            (1.0 + test_bt["benchmark_ret"]).cumprod(),
            test_bt["benchmark_ret"],
        )

        periods.append({
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test_bt.index[0],
            "test_end": test_bt.index[-1],
            "candidate": best,
            "train_metrics": train_m,
            "test_metrics": test_m,
            "benchmark_metrics": bench_m,
            "beat_benchmark": test_m["sharpe"] > bench_m["sharpe"],
        })
        oos_parts.append(test_bt[["port_ret", "benchmark_ret"]])
        start += step

    if not periods:
        raise ValueError("数据长度不足以形成一个滚动窗口")

    oos_ret = pd.concat(oos_parts).sort_index()
    # 防止窗口重叠时重复日期；本项目默认 step=test_size，不重叠，这里仍做保护。
    oos_ret = oos_ret[~oos_ret.index.duplicated(keep="first")]
    oos = pd.DataFrame(index=oos_ret.index)
    oos["port_ret"] = oos_ret["port_ret"]
    oos["benchmark_ret"] = oos_ret["benchmark_ret"]
    oos["equity"] = (1.0 + oos["port_ret"]).cumprod()
    oos["benchmark"] = (1.0 + oos["benchmark_ret"]).cumprod()

    return {"periods": periods, "oos": oos}


def stability_summary(periods: list[dict]) -> dict:
    """滚动窗口稳定性汇总。"""
    test_sharpes = pd.Series([p["test_metrics"]["sharpe"] for p in periods])
    return {
        "n_periods": len(periods),
        "positive_sharpe_rate": float((test_sharpes > 0).mean()),
        "beat_benchmark_rate": float(pd.Series([p["beat_benchmark"] for p in periods]).mean()),
        "median_oos_sharpe": float(test_sharpes.median()),
        "mean_oos_sharpe": float(test_sharpes.mean()),
    }
