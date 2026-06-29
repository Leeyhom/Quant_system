"""batch —— 多标的批量回测（见 docs/05）。

对一组股票跑同一个策略（同一组参数），汇总每只的绩效与「是否跑赢买入持有」，
再算横截面统计（胜率、收益/夏普中位数），判断策略是否普适而非靠个股运气。

策略以「信号函数」形式传入：signal_fn(df) -> 0/1 Series，
这样双均线、均值回归（含止损）都能复用，无需为每个策略写一套批量逻辑。
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from quant.data.akshare_loader import fetch_daily, save_parquet, load_parquet
from quant.config import RAW_DATA_DIR
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import summary

SignalFn = Callable[[pd.DataFrame], pd.Series]


def _load_or_fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """优先读本地 Parquet，没有则联网拉取并落地。"""
    path = RAW_DATA_DIR / f"{symbol}.parquet"
    if path.exists():
        return load_parquet(symbol)
    df = fetch_daily(symbol, start=start, end=end)
    save_parquet(df, symbol)
    return df


def batch_backtest(
    symbols: list[str],
    signal_fn: SignalFn,
    start: str = "20240101",
    end: str = "20251231",
) -> pd.DataFrame:
    """对每只股票跑策略并对比买入持有，返回逐股票汇总表。

    返回列：symbol, strat_return, strat_sharpe, strat_mdd,
            bh_return, bh_sharpe, beat_bh(bool)
    """
    rows = []
    for sym in symbols:
        df = _load_or_fetch(sym, start, end)
        bt = run_backtest(df, signal_fn(df))
        strat = summary(bt["equity"], bt["strat_ret"])
        bench = summary(bt["benchmark"], bt["ret"])
        rows.append({
            "symbol": sym,
            "strat_return": strat["total_return"],
            "strat_sharpe": strat["sharpe"],
            "strat_mdd": strat["max_drawdown"],
            "bh_return": bench["total_return"],
            "bh_sharpe": bench["sharpe"],
            "beat_bh": strat["sharpe"] > bench["sharpe"],
        })
    return pd.DataFrame(rows)


def cross_section_stats(result: pd.DataFrame) -> dict:
    """横截面统计：胜率 + 收益/夏普中位数（中位数比均值抗极端值）。"""
    return {
        "n": len(result),
        "win_rate_vs_bh": float(result["beat_bh"].mean()),
        "median_strat_return": float(result["strat_return"].median()),
        "median_strat_sharpe": float(result["strat_sharpe"].median()),
        "median_bh_return": float(result["bh_return"].median()),
    }
