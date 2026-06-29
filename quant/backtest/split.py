"""split —— 样本内/样本外切分，检验策略是否过拟合（见 docs/05）。

流程：
1. 按时间把行情切成 train（前段）和 test（后段）—— 时序数据不能打乱。
2. 在 train 上扫描参数，选出最优组合（样本内寻优）。
3. 用这组参数在 test 上回测（样本外验证）。
4. 对比 train/test 表现：都好=可信；train好test垮=过拟合。
"""
from __future__ import annotations

import pandas as pd

from quant.strategy.dual_ma import dual_ma_signal
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import summary


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.7):
    """按时间顺序切分（不打乱），返回 (train_df, test_df)，索引均重置。"""
    n = len(df)
    cut = int(n * train_ratio)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    return train, test


def _eval_dual_ma(df: pd.DataFrame, short: int, long: int) -> dict:
    """在给定数据上用一组双均线参数回测，返回绩效。"""
    signal = dual_ma_signal(df, short_window=short, long_window=long)
    bt = run_backtest(df, signal)
    return summary(bt["equity"], bt["strat_ret"])


def select_best_dual_ma(
    train: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    metric: str = "sharpe",
):
    """在 train 上遍历参数，按 metric 选最优组合。

    返回 (best_short, best_long, best_train_metrics)。
    """
    best = None
    for s in short_windows:
        for l in long_windows:
            if s >= l:
                continue
            m = _eval_dual_ma(train, s, l)
            if best is None or m[metric] > best[2][metric]:
                best = (s, l, m)
    return best


def walk_forward_dual_ma(
    df: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    train_ratio: float = 0.7,
    metric: str = "sharpe",
) -> dict:
    """完整的一次样本内外检验，返回 train/test 绩效与所选参数。"""
    train, test = train_test_split(df, train_ratio)
    best_short, best_long, train_metrics = select_best_dual_ma(
        train, short_windows, long_windows, metric
    )
    test_metrics = _eval_dual_ma(test, best_short, best_long)
    return {
        "best_short": best_short,
        "best_long": best_long,
        "n_train": len(train),
        "n_test": len(test),
        "train": train_metrics,
        "test": test_metrics,
    }
