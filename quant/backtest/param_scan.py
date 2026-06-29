"""param_scan —— 双均线参数网格扫描，评估策略对参数的敏感性。

思路（呼应 docs/03）：
- 遍历一批 (short, long) 窗口组合，每组跑一次回测，记录绩效。
- 汇总成表格，并能透视成「短窗口 × 长窗口」的矩阵，便于画热力图看「平原 vs 尖峰」。
- 只接受 short < long 的合理组合。
"""
from __future__ import annotations

import pandas as pd

from quant.strategy.dual_ma import dual_ma_signal
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import summary


def scan(
    df: pd.DataFrame,
    short_windows: list[int],
    long_windows: list[int],
    price_col: str = "close",
) -> pd.DataFrame:
    """对所有 short < long 的窗口组合跑回测，返回长表。

    返回列：short, long, total_return, annualized_return, max_drawdown, sharpe
    """
    rows = []
    for s in short_windows:
        for l in long_windows:
            if s >= l:
                continue  # 短窗口必须小于长窗口
            signal = dual_ma_signal(df, short_window=s, long_window=l, price_col=price_col)
            bt = run_backtest(df, signal, price_col=price_col)
            m = summary(bt["equity"], bt["strat_ret"])
            rows.append({"short": s, "long": l, **m})
    return pd.DataFrame(rows)


def pivot(result: pd.DataFrame, metric: str = "sharpe") -> pd.DataFrame:
    """把长表透视成「short(行) × long(列)」矩阵，便于画热力图。"""
    return result.pivot(index="short", columns="long", values=metric)
