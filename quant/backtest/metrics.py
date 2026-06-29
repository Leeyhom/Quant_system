"""metrics —— 绩效指标：从净值序列算出总收益、年化、最大回撤、夏普。

公式与直觉见 docs/02 第四节。所有函数输入「净值序列」或「日收益序列」，
不依赖具体策略，可复用于任意净值曲线（策略或基准）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # A股年均交易日，用于年化


def total_return(equity: pd.Series) -> float:
    """总收益率 = 期末净值 / 期初净值 − 1。"""
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(equity: pd.Series) -> float:
    """年化收益率：把总收益折算成「一年」的复合增速。"""
    n = len(equity)
    if n < 2:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (TRADING_DAYS / n) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤：从历史最高净值跌下来的最大比例（返回正数，如 0.23 表示 -23%）。"""
    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    return float(drawdown.max())


def sharpe_ratio(daily_ret: pd.Series, risk_free: float = 0.0) -> float:
    """夏普比率：单位波动换来的超额收益，按交易日年化。

    risk_free 为「年化」无风险利率，内部换算成日度再扣除。
    日收益波动为 0 时返回 0，避免除零。
    """
    rf_daily = risk_free / TRADING_DAYS
    excess = daily_ret - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def summary(equity: pd.Series, daily_ret: pd.Series) -> dict:
    """汇总四个核心指标，返回 dict 便于打印/对比。"""
    return {
        "total_return": total_return(equity),
        "annualized_return": annualized_return(equity),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(daily_ret),
    }
