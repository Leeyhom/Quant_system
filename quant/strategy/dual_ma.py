"""dual_ma —— 双均线策略：金叉持有、死叉空仓。

为什么这么设计（呼应 docs/02）：
- 策略只产出「目标仓位信号」(0/1)，把交易细节留给回测引擎，职责单一。
- 关键防坑点：均线要收盘才算得出，因此**信号必须 shift(1)**——今天算出的
  信号，明天才据此交易。否则就是「未来函数」，回测收益虚高且不可信。
"""
from __future__ import annotations

import pandas as pd


def dual_ma_signal(
    df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    price_col: str = "close",
) -> pd.Series:
    """根据双均线生成每日目标仓位信号。

    参数:
        df: 含日线行情的 DataFrame，至少有 price_col 列，按日期升序。
        short_window/long_window: 短/长均线窗口（交易日）。
        price_col: 用于计算均线的价格列，默认收盘价。

    返回:
        与 df 等长的 Series，值为 1（持有）或 0（空仓）。
        已做 shift(1)：第 t 天的信号代表「基于 t-1 及之前的数据，t 天应持有的仓位」，
        可直接喂给回测引擎，无未来函数。
    """
    price = df[price_col]
    ma_short = price.rolling(window=short_window).mean()
    ma_long = price.rolling(window=long_window).mean()

    # 短线在长线上方 -> 想持有(1)，否则空仓(0)
    raw_signal = (ma_short > ma_long).astype(int)

    # 防未来函数：今天算出的信号，明天才执行
    signal = raw_signal.shift(1).fillna(0).astype(int)
    signal.name = "signal"
    return signal
