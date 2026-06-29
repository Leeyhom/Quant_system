"""mean_reversion —— 均值回归策略：跌深了买、回到均值附近卖。

与 dual_ma 思路相反（见 docs/04~05）：趋势跟随追涨，均值回归抄底。
两者都只产出 0/1 目标仓位信号，接口一致，便于横向对比。

核心量化：用 z-score 衡量价格相对近 N 日均值的偏离程度
    z = (price - rolling_mean) / rolling_std

可选止损（stop_loss）：从买入成本价起算，浮亏超过阈值则无条件清仓，
防止单边阴跌时「接飞刀」。默认关闭，开启不影响其它逻辑（见 docs/05）。
"""
from __future__ import annotations

import pandas as pd


def mean_reversion_signal(
    df: pd.DataFrame,
    window: int = 20,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    price_col: str = "close",
    stop_loss: float | None = None,
) -> pd.Series:
    """基于 z-score 的均值回归信号（只做多，0/1）。

    参数:
        df: 行情，含 price_col，按日期升序。
        window: 计算均值/标准差的回看窗口。
        entry_z: 入场阈值。z 跌破 -entry_z（跌得够深）时买入。
        exit_z: 出场阈值。z 回升到 -exit_z 及以上（价格回到均值附近）时卖出。
        price_col: 价格列。
        stop_loss: 可选止损比例，如 0.08=浮亏8%清仓。None=关闭（默认）。

    返回:
        与 df 等长的 0/1 Series，已 shift(1) 防未来函数。

    实现说明：
        入场/出场是「状态翻转」，需逐日维持仓位（无新信号则沿用昨日）。
        止损需跟踪「买入成本价」，故用显式状态机逐日推进；
        stop_loss=None 时行为与纯向量化版本一致。
    """
    price = df[price_col]
    rolling_mean = price.rolling(window).mean()
    rolling_std = price.rolling(window).std()
    z = (price - rolling_mean) / rolling_std

    position = pd.Series(0.0, index=df.index)
    holding = False        # 当前是否持仓
    entry_price = 0.0      # 持仓时的买入成本价（用于止损）

    for i in range(len(df)):
        zi = z.iloc[i]
        pi = price.iloc[i]

        if holding:
            # 1) 止损优先：浮亏超阈值，强制清仓（哪怕信号仍说持有）
            if stop_loss is not None and pi <= entry_price * (1 - stop_loss):
                holding = False
            # 2) 正常出场：价格回到均值附近
            elif pd.notna(zi) and zi >= -exit_z:
                holding = False
        else:
            # 入场：跌得足够深
            if pd.notna(zi) and zi <= -entry_z:
                holding = True
                entry_price = pi

        position.iloc[i] = 1.0 if holding else 0.0

    # 防未来函数：今天算出的信号，明天才执行
    signal = position.shift(1).fillna(0.0).astype(int)
    signal.name = "signal"
    return signal
