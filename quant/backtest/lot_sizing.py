"""lot_sizing —— 整手/碎股可行性工具（三市场统一）。

为什么需要它：
    小资金实盘有个回测看不见的硬约束——【最小可买单位】。
      - 美股：可买碎股（fractional），最小 1 股即可，约束最轻。
      - A股：一手 = 100 股，必须整手。6w 买 15 只时单只预算仅 ¥4000，
        实测当前池 34% 的票一手就超预算买不进——这是 A股小资金最致命的约束。
      - 港股：每手股数因标的而异（100/500/1000…），约束类似 A股。

    回测用连续权重（可买 0.013 股），实盘不行。本工具把目标权重换算成
    真实可买手数，并标出「一手都买不起」的票，供实盘清单做可行性校验。
"""

from __future__ import annotations

import math


# 各市场默认每手股数 / 是否允许碎股
MARKET_LOT = {
    "US": {"lot_size": 1, "allow_fractional": True},   # 碎股
    "CN": {"lot_size": 100, "allow_fractional": False},  # 一手100股
    "HK": {"lot_size": 100, "allow_fractional": False},  # 每手不固定，默认100，实盘需查
}


def affordable_lots(
    weights: dict,
    prices: dict,
    capital: float,
    lot_size: int = 1,
    allow_fractional: bool = False,
) -> dict:
    """把目标权重换算成真实可买手数/股数 + 买不起预警。

    参数:
        weights: {代码: 目标权重}
        prices: {代码: 最新价}
        capital: 总本金
        lot_size: 每手股数（美股碎股用 1）
        allow_fractional: 是否允许碎股（美股 True）。True 时按整数股向下取整。

    返回 dict:
        shares: {代码: 可买股数}（整手市场为 lot_size 的整数倍）
        lots:   {代码: 可买手数}
        unaffordable: [一手/一股都买不起的代码]
        notional: {代码: 实际占用资金}
    """
    shares, lots, notional, unaffordable = {}, {}, {}, []
    for sym, w in weights.items():
        px = prices.get(sym, float("nan"))
        budget = capital * w
        if not (px == px) or px <= 0:   # 价格无效
            shares[sym] = 0; lots[sym] = 0; notional[sym] = 0.0
            unaffordable.append(sym)
            continue
        if allow_fractional:
            n_sh = int(budget // px)     # 碎股：按整数股向下取整
        else:
            n_lots = int(budget // (px * lot_size))  # 整手：向下取整到手
            n_sh = n_lots * lot_size
            lots[sym] = n_lots
        shares[sym] = n_sh
        notional[sym] = n_sh * px
        if n_sh == 0:
            unaffordable.append(sym)
    return {
        "shares": shares,
        "lots": lots,
        "notional": notional,
        "unaffordable": unaffordable,
    }


def market_lot_config(market: str) -> dict:
    """返回某市场的 {lot_size, allow_fractional}，未知市场退回美股碎股口径。"""
    return MARKET_LOT.get(market, MARKET_LOT["US"])
