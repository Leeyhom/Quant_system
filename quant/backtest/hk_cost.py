"""hk_cost —— 港股专属交易费用模型（小资金口径，仿 us_cost 工厂模式）。

为什么需要它：
    同 cn_cost——原比例成本忽略小资金的固定费/最低费。港股费用结构比 A股更碎：
    佣金 + 平台费（互联网券商每笔固定）+ 印花税（双边）+ 交易费 + 交收费 + 证监会征费。
    6w 港币买 15 只 → 单只 ¥4000，每笔固定平台费占比可观。

港股互联网券商费率（用户确认口径，如富途/老虎/星财富）：
    - 佣金：万1.5 左右（0.00015），常有每笔最低（如 0），按互联网券商低佣口径。
    - 平台费：每笔固定（如 15 港币/笔），小资金主要拖累。
    - 印花税：0.1%（0.001），**买卖双边**，不足 1 港币向上取整（这里近似为比例）。
    - 交易费（联交所）：0.00565%（0.0000565），双边。
    - 交收费（结算）：0.002%，最低 2 / 最高 100 港币（这里近似为比例，未建封顶）。
    - 证监会征费：0.0027%，双边。
    - 滑点：0.05%，双边。

设计与 us_cost / cn_cost 一致：
    - hk_trade_cost_hkd(...)：纯函数，给定各票成交额，返回总港币费用。
    - make_layered_cost_fn(...)：工厂，产 cost_fn(weight_change, prices) -> 收益拖累比例。

口径说明（诚实标注近似）：
    印花税双边、平台费每笔固定（按发生交易的票各算一笔），交收费的最低 2/最高 100
    封顶未精确建模（按比例近似）。实盘前需用券商实际账单校准。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── 港股费用默认参数（互联网券商低佣口径） ───
DEFAULT_COMMISSION = 0.00015       # 佣金万1.5，双边
DEFAULT_PLATFORM_FEE = 15.0        # 平台费每笔固定 15 港币（小资金杀手）
DEFAULT_STAMP_DUTY = 0.001         # 印花税 0.1%，双边
DEFAULT_TRADING_FEE = 0.0000565    # 联交所交易费，双边
DEFAULT_SETTLEMENT_FEE = 0.00002   # 交收费 0.002%，双边（最低2/最高100未建封顶）
DEFAULT_SFC_LEVY = 0.000027        # 证监会征费 0.0027%，双边
DEFAULT_SLIPPAGE = 0.0005          # 滑点 0.05%，双边
DEFAULT_PORTFOLIO_VALUE = 60_000.0  # 名义本金 = 实盘 6w 港币


def hk_trade_cost_hkd(
    notional: pd.Series,
    commission_rate: float = DEFAULT_COMMISSION,
    platform_fee: float = DEFAULT_PLATFORM_FEE,
    stamp_duty: float = DEFAULT_STAMP_DUTY,
    trading_fee: float = DEFAULT_TRADING_FEE,
    settlement_fee: float = DEFAULT_SETTLEMENT_FEE,
    sfc_levy: float = DEFAULT_SFC_LEVY,
    slippage: float = DEFAULT_SLIPPAGE,
) -> float:
    """给定各票本次成交额（港币，绝对值），返回总费用（港币）。

    每只发生交易（成交额>0）的票：
      佣金     = 成交额 × 佣金率
      平台费   = 15 港币/笔（固定，小资金占比高）
      印花税   = 成交额 × 0.1%（双边）
      交易费   = 成交额 × 0.00565%
      交收费   = 成交额 × 0.002%
      征费     = 成交额 × 0.0027%
      滑点     = 成交额 × 0.05%
    """
    traded = notional[notional > 0]
    n = len(traded)
    if n == 0:
        return 0.0
    commission = traded * commission_rate
    platform = platform_fee * n            # 每只一笔固定平台费
    proportional = traded * (stamp_duty + trading_fee + settlement_fee + sfc_levy + slippage)
    return float(commission.sum() + platform + proportional.sum())


def make_layered_cost_fn(
    commission_rate: float = DEFAULT_COMMISSION,
    platform_fee: float = DEFAULT_PLATFORM_FEE,
    stamp_duty: float = DEFAULT_STAMP_DUTY,
    trading_fee: float = DEFAULT_TRADING_FEE,
    settlement_fee: float = DEFAULT_SETTLEMENT_FEE,
    sfc_levy: float = DEFAULT_SFC_LEVY,
    slippage: float = DEFAULT_SLIPPAGE,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
):
    """产出与 layered_backtest / fixed_topn_portfolio 的 cost_fn 注入点兼容的回调。"""
    def cost_fn(weight_change: pd.Series, prices: pd.Series) -> float:
        wc = weight_change.abs()
        valid = (wc > 0)
        if not valid.any():
            return 0.0
        notional = wc[valid] * portfolio_value
        hkd = hk_trade_cost_hkd(
            notional, commission_rate, platform_fee, stamp_duty,
            trading_fee, settlement_fee, sfc_levy, slippage,
        )
        return hkd / portfolio_value

    return cost_fn
