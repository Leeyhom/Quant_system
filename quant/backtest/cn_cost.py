"""cn_cost —— A股专属交易费用模型（小资金口径，仿 us_cost 工厂模式）。

为什么需要它：
    项目原本 A股/港股回测都用 engine.py / long_top_layer 的【比例成本】
    （0.1%换手 + 0.05%滑点），它假设费用随金额等比缩放——这恰恰忽略了小资金
    最致命的「每笔最低佣金」。A股佣金虽万2.5，但每笔最低 5 元：6w 本金买 15 只
    → 单只 ¥4000，一笔 5 元 = 0.125%（和美股 $1 固定费同构）。本模型把最低佣金、
    印花税（卖出单边）、过户费（双边）都建进去，给小资金一个真实的费用拖累。

A股标准散户费率（用户确认，2023 印花税减半后口径）：
    - 佣金：万2.5（0.025%），每笔最低 5 元。买卖双边都收。
    - 印花税：0.05%（0.0005），**仅卖出**收。
    - 过户费：万0.1（0.00001），买卖双边（沪深已统一收取）。
    - 滑点：0.05%（保留，模拟真实成交价偏差）。

设计与 us_cost 一致：
    - cn_trade_cost_yuan(...)：纯函数，给定各票成交额，返回总人民币费用。
    - make_layered_cost_fn(...)：工厂，产 cost_fn(weight_change, prices) -> 收益拖累比例。

口径说明（诚实标注近似）：
    cost_fn 拿到的是 |weight_change|（买卖合并的绝对权重变动），无法区分买/卖方向。
    印花税只在卖出侧收，这里用 0.5 系数近似（假设半数成交额是卖出）。佣金最低 5 元
    按「每只发生交易的票算一笔」近似（实际一次调仓买卖可能算两笔，偏保守）。
    实盘前需用券商实际账单校准——与 us_cost「实盘从券商取实时费率」同口径。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── A股费用默认参数（标准散户费率） ───
DEFAULT_COMMISSION = 0.00025       # 佣金万2.5，双边
DEFAULT_MIN_COMMISSION = 5.0       # 每笔最低 5 元（小资金杀手）
DEFAULT_STAMP_DUTY = 0.0005        # 印花税 0.05%，仅卖出
DEFAULT_TRANSFER_FEE = 0.00001     # 过户费万0.1，双边
DEFAULT_SLIPPAGE = 0.0005          # 滑点 0.05%，双边
DEFAULT_PORTFOLIO_VALUE = 60_000.0  # 名义本金 = 实盘 6w 人民币


def cn_trade_cost_yuan(
    notional: pd.Series,
    commission_rate: float = DEFAULT_COMMISSION,
    min_commission: float = DEFAULT_MIN_COMMISSION,
    stamp_duty: float = DEFAULT_STAMP_DUTY,
    transfer_fee: float = DEFAULT_TRANSFER_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> float:
    """给定各票本次成交额（人民币，绝对值），返回总费用（人民币）。

    每只发生交易（成交额>0）的票：
      佣金   = max(成交额 × 佣金率, 5 元)        ← 最低 5 元在这里生效
      过户费 = 成交额 × 万0.1                     （双边）
      印花税 = 成交额 × 0.05% × 0.5              （仅卖出，0.5 系数近似买卖各半）
      滑点   = 成交额 × 0.05%                     （双边）
    """
    traded = notional[notional > 0]
    if len(traded) == 0:
        return 0.0
    commission = np.maximum(traded * commission_rate, min_commission)
    transfer = traded * transfer_fee
    stamp = traded * stamp_duty * 0.5   # 印花税仅卖出，合并口径用 0.5 近似
    slip = traded * slippage
    return float((commission + transfer + stamp + slip).sum())


def make_layered_cost_fn(
    commission_rate: float = DEFAULT_COMMISSION,
    min_commission: float = DEFAULT_MIN_COMMISSION,
    stamp_duty: float = DEFAULT_STAMP_DUTY,
    transfer_fee: float = DEFAULT_TRANSFER_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
):
    """产出与 layered_backtest / fixed_topn_portfolio 的 cost_fn 注入点兼容的回调。

    回调签名：cost_fn(weight_change: Series, prices: Series) -> fractional_cost
    每票成交额 = |权重变动| × 本金；总费用见 cn_trade_cost_yuan；返回 费用/本金。
    """
    def cost_fn(weight_change: pd.Series, prices: pd.Series) -> float:
        wc = weight_change.abs()
        valid = (wc > 0)
        if not valid.any():
            return 0.0
        notional = wc[valid] * portfolio_value
        yuan = cn_trade_cost_yuan(
            notional, commission_rate, min_commission,
            stamp_duty, transfer_fee, slippage,
        )
        return yuan / portfolio_value

    return cost_fn
