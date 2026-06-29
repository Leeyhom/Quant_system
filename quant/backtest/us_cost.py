"""us_cost —— 美股专属交易费用模型 + 做空可用性（v3，2026-06-27升级）。

v1 (M17)：仅每股费+每笔最低费，未区分多空。
v2 升级：做空可用性+借券费率。
v3 升级（对齐用户星财富实盘 1w 美元本金）：
  ① 名义本金 100万 → 1万（DEFAULT_PORTFOLIO_VALUE）。$1/笔是绝对固定费、
     不随本金缩放——100万口径下费率被低估 100 倍，必须按真实本金算。
  ② 每股费 0.005 → 0（用户「一世免佣」，仅剩 $1 平台费 + 极小交收费）。
     保留 per_share 参数以便将来切到非免佣券商时复用。
v2 升级要点（保留）：
  ① 明确 $1 每笔最低平台费（用户确认）
  ② 做空可用性 + 动态借券费率：按市值分档
     - Mega-cap (>$100B): ETB, 借券费 0.5%/年
     - Large-cap ($10B-$100B): ETB, 借券费 1.5%/年
     - Mid-cap ($1B-$10B): HTB可能, 借券费 5%/年
     - Small-cap (<$1B): 非常难借, 借券费 15%/年（或不许做空）
  ③ T+0 说明：美股T+0允许当日买卖，本文模型为日频（用收盘价），
     T+0只影响日内执行，不影响因子/回测框架。
  ④ 做空保证金利息：保守假设卖空所得无利息（各券商差异大）。

设计：
  - us_trade_cost_dollars(shares, n_trades, ...)：纯函数，计算美元费用。
  - make_layered_cost_fn(...)：工厂，产 cost_fn 回调（与layered_backtest兼容）。
  - estimate_borrow_rate(symbol, close, shares_out=None)：按市值估算借券费率。
  - make_short_borrow_cost_fn(...)：工厂，产多空组合的借券费回调。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── 费用默认参数（v3：对齐用户星财富实盘 1w 本金 + 一世免佣） ───
DEFAULT_PER_SHARE = 0.0            # 一世免佣 → 每股费 0（保留参数以便非免佣券商复用）
DEFAULT_MIN_PER_TRADE = 1.0        # 每笔最低 $1.00（星财富平台费，用户确认）
DEFAULT_PORTFOLIO_VALUE = 10_000.0  # 名义组合市值 = 实盘本金 1w 美元（旧值 100万低估费率100倍）

# ─── 借券费率分档（年化） ───
# 市值阈值与费率——基于美股市场实践的大致估计。
# 实盘时需从券商(IBKR/Alpaca)实时获取每只票的 borrow_rate。
MEGA_CAP_THRESHOLD = 100e9   # $100B
LARGE_CAP_THRESHOLD = 10e9   # $10B
MID_CAP_THRESHOLD = 1e9      # $1B
BORROW_RATE_MEGA = 0.005     # 0.5%/年
BORROW_RATE_LARGE = 0.015    # 1.5%/年
BORROW_RATE_MID = 0.05       # 5%/年
BORROW_RATE_SMALL = 0.15     # 15%/年（或不许做空）


def us_trade_cost_dollars(
    shares: pd.Series,
    per_share: float = DEFAULT_PER_SHARE,
    min_per_trade: float = DEFAULT_MIN_PER_TRADE,
) -> float:
    """给定各票当次换手股数，返回总美元费用 = Σ max(每股费×股数, 每笔最低费)。

    参数:
        shares: 各股票本次买/卖的股数（绝对值，0=不交易该票）。
        per_share/min_per_trade: 每股费、每笔最低费。

    每只发生交易（股数>0）的票按 max(per_share×shares, min_per_trade) 计费，
    刻画「小额也要付最低费」的现实。不交易的票不计费。
    """
    traded = shares[shares > 0]
    if len(traded) == 0:
        return 0.0
    per_trade = np.maximum(traded * per_share, min_per_trade)
    return float(per_trade.sum())


def make_layered_cost_fn(
    per_share: float = DEFAULT_PER_SHARE,
    min_per_trade: float = DEFAULT_MIN_PER_TRADE,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
):
    """产出与 layered_backtest 的 cost_fn 注入点兼容的回调。

    回调签名：cost_fn(weight_change, prices) -> fractional_cost
    换算：每票名义成交额 = |权重变动| × 组合市值；股数 = 名义成交额 / 价格。
    美元费用 = Σ max(每股费×股数, $1 最低平台费)。
    返回收益拖累（美元费用 / 组合市值）。
    """
    def cost_fn(weight_change: pd.Series, prices: pd.Series) -> float:
        wc = weight_change.abs()
        valid = prices.notna() & (prices > 0) & (wc > 0)
        if not valid.any():
            return 0.0
        notional = wc[valid] * portfolio_value
        shares = notional / prices[valid]
        dollars = us_trade_cost_dollars(shares, per_share, min_per_trade)
        return dollars / portfolio_value

    return cost_fn


def estimate_borrow_rate(
    symbol: str,
    close_price: float,
    shares_outstanding: float | None = None,
) -> float:
    """按市值估算年化借券费率。

    输入:
        symbol: 股票代码。
        close_price: 当日价格。
        shares_outstanding: 总股本（如果可从基本面获取）。若为None，仅用价格估算。

    返回: 年化借券费率（如 0.02 = 2%/年）。
    市值 > $100B: 0.5%/年 | $10B-$100B: 1.5%/年 | $1B-$10B: 5%/年 | <$1B: 15%/年。
    若 shares_outstanding 不可得，用价格作弱代理（低价股往往小市值）。
    """
    if shares_outstanding is not None and shares_outstanding > 0:
        mkt_cap = close_price * shares_outstanding
    else:
        # 弱代理：假设股本 ~1B（对S&P500成分股合理，对小盘股高估）
        mkt_cap = close_price * 1e9

    if mkt_cap >= MEGA_CAP_THRESHOLD:
        return BORROW_RATE_MEGA
    elif mkt_cap >= LARGE_CAP_THRESHOLD:
        return BORROW_RATE_LARGE
    elif mkt_cap >= MID_CAP_THRESHOLD:
        return BORROW_RATE_MID
    else:
        return BORROW_RATE_SMALL


def is_shortable(
    symbol: str,
    close_price: float,
    shares_outstanding: float | None = None,
    min_mkt_cap: float = 500e6,  # $500M 以下不做空（流动性风险）
) -> bool:
    """判断股票是否可做空（基于市值代理）。

    回测中无法实时获取各券商的可做空清单，用市值作为粗糙代理：
    - 市值 > $500M 且价格 > $5：认为可做空
    - 市值 < $500M 或价格 < $5：认为不可做空（借不到券 / 流动性太差）

    实盘时需替换为券商实时数据（IBKR的 shortable 字段 / Alpaca的 easy_to_borrow）。
    """
    if close_price < 5.0:
        return False
    if shares_outstanding is not None and shares_outstanding > 0:
        return close_price * shares_outstanding >= min_mkt_cap
    return close_price * 1e9 >= min_mkt_cap  # 若不知股本，默认假设可做空


def short_borrow_daily_cost(
    symbol: str,
    close_price: float,
    shares_outstanding: float | None = None,
) -> float:
    """单只股票单日借券费（占空头市值的比例）= 年化费率 / 252。"""
    rate = estimate_borrow_rate(symbol, close_price, shares_outstanding)
    return rate / 252.0
