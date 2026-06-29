# -*- coding: utf-8 -*-
"""Search JoinQuant-compatible A-share strategy variants on the 2019-2025 window.

The previous v3 comparison was polluted by a start-date mismatch: JoinQuant was
tested from 2019-01-01, while the local audit file also contained 2018-start
results. This script keeps the factor lookback from 2018, but starts portfolio
capital and performance accounting at 2019-01-01.

Only use ingredients that can be reproduced inside JoinQuant:
daily price/money, valuation fields, indicator growth fields, industry and
market-cap neutralization, fixed top-N selection, 100-share lots, and A-share
fees.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.backtest.cn_cost import cn_trade_cost_yuan
from quant.backtest.metrics import summary
from quant.data.industry import industry_series
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.universe import DEFAULT_POOL
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize


OUT_DIR = PROJECT_ROOT / "jointquant" / "v2"
OUT_CSV = OUT_DIR / "joinquant_v4_aligned_search.csv"
OUT_MD = OUT_DIR / "v4_aligned_search.md"

FACTOR_START = "20180101"
BACKTEST_START = "20190101"
BACKTEST_END = "20251231"
CAPITAL = 60_000.0
MAX_EXPOSURE = 0.95


def _zscore_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def build_factor_library(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    value: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build a JoinQuant-reproducible factor library.

    Size and industry neutralization is applied before rank combination. This
    keeps the strategy from becoming a disguised sector/market-cap beta bet.
    """
    ind = industry_series(list(close.columns))
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "book_to_price": F.book_to_price(value["pb"]),
        "sales_yield": F.sales_yield(value["ps"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "growth_peg": F.growth_peg(value["peg"]),
        "quality_roe": F.quality_roe(value["pe_ttm"], value["pb"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "reversal_20": F.reversal(close, 20),
        "reversal_60": F.reversal(close, 60),
        "momentum_60": F.momentum(close, 60),
        "momentum_120": F.momentum(close, 120),
        "low_vol_20": F.low_volatility(close, 20),
        "low_vol_60": F.low_volatility(close, 60),
        "atr_low_20": F.atr_volatility(high, low, close, 20),
        "ma_slope_60": F.ma_slope(close, 60),
        "price_to_ma_60": F.price_to_ma(close, 60),
    }
    return {
        name: neutralize(factor, industry=ind, log_mv=log_mv, mode="full")
        for name, factor in raw.items()
    }


def combine_named(lib: dict[str, pd.DataFrame], names: tuple[str, ...]) -> pd.DataFrame:
    return combine_factors(*(lib[name] for name in names))


def _industry_limited_selection(
    scores: pd.Series,
    prices: pd.Series,
    industry: dict[str, str],
    top_n: int,
    industry_cap: int,
    slot_value: float,
) -> list[str]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for stock in scores.index:
        price = prices.get(stock, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        if price * 100 > slot_value * 1.15:
            continue
        ind = industry.get(stock, "其他")
        if industry_cap > 0 and counts.get(ind, 0) >= industry_cap:
            continue
        selected.append(stock)
        counts[ind] = counts.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def lot_backtest_aligned(
    close_full: pd.DataFrame,
    factor_full: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
    industry_cap: int,
    max_exposure: float = MAX_EXPOSURE,
    weight_mode: str = "equal",
) -> dict:
    """100-share-lot backtest, starting wealth at BACKTEST_START.

    On rebalance day, use the previous available factor row to avoid future
    leakage. This lets the first 2019 rebalance consume 2018 lookback history.
    """
    close_full = close_full.sort_index()
    factor_full = factor_full.reindex_like(close_full)
    close = close_full.loc[BACKTEST_START:BACKTEST_END]
    industry = industry_series(list(close.columns)).to_dict()

    cash = CAPITAL
    shares = pd.Series(0.0, index=close.columns)
    values: list[float] = []
    returns: list[float] = []
    turnovers: list[float] = []
    holdings: list[int] = []
    fees: list[float] = []
    cash_ratios: list[float] = []
    prev_value = CAPITAL

    for i, date in enumerate(close.index):
        prices = close.loc[date]
        stock_value = float((shares * prices.fillna(0.0)).sum())
        total_value = cash + stock_value
        should_rebalance = i % rebalance_every == 0

        if should_rebalance:
            factor_hist = factor_full.loc[:date]
            if len(factor_hist) < 2:
                scores = pd.Series(dtype=float)
            else:
                scores = factor_hist.iloc[-2].dropna().sort_values(ascending=False)
                scores = scores[scores.index.isin(prices.dropna().index)]

            slot = total_value * max_exposure / top_n
            selected = _industry_limited_selection(scores, prices, industry, top_n, industry_cap, slot)

            target_shares = pd.Series(0.0, index=close.columns)
            if selected:
                if weight_mode == "score":
                    raw = scores.reindex(selected).rank(pct=True).clip(lower=0.2)
                    weights = raw / raw.sum()
                elif weight_mode == "inverse_vol":
                    ret_hist = close_full.loc[:date, selected].pct_change(fill_method=None).tail(60)
                    inv = 1.0 / ret_hist.std().replace(0, np.nan)
                    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                    weights = inv / inv.sum()
                else:
                    weights = pd.Series(1.0 / len(selected), index=selected)

                for stock in selected:
                    target_value = total_value * max_exposure * float(weights[stock])
                    target_shares[stock] = math.floor(target_value / prices[stock] / 100.0) * 100.0

            delta = target_shares - shares
            notional = (delta.abs() * prices).fillna(0.0)
            fee = cn_trade_cost_yuan(notional, slippage=0.0005)
            cash = total_value - float((target_shares * prices.fillna(0.0)).sum()) - fee
            shares = target_shares
            stock_value = float((shares * prices.fillna(0.0)).sum())
            total_value = cash + stock_value

            turnovers.append(float(notional.sum() / max(total_value, 1.0)))
            holdings.append(int((shares > 0).sum()))
            fees.append(float(fee))

        values.append(total_value)
        returns.append(total_value / prev_value - 1.0 if len(values) > 1 else 0.0)
        cash_ratios.append(cash / total_value if total_value > 0 else 0.0)
        prev_value = total_value

    ret = pd.Series(returns, index=close.index)
    equity = pd.Series(values, index=close.index) / CAPITAL
    out = summary(equity, ret)
    out.update(
        {
            "final_value": float(values[-1]),
            "avg_turnover": float(np.mean(turnovers)),
            "avg_holdings": float(np.mean(holdings)),
            "avg_fee": float(np.mean(fees)),
            "avg_cash": float(np.mean(cash_ratios)),
        }
    )
    return out


def candidate_factor_sets() -> list[tuple[str, tuple[str, ...]]]:
    return [
        (
            "v3_full",
            ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud"),
        ),
        (
            "value_cash_growth",
            ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg"),
        ),
        (
            "value_quality_cash_growth",
            ("earnings_yield", "cashflow_yield", "quality_roe", "growth_peg"),
        ),
        (
            "quality_growth_momentum",
            ("quality_roe", "growth_peg", "momentum_60", "ma_slope_60"),
        ),
        (
            "value_lowvol",
            ("earnings_yield", "cashflow_yield", "sales_yield", "low_vol_60"),
        ),
        (
            "cash_growth_lowvol",
            ("cashflow_yield", "growth_peg", "quality_roe", "low_vol_60"),
        ),
        (
            "value_cash_growth_lowvol",
            ("earnings_yield", "cashflow_yield", "growth_peg", "low_vol_60"),
        ),
        (
            "value_cash_quality_lowvol",
            ("earnings_yield", "cashflow_yield", "quality_roe", "low_vol_60"),
        ),
        (
            "value_growth_lowvol_momo",
            ("earnings_yield", "growth_peg", "low_vol_60", "momentum_60"),
        ),
        (
            "cash_growth_quality_momo",
            ("cashflow_yield", "growth_peg", "quality_roe", "momentum_60"),
        ),
        (
            "v3_plus_lowvol",
            ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud", "low_vol_60"),
        ),
        (
            "v3_plus_quality",
            ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud", "quality_roe"),
        ),
        (
            "v3_no_amihud_quality",
            ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "quality_roe"),
        ),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = build_ohlcv_panels(DEFAULT_POOL, start=FACTOR_START, end=BACKTEST_END)
    close = panels["close"]
    high = panels["high"]
    low = panels["low"]
    amount = panels["amount"]
    value = build_value_panels(DEFAULT_POOL, start=FACTOR_START, end=BACKTEST_END, align_to=close)

    lib = build_factor_library(close, high, low, amount, value)
    rows = []
    for factor_name, names in candidate_factor_sets():
        factor = combine_named(lib, names)
        for top_n in (6, 8, 10, 12, 15):
            for rebalance in (20, 30, 40, 60):
                for industry_cap in (1, 2):
                    for weight_mode in ("equal", "inverse_vol"):
                        m = lot_backtest_aligned(
                            close,
                            factor,
                            top_n=top_n,
                            rebalance_every=rebalance,
                            industry_cap=industry_cap,
                            weight_mode=weight_mode,
                        )
                        rows.append(
                            {
                                "factor_name": factor_name,
                                "factors": ",".join(names),
                                "top_n": top_n,
                                "rebalance_days": rebalance,
                                "industry_cap": industry_cap,
                                "weight_mode": weight_mode,
                                **m,
                            }
                        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["sharpe", "annualized_return", "total_return"], ascending=False)
    df.to_csv(OUT_CSV, index=False)

    top = df.head(20)
    lines = [
        "# JoinQuant v4 Aligned Search",
        "",
        f"- Local window: {BACKTEST_START} ~ {BACKTEST_END}",
        f"- Factor lookback starts: {FACTOR_START}",
        "- Constraint: JoinQuant-reproducible fields, 100-share lots, A-share fees, 95% max exposure.",
        "",
        "## Top 20 by Sharpe",
        "",
        top[
            [
                "factor_name",
                "factors",
                "top_n",
                "rebalance_days",
                "industry_cap",
                "weight_mode",
                "total_return",
                "annualized_return",
                "sharpe",
                "max_drawdown",
                "avg_cash",
                "avg_turnover",
            ]
        ].to_markdown(index=False),
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_MD}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
