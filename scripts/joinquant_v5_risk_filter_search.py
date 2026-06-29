# -*- coding: utf-8 -*-
"""Search v5 candidates after JoinQuant v4 failed out of sample.

v4 taught us that concentrating top-6 value picks amplifies value traps in the
JoinQuant data/execution environment. This search starts from the proven v3
shape and tests only simple, live-replicable risk filters:

- avoid deeply negative 120-day momentum
- avoid names far below their 120-day moving average
- penalize high 60-day volatility

The goal is not to maximize local in-sample return again, but to find a version
that keeps v3's diversification while reducing the v4 drawdown mechanism.
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


OUT_DIR = PROJECT_ROOT / "jointquant" / "v4"
OUT_CSV = OUT_DIR / "joinquant_v5_risk_filter_search.csv"
OUT_MD = OUT_DIR / "v5_risk_filter_search.md"

FACTOR_START = "20180101"
BACKTEST_START = "20190101"
BACKTEST_END = "20251231"
CAPITAL = 60_000.0
MAX_EXPOSURE = 0.95


def build_library(close: pd.DataFrame, amount: pd.DataFrame, value: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ind = industry_series(list(close.columns))
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "quality_roe": F.quality_roe(value["pe_ttm"], value["pb"]),
        "low_vol_60": F.low_volatility(close, 60),
        "momentum_120": F.momentum(close, 120),
        "price_to_ma120": F.price_to_ma(close, 120),
    }
    out = {
        name: neutralize(factor, industry=ind, log_mv=log_mv, mode="full")
        for name, factor in raw.items()
        if name not in {"momentum_120", "price_to_ma120"}
    }
    out["momentum_120_raw"] = raw["momentum_120"]
    out["price_to_ma120_raw"] = raw["price_to_ma120"]
    return out


def combine_named(lib: dict[str, pd.DataFrame], names: tuple[str, ...], weights: tuple[float, ...] | None = None) -> pd.DataFrame:
    if weights is None:
        return combine_factors(*(lib[n] for n in names))
    ranked = [lib[n].rank(axis=1, pct=True) * w for n, w in zip(names, weights)]
    return sum(ranked) / sum(weights)


def _passes_filter(stock: str, date_idx: int, close: pd.DataFrame, mode: str) -> bool:
    if mode == "none":
        return True
    prices = close[stock].iloc[: date_idx + 1].dropna()
    if len(prices) < 130:
        return True
    px = prices.iloc[-1]
    mom120 = px / prices.iloc[-121] - 1.0
    ma120 = prices.tail(120).mean()
    pma = px / ma120 - 1.0 if ma120 > 0 else 0.0
    dd120 = px / prices.tail(120).max() - 1.0
    if mode == "mom120_gt_neg10":
        return mom120 > -0.10
    if mode == "mom120_gt_neg20":
        return mom120 > -0.20
    if mode == "not_deep_downtrend":
        return not (mom120 < -0.15 and pma < -0.08)
    if mode == "not_120dd25":
        return dd120 > -0.25
    if mode == "trend_combo":
        return mom120 > -0.20 and pma > -0.12 and dd120 > -0.30
    raise ValueError(mode)


def select_targets(
    scores: pd.Series,
    prices: pd.Series,
    industry: dict[str, str],
    top_n: int,
    industry_cap: int,
    slot_value: float,
    close_full: pd.DataFrame,
    date_idx: int,
    filter_mode: str,
) -> list[str]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for stock in scores.index:
        price = prices.get(stock, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        if price * 100 > slot_value * 1.15:
            continue
        if not _passes_filter(stock, date_idx, close_full, filter_mode):
            continue
        ind = industry.get(stock, "其他")
        if industry_cap > 0 and counts.get(ind, 0) >= industry_cap:
            continue
        selected.append(stock)
        counts[ind] = counts.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def lot_backtest(
    close_full: pd.DataFrame,
    factor_full: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
    industry_cap: int,
    filter_mode: str,
) -> dict:
    close_full = close_full.sort_index()
    factor_full = factor_full.reindex_like(close_full)
    close = close_full.loc[BACKTEST_START:BACKTEST_END]
    industry = industry_series(list(close.columns)).to_dict()
    full_dates = close_full.index

    cash = CAPITAL
    shares = pd.Series(0.0, index=close.columns)
    values: list[float] = []
    returns: list[float] = []
    turnovers: list[float] = []
    holdings: list[int] = []
    fees: list[float] = []
    cash_ratios: list[float] = []
    skipped_slots: list[int] = []
    prev_value = CAPITAL

    for i, date in enumerate(close.index):
        prices = close.loc[date]
        total_value = cash + float((shares * prices.fillna(0.0)).sum())
        if i % rebalance_every == 0:
            date_idx = full_dates.get_loc(date)
            scores = factor_full.iloc[date_idx - 1].dropna().sort_values(ascending=False)
            scores = scores[scores.index.isin(prices.dropna().index)]
            slot = total_value * MAX_EXPOSURE / top_n
            selected = select_targets(
                scores,
                prices,
                industry,
                top_n,
                industry_cap,
                slot,
                close_full,
                date_idx - 1,
                filter_mode,
            )

            target_shares = pd.Series(0.0, index=close.columns)
            if selected:
                slot = total_value * MAX_EXPOSURE / len(selected)
                for stock in selected:
                    target_shares[stock] = math.floor(slot / prices[stock] / 100.0) * 100.0

            delta = target_shares - shares
            notional = (delta.abs() * prices).fillna(0.0)
            fee = cn_trade_cost_yuan(notional, slippage=0.0005)
            cash = total_value - float((target_shares * prices.fillna(0.0)).sum()) - fee
            shares = target_shares
            total_value = cash + float((shares * prices.fillna(0.0)).sum())
            turnovers.append(float(notional.sum() / max(total_value, 1.0)))
            holdings.append(int((shares > 0).sum()))
            fees.append(float(fee))
            skipped_slots.append(top_n - len(selected))

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
            "avg_skipped_slots": float(np.mean(skipped_slots)),
        }
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = build_ohlcv_panels(DEFAULT_POOL, start=FACTOR_START, end=BACKTEST_END)
    close = panels["close"]
    amount = panels["amount"]
    value = build_value_panels(DEFAULT_POOL, start=FACTOR_START, end=BACKTEST_END, align_to=close)
    lib = build_library(close, amount, value)

    factor_defs = [
        ("v3_full", ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud"), None),
        ("v3_plus_quality", ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud", "quality_roe"), None),
        ("v3_plus_lowvol", ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud", "low_vol_60"), None),
        ("quality_half", ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud", "quality_roe"), (1, 1, 1, 1, 1, 0.5)),
        ("lowvol_half", ("earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud", "low_vol_60"), (1, 1, 1, 1, 1, 0.5)),
    ]
    filter_modes = [
        "none",
        "mom120_gt_neg10",
        "mom120_gt_neg20",
        "not_deep_downtrend",
        "not_120dd25",
        "trend_combo",
    ]
    rows = []
    for factor_name, names, weights in factor_defs:
        factor = combine_named(lib, names, weights)
        for top_n in (8, 10, 12):
            for rebalance in (30, 40, 50, 60):
                for industry_cap in (1, 2):
                    for filter_mode in filter_modes:
                        m = lot_backtest(close, factor, top_n, rebalance, industry_cap, filter_mode)
                        rows.append(
                            {
                                "factor_name": factor_name,
                                "factors": ",".join(names),
                                "weights": "" if weights is None else ",".join(map(str, weights)),
                                "top_n": top_n,
                                "rebalance_days": rebalance,
                                "industry_cap": industry_cap,
                                "filter_mode": filter_mode,
                                **m,
                            }
                        )

    df = pd.DataFrame(rows).sort_values(["sharpe", "annualized_return"], ascending=False)
    df.to_csv(OUT_CSV, index=False)
    top = df.head(25)
    lines = [
        "# JoinQuant v5 Risk Filter Search",
        "",
        f"- Window: {BACKTEST_START} ~ {BACKTEST_END}",
        "- Search starts from v3/v4 evidence, not a wide in-sample optimization.",
        "",
        "## Top 25 By Sharpe",
        "",
        top[
            [
                "factor_name",
                "top_n",
                "rebalance_days",
                "industry_cap",
                "filter_mode",
                "total_return",
                "annualized_return",
                "sharpe",
                "max_drawdown",
                "avg_cash",
                "avg_holdings",
                "avg_skipped_slots",
            ]
        ].to_markdown(index=False),
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_MD}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
