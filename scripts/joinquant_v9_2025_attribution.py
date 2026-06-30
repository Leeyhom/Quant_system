#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Attribute the real JoinQuant v9 export around the 2025 cold-start concern.

This script only reads JoinQuant's exported transaction/position/log files.
It deliberately does not rebuild factors, so the attribution reflects what
JoinQuant actually executed rather than a local approximation.

Run:
    PYTHONPATH=. python scripts/joinquant_v9_2025_attribution.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_joinquant_exports import (  # noqa: E402
    build_daily,
    parse_position,
    parse_rebalance_log,
    parse_transaction,
)


JQ_DIR = PROJECT_ROOT / "jointquant" / "v9"
OUT_MD = JQ_DIR / "v9_2025_attribution.md"
OUT_STOCK = JQ_DIR / "v9_2025_stock_pnl.csv"
OUT_INDUSTRY = JQ_DIR / "v9_2025_industry_pnl.csv"
OUT_MONTHLY = JQ_DIR / "v9_2025_monthly_path.csv"
PERIOD_START = pd.Timestamp("2025-01-01")
PERIOD_END = pd.Timestamp("2025-12-31")


def load_v9_strategy():
    path = PROJECT_ROOT / "scripts" / "joinquant_cn_sim_strategy_v9.py"
    spec = importlib.util.spec_from_file_location("jq_v9_strategy_for_attr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def period_return(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    before = daily.loc[daily.index < start]
    inside = daily.loc[(daily.index >= start) & (daily.index <= end)].copy()
    if before.empty or inside.empty:
        raise ValueError("Daily metrics do not cover the requested period.")

    start_value = float(before["total_value"].iloc[-1])
    end_value = float(inside["total_value"].iloc[-1])
    min_value = float(inside["total_value"].min())
    max_value = float(inside["total_value"].max())
    period_equity = pd.concat(
        [
            pd.Series([start_value], index=[before.index[-1]]),
            inside["total_value"],
        ]
    )
    dd = period_equity / period_equity.cummax() - 1.0
    stats = {
        "start_anchor": before.index[-1],
        "start_value": start_value,
        "end_value": end_value,
        "period_return": end_value / start_value - 1.0,
        "min_value": min_value,
        "max_value": max_value,
        "max_drawdown": -float(dd.min()),
        "max_drawdown_end": dd.idxmin(),
        "avg_exposure": float(inside["exposure"].mean()),
        "avg_cash": float(inside["cash_ratio"].mean()),
        "avg_holdings": float(inside["holdings"].mean()),
    }
    return inside, stats


def monthly_path(daily_2025: pd.DataFrame, start_anchor_value: float) -> pd.DataFrame:
    rows = []
    prev_value = start_anchor_value
    for month, part in daily_2025.groupby(daily_2025.index.to_period("M")):
        end_value = float(part["total_value"].iloc[-1])
        rows.append(
            {
                "month": str(month),
                "start_value": prev_value,
                "end_value": end_value,
                "return": end_value / prev_value - 1.0,
                "avg_exposure": float(part["exposure"].mean()),
                "min_drawdown_in_full_run": float(part["drawdown"].min()),
            }
        )
        prev_value = end_value
    return pd.DataFrame(rows)


def stock_attribution(stock: pd.DataFrame, trx: pd.DataFrame, strategy) -> tuple[pd.DataFrame, pd.DataFrame]:
    s2025 = stock[(stock["date"] >= PERIOD_START) & (stock["date"] <= PERIOD_END)].copy()
    t2025 = trx[(trx["date"] >= PERIOD_START) & (trx["date"] <= PERIOD_END)].copy()
    code6 = s2025["code"].astype(str).str[:6]
    s2025["industry"] = code6.map(strategy.INDUSTRY_BY_STOCK).fillna("其他")

    stock_pnl = s2025.groupby("code").agg(
        name=("name", "last"),
        industry=("industry", "last"),
        held_days=("date", "nunique"),
        day_pnl=("day_pnl", "sum"),
        avg_value=("value", "mean"),
        max_value=("value", "max"),
        ending_value=("value", "last"),
        ending_float_pnl=("float_pnl", "last"),
    )
    stock_pnl["pnl_on_avg_value"] = stock_pnl["day_pnl"] / stock_pnl["avg_value"].replace(0, np.nan)

    sells = t2025[t2025["side"].eq("卖")].copy()
    if not sells.empty:
        realized = sells.groupby("code").agg(
            realized_pnl=("pnl", "sum"),
            sell_amount=("amount", lambda x: -x.sum()),
            sell_trades=("code", "count"),
        )
        stock_pnl = stock_pnl.join(realized, how="left")
    for col in ["realized_pnl", "sell_amount", "sell_trades"]:
        if col not in stock_pnl:
            stock_pnl[col] = 0.0
    stock_pnl[["realized_pnl", "sell_amount", "sell_trades"]] = stock_pnl[
        ["realized_pnl", "sell_amount", "sell_trades"]
    ].fillna(0.0)

    industry_pnl = stock_pnl.groupby("industry").agg(
        names=("name", "count"),
        day_pnl=("day_pnl", "sum"),
        avg_value=("avg_value", "sum"),
        realized_pnl=("realized_pnl", "sum"),
        held_days=("held_days", "sum"),
    )
    industry_pnl["pnl_on_avg_value"] = industry_pnl["day_pnl"] / industry_pnl["avg_value"].replace(0, np.nan)
    return stock_pnl.sort_values("day_pnl"), industry_pnl.sort_values("day_pnl")


def main() -> None:
    strategy = load_v9_strategy()
    trx = parse_transaction(JQ_DIR)
    stock, cash = parse_position(JQ_DIR)
    daily = build_daily(stock, cash)
    daily_2025, stats = period_return(daily, PERIOD_START, PERIOD_END)
    monthly = monthly_path(daily_2025, stats["start_value"])
    reb = parse_rebalance_log(JQ_DIR, "v9")
    reb_2025 = reb[(reb["date"] >= PERIOD_START) & (reb["date"] <= PERIOD_END)].copy()
    stock_pnl, industry_pnl = stock_attribution(stock, trx, strategy)

    stock_pnl.to_csv(OUT_STOCK, encoding="utf-8")
    industry_pnl.to_csv(OUT_INDUSTRY, encoding="utf-8")
    monthly.to_csv(OUT_MONTHLY, index=False, encoding="utf-8")

    start_holdings = (
        stock[stock["date"].eq(daily_2025.index[0])]
        .sort_values("value", ascending=False)[["code", "name", "value", "weight_pct", "day_pnl"]]
    )
    worst = stock_pnl.head(12)
    best = stock_pnl.sort_values("day_pnl", ascending=False).head(12)

    lines = [
        "# JoinQuant v9 2025 Attribution",
        "",
        "## Period Path",
        "",
        f"- Anchor: {stats['start_anchor'].date()} close -> {PERIOD_END.date()} close",
        f"- Start value: {stats['start_value']:.2f}",
        f"- End value: {stats['end_value']:.2f}",
        f"- 2025 return from prior close: {stats['period_return']:.2%}",
        f"- Max drawdown inside 2025 path: {stats['max_drawdown']:.2%}, ended {stats['max_drawdown_end'].date()}",
        f"- Average exposure: {stats['avg_exposure']:.2%}",
        f"- Average cash: {stats['avg_cash']:.2%}",
        f"- Average holdings: {stats['avg_holdings']:.2f}",
        "",
        "## Monthly Path",
        "",
        monthly.to_markdown(index=False),
        "",
        "## 2025 Rebalance Targets",
        "",
        reb_2025[["date", "candidate_count", "score_count", "targets"]].to_markdown(index=False),
        "",
        "## First 2025 Trading Day Holdings",
        "",
        start_holdings.to_markdown(index=False),
        "",
        "## Industry PnL Attribution",
        "",
        industry_pnl.to_markdown(),
        "",
        "## Worst Stock Day-PnL Contributors",
        "",
        worst[
            ["name", "industry", "held_days", "day_pnl", "pnl_on_avg_value", "realized_pnl", "ending_float_pnl"]
        ].to_markdown(),
        "",
        "## Best Stock Day-PnL Contributors",
        "",
        best[
            ["name", "industry", "held_days", "day_pnl", "pnl_on_avg_value", "realized_pnl", "ending_float_pnl"]
        ].to_markdown(),
        "",
        "## Outputs",
        "",
        f"- {OUT_STOCK.name}",
        f"- {OUT_INDUSTRY.name}",
        f"- {OUT_MONTHLY.name}",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_MD}")
    print(f"[ok] wrote {OUT_STOCK}")
    print(f"[ok] wrote {OUT_INDUSTRY}")
    print(f"[ok] wrote {OUT_MONTHLY}")
    print("\n".join(lines[:48]))


if __name__ == "__main__":
    main()
