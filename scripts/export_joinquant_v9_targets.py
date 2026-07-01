#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export JoinQuant v9 target weights for third-party platform replay.

The project owns factor construction and target generation. External platforms
such as RQAlpha, Backtrader, or LEAN should first replay this target book as an
execution auditor before we rewrite the strategy natively for each engine.

Default output:
    data/rqalpha_bridge/jq_v9_target_weights.csv

To replay with the existing RQAlpha bridge:
    PYTHONPATH=. python scripts/export_joinquant_v9_targets.py --out data/rqalpha_bridge/cn_target_weights.csv
    PYTHONPATH=. python scripts/rqalpha_run_cn_bridge.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.config import DATA_DIR, RAW_DATA_DIR
from quant.data.industry import industry_series
from quant.data.panel import build_ohlcv_panels, build_value_panels
from scripts.joinquant_v9_validation import (
    BACKTEST_END,
    BACKTEST_START,
    FACTOR_START,
    build_market_equity,
    build_v9_factor,
    exposure_for,
    load_jq_strategy,
    score_weights,
    select_targets,
)


DEFAULT_OUT = DATA_DIR / "rqalpha_bridge" / "jq_v9_target_weights.csv"


def to_rqalpha_order_book_id(symbol: str) -> str:
    s = str(symbol).strip()
    if "." in s:
        return s
    if s.startswith(("6", "9")):
        return f"{s}.XSHG"
    return f"{s}.XSHE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export JoinQuant v9 target weights")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--rebalance", type=int, default=60)
    p.add_argument("--industry-cap", type=int, default=2)
    p.add_argument("--exposure", choices=["fixed95", "fixed98"], default="fixed98")
    p.add_argument("--hold-multiplier", type=float, default=1.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jq = load_jq_strategy()
    pool_all = [s.split(".")[0] for s in jq.STOCK_POOL]
    symbols = [
        s for s in pool_all
        if (RAW_DATA_DIR / f"{s}.parquet").exists()
        and (RAW_DATA_DIR / f"{s}_value.parquet").exists()
    ]
    if not symbols:
        raise RuntimeError("No cached JoinQuant pool data. Run refetch_joinquant_pool.py first.")

    panels = build_ohlcv_panels(symbols, start=FACTOR_START, end=BACKTEST_END)
    close, amount = panels["close"], panels["amount"]
    value = build_value_panels(symbols, start=FACTOR_START, end=BACKTEST_END, align_to=close)
    industry = industry_series(list(close.columns))
    industry_map = industry.to_dict()
    factor = build_v9_factor(close, amount, value, industry)
    market_equity = build_market_equity(close)
    vol60 = close.pct_change(fill_method=None).rolling(60).std()

    close_bt = close.loc[BACKTEST_START:BACKTEST_END]
    current: list[str] = []
    records = []
    for i, date in enumerate(close_bt.index):
        if i == 0 or i % args.rebalance != 0:
            continue
        full_idx = close.index.get_loc(date)
        prev_idx = max(full_idx - 1, 0)
        prices = close.loc[date]
        exposure = exposure_for(args.exposure, market_equity, prev_idx)
        scores = factor.iloc[prev_idx].dropna()
        scores = scores[scores.index.isin(prices.dropna().index)]
        selected = select_targets(
            scores,
            prices,
            industry_map,
            top_n=args.top_n,
            industry_cap=args.industry_cap,
            slot_value=60_000.0 * exposure / args.top_n,
            close_full=close,
            date_idx=prev_idx,
            current=current,
            hold_multiplier=args.hold_multiplier,
        )
        weights = score_weights(selected, vol60.iloc[prev_idx], exposure=exposure, mode="equal")
        current = list(weights.index)
        for symbol, weight in weights.items():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "order_book_id": to_rqalpha_order_book_id(symbol),
                "weight": float(weight),
                "score": float(scores.get(symbol, float("nan"))),
                "strategy": "joinquant_v9_less_cash_buffer",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(records)
    out.to_csv(args.out, index=False)
    print(f"Exported {len(out)} target rows to {args.out}")
    print(f"Cached pool coverage: {len(symbols)}/{len(pool_all)}")


if __name__ == "__main__":
    main()
