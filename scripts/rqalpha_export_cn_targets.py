"""Export current A-share strategy target weights for RQAlpha replay.

This is a bridge, not a strategy migration.  The project still owns factor
construction and target generation; RQAlpha only acts as an independent
execution/matching engine by reading the exported target weights.

Outputs:
    data/rqalpha_bridge/cn_target_weights.csv
    data/rqalpha_bridge/cn_self_backtest.csv

The first file is consumed by ``scripts/rqalpha_cn_replay_strategy.py``.
The second file is the local framework baseline using the same target logic.

Factor sets:
  v3_old  (default) - legacy factor set from quant_engine.py, for backward
                       compatibility and engine-level difference measurements.
  v4_ssot - SSOT-aligned factors (earnings/cashflow/sales yield, growth from
             year-over-year, amihud, holder_concentration), aligned with
             JoinQuant strategy v4, using honest walk-forward orientation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.backtest.cn_cost import make_layered_cost_fn
from quant.backtest.ic_analysis import daily_ic, forward_returns, ic_summary
from quant.backtest.layered import fixed_topn_portfolio
from quant.config import DATA_DIR, HISTORY_END, HISTORY_START
from quant.factor.factors import combine_factors
from scripts.quant_engine import QuantMarket

OUT_DIR = DATA_DIR / "rqalpha_bridge"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_rqalpha_order_book_id(symbol: str) -> str:
    """Convert project 6-digit A-share code to RQAlpha order_book_id."""
    s = str(symbol).strip()
    if "." in s:
        return s
    if s.startswith(("6", "9")):
        return f"{s}.XSHG"
    return f"{s}.XSHE"


def build_full_sample_composite(close: pd.DataFrame, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Orient factors by full-sample IC and combine.

    This intentionally matches the current report-style strategy construction.
    It is not an alpha-valid OOS protocol; for this bridge we want identical
    target weights so that RQAlpha can audit execution differences.
    """
    fwd = forward_returns(close, horizon=20)
    oriented = []
    rows = []
    for name, fac in factors.items():
        s = ic_summary(daily_ic(fac, fwd))
        ic = s["mean_ic"]
        oriented.append(fac if pd.isna(ic) or ic >= 0 else -fac)
        rows.append({
            "factor": name,
            "mean_ic": ic,
            "t_stat": s["t_stat"],
            "direction": "+" if pd.isna(ic) or ic >= 0 else "-",
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "cn_factor_directions.csv", index=False)
    return combine_factors(*oriented)


def _train_signs(lib: dict, fwd_ret: pd.DataFrame, upto_date, horizon: int = 20) -> dict:
    """Orient factors using only train segment IC (honest, no look-ahead)."""
    fwd_train = fwd_ret.loc[fwd_ret.index < upto_date]
    if len(fwd_train) > horizon:
        fwd_train = fwd_train.iloc[:-horizon]
    signs = {}
    for name, fac in lib.items():
        s = ic_summary(daily_ic(fac, fwd_train, min_count=5))
        ic = float(s["mean_ic"]) if pd.notna(s["mean_ic"]) else 0.0
        signs[name] = 1.0 if ic >= 0 else -1.0
    return signs


def _oriented_equal_composite(lib: dict, signs: dict) -> pd.DataFrame:
    """Equal weight composite after orienting each factor by train IC sign."""
    oriented = [(lib[n] if signs[n] >= 0 else -lib[n]) for n in lib]
    return combine_factors(*oriented)


def build_honest_composite(close: pd.DataFrame, factors: dict[str, pd.DataFrame], train_days: int = 480) -> pd.DataFrame:
    """OOS honest composite: only train IC to orient, no full-sample leakage.

    Walk-forward orientation: every test window uses only its train segment
    to decide factor direction. No leakage from the future.
    """
    fwd_ret = forward_returns(close, horizon=20)
    dates = close.index
    n = len(dates)

    oriented_chunks = []
    start = 0
    step = 120  # test_days
    while start + train_days < n:
        test_start = start + train_days
        test_end = min(test_start + step, n)
        cut = dates[test_start]

        signs = _train_signs(factors, fwd_ret, cut, 20)
        comp = _oriented_equal_composite(factors, signs)

        oriented_chunks.append(comp.iloc[test_start:test_end])
        start += step

    full_comp = pd.concat(oriented_chunks).sort_index()
    return full_comp.reindex(dates)


def export_target_weights(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
) -> pd.DataFrame:
    """Generate rebalance-date target weights using yesterday's factor."""
    close = close.sort_index()
    factor = factor.reindex_like(close)
    records = []

    for i, dt in enumerate(close.index):
        if i == 0 or i % rebalance_every != 0:
            continue
        scores = factor.iloc[i - 1].dropna().sort_values(ascending=False)
        if scores.empty:
            continue
        selected = scores.head(min(top_n, len(scores))).index.tolist()
        weight = 1.0 / len(selected)
        for symbol in selected:
            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "order_book_id": to_rqalpha_order_book_id(symbol),
                "weight": weight,
                "score": float(scores.loc[symbol]),
            })

    out = pd.DataFrame(records)
    out.to_csv(OUT_DIR / "cn_target_weights.csv", index=False)
    return out


def load_v4_ssot_factors():
    """Load v4 SSOT factors: earnings_yield/cashflow_yield/sales_yield/growth/
    amihud + holder_concentration, industry+size neutralized.
    """
    from quant.data.universe import DEFAULT_POOL
    from quant.data.panel import build_ohlcv_panels, build_value_panels, build_cn_quarterly_panels, build_cn_holder_panels
    from quant.data.industry import industry_series
    from quant.factor import factors as F
    from quant.factor.neutralize import neutralize

    symbols = DEFAULT_POOL
    ohlcv = build_ohlcv_panels(symbols)
    close = ohlcv["close"]
    val = build_value_panels(symbols, align_to=close)
    q = build_cn_quarterly_panels(symbols, align_to=close)
    hp = build_cn_holder_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))

    factors = {}
    for name, fac in {
        "earnings_yield": F.earnings_yield(val["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(val["pcf"]),
        "sales_yield": F.sales_yield(val["ps"]),
        "growth": F.growth_yoy_over_pe(q["net_profit_yoy"], val["pe_ttm"]),
        "amihud": F.amihud_illiquidity(close, ohlcv["amount"], 20),
        "holder_concentration": F.holder_concentration(hp["change_ratio"]),
    }.items():
        factors[name] = neutralize(fac, industry=ind, mode="full")
    return close, factors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CN target weights for RQAlpha execution replay")
    p.add_argument("--start", default=HISTORY_START)
    p.add_argument("--end", default=HISTORY_END)
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--rebalance", type=int, default=60)
    p.add_argument("--capital", type=float, default=60_000.0)
    p.add_argument("--factor-set", default="v3_old", choices=["v3_old", "v4_ssot"],
                   help="Which factor set to use (default: v3_old, legacy QuantMarket factors)")
    p.add_argument("--orientation", default="full_sample", choices=["full_sample", "honest"],
                   help="Factor orientation method: full_sample (use full period IC for engine diffs), "
                        "honest (walk-forward train-only IC for honest OOS comparison with JoinQuant)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.factor_set == "v3_old":
        close, factors = QuantMarket.load_cn()
    else:
        close, factors = load_v4_ssot_factors()

    close = close.loc[pd.to_datetime(args.start):pd.to_datetime(args.end)]
    factors = {name: fac.reindex_like(close) for name, fac in factors.items()}

    if args.orientation == "full_sample":
        composite = build_full_sample_composite(close, factors)
    else:
        composite = build_honest_composite(close, factors)

    weights = export_target_weights(close, composite, args.top_n, args.rebalance)

    cost_fn = make_layered_cost_fn(portfolio_value=args.capital)
    bt = fixed_topn_portfolio(
        close,
        composite,
        top_n=args.top_n,
        rebalance_every=args.rebalance,
        cost_fn=cost_fn,
    )
    bt.to_csv(OUT_DIR / "cn_self_backtest.csv", index_label="date")

    summary = {
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "n_days": int(len(close)),
        "n_stocks": int(close.notna().any().sum()),
        "top_n": args.top_n,
        "rebalance": args.rebalance,
        "capital": args.capital,
        "factor_set": args.factor_set,
        "orientation": args.orientation,
        "n_rebalance_dates": int(weights["date"].nunique()) if len(weights) else 0,
        "target_file": str(OUT_DIR / "cn_target_weights.csv"),
        "self_backtest_file": str(OUT_DIR / "cn_self_backtest.csv"),
    }
    pd.Series(summary).to_json(OUT_DIR / "cn_bridge_meta.json", force_ascii=False, indent=2)

    print("Exported RQAlpha bridge targets")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
