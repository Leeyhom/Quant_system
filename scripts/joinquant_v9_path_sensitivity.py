#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose v9 path sensitivity: warm 2025 slice vs cold 2025 starts.

The user observed that JoinQuant v9 looks best on 2019-2025, but can lose money
when the backtest starts on 2025-01-01. This is exactly the kind of problem a
stateful strategy can have: carried positions, rebalance phase, and the holding
buffer change the portfolio even when the factor formula is unchanged.

This local test uses the cached JoinQuant pool subset. It is a mechanism check,
not a replacement for the real JoinQuant export.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.backtest.cn_cost import cn_trade_cost_yuan  # noqa: E402
from quant.backtest.metrics import summary  # noqa: E402
from quant.config import RAW_DATA_DIR  # noqa: E402
from quant.data.industry import industry_series  # noqa: E402
from quant.data.panel import build_ohlcv_panels, build_value_panels  # noqa: E402
from scripts.joinquant_v6_validation import BACKTEST_END, BACKTEST_START, CAPITAL, FACTOR_START  # noqa: E402
from scripts.joinquant_v9_validation import (  # noqa: E402
    build_market_equity,
    build_v9_factor,
    exposure_for,
    load_jq_strategy,
    score_weights,
    select_targets,
)


OUT_DIR = PROJECT_ROOT / "jointquant" / "v9"
OUT_CSV = OUT_DIR / "v9_2025_path_sensitivity.csv"
OUT_REB = OUT_DIR / "v9_2025_path_sensitivity_rebalances.csv"
OUT_MD = OUT_DIR / "v9_2025_path_sensitivity.md"


def metric_from_daily(daily: pd.DataFrame) -> dict:
    equity = daily["total_value"] / daily["total_value"].iloc[0]
    ret = equity.pct_change(fill_method=None).fillna(0.0)
    out = summary(equity, ret)
    out["start_value"] = float(daily["total_value"].iloc[0])
    out["end_value"] = float(daily["total_value"].iloc[-1])
    out["avg_exposure"] = float(daily["exposure"].mean())
    out["avg_cash"] = float(daily["cash_ratio"].mean())
    out["avg_holdings"] = float(daily["holdings"].mean())
    return out


def trace_backtest(
    close_full: pd.DataFrame,
    factor_full: pd.DataFrame,
    market_equity: pd.Series,
    *,
    start: str,
    end: str,
    top_n: int = 10,
    rebalance_every: int = 60,
    industry_cap: int = 2,
    hold_multiplier: float = 1.5,
    exposure_mode: str = "fixed98",
    weight_mode: str = "equal",
    exposure_ramp: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_full = close_full.sort_index()
    factor_full = factor_full.reindex_like(close_full)
    close = close_full.loc[start:end]
    full_dates = close_full.index
    industry = industry_series(list(close.columns)).to_dict()
    vol60 = close_full.pct_change(fill_method=None).rolling(60).std()

    cash = CAPITAL
    shares = pd.Series(0.0, index=close.columns)
    prev_value = CAPITAL
    rebalance_count = 0
    daily_rows: list[dict] = []
    rebalance_rows: list[dict] = []

    for i, date in enumerate(close.index):
        prices = close.loc[date]
        stock_value = float((shares * prices.fillna(0.0)).sum())
        total_value = cash + stock_value

        if i % rebalance_every == 0:
            date_idx = full_dates.get_loc(date)
            prev_idx = max(date_idx - 1, 0)
            base_exposure = exposure_for(exposure_mode, market_equity, prev_idx)
            if exposure_ramp:
                exposure = min(base_exposure, exposure_ramp[min(rebalance_count, len(exposure_ramp) - 1)])
            else:
                exposure = base_exposure

            scores = factor_full.iloc[prev_idx].dropna().sort_values(ascending=False)
            scores = scores[scores.index.isin(prices.dropna().index)]
            current = shares[shares > 0].index.tolist()
            selected = select_targets(
                scores,
                prices,
                industry,
                top_n=top_n,
                industry_cap=industry_cap,
                slot_value=total_value * exposure / top_n,
                close_full=close_full,
                date_idx=prev_idx,
                current=current,
                hold_multiplier=hold_multiplier,
            )
            weights = score_weights(selected, vol60.iloc[prev_idx], exposure=exposure, mode=weight_mode)
            target_shares = pd.Series(0.0, index=close.columns)
            for stock, weight in weights.items():
                if stock not in prices.index or not np.isfinite(prices[stock]) or prices[stock] <= 0:
                    continue
                target_value = total_value * weight
                target_shares[stock] = math.floor(target_value / prices[stock] / 100.0) * 100.0

            delta = target_shares - shares
            notional = (delta.abs() * prices).fillna(0.0)
            fee = cn_trade_cost_yuan(notional, slippage=0.0005)
            cash = total_value - float((target_shares * prices.fillna(0.0)).sum()) - fee
            shares = target_shares
            stock_value = float((shares * prices.fillna(0.0)).sum())
            total_value = cash + stock_value
            rebalance_rows.append(
                {
                    "date": date,
                    "rebalance_count": rebalance_count + 1,
                    "target_exposure": exposure,
                    "actual_exposure": stock_value / total_value if total_value > 0 else 0.0,
                    "turnover": float(notional.sum() / max(total_value, 1.0)),
                    "fee": float(fee),
                    "holdings": int((shares > 0).sum()),
                    "targets": ",".join(selected),
                }
            )
            rebalance_count += 1

        stock_value = float((shares * prices.fillna(0.0)).sum())
        total_value = cash + stock_value
        daily_rows.append(
            {
                "date": date,
                "total_value": total_value,
                "ret": total_value / prev_value - 1.0 if daily_rows else 0.0,
                "stock_value": stock_value,
                "cash": cash,
                "exposure": stock_value / total_value if total_value > 0 else 0.0,
                "cash_ratio": cash / total_value if total_value > 0 else 0.0,
                "holdings": int((shares > 0).sum()),
            }
        )
        prev_value = total_value

    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["drawdown"] = daily["total_value"] / daily["total_value"].cummax() - 1.0
    rebalances = pd.DataFrame(rebalance_rows)
    return daily, rebalances


def slice_with_anchor(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    inside = daily.loc[start:end].copy()
    anchor = daily.loc[daily.index < start_ts].tail(1).copy()
    if anchor.empty:
        return inside
    return pd.concat([anchor, inside])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jq = load_jq_strategy()
    pool_all = [s.split(".")[0] for s in jq.STOCK_POOL]
    symbols = [
        s for s in pool_all
        if (RAW_DATA_DIR / f"{s}.parquet").exists()
        and (RAW_DATA_DIR / f"{s}_value.parquet").exists()
    ]
    if not symbols:
        raise RuntimeError("No cached JoinQuant pool data available.")

    panels = build_ohlcv_panels(symbols, start=FACTOR_START, end=BACKTEST_END)
    close, amount = panels["close"], panels["amount"]
    value = build_value_panels(symbols, start=FACTOR_START, end=BACKTEST_END, align_to=close)
    industry = industry_series(list(close.columns))
    factor = build_v9_factor(close, amount, value, industry)
    market_equity = build_market_equity(close)

    full_v9, full_v9_reb = trace_backtest(
        close,
        factor,
        market_equity,
        start=BACKTEST_START,
        end=BACKTEST_END,
        exposure_mode="fixed98",
    )
    full_v10_ramp, full_v10_ramp_reb = trace_backtest(
        close,
        factor,
        market_equity,
        start=BACKTEST_START,
        end=BACKTEST_END,
        exposure_mode="fixed98",
        exposure_ramp=(0.80, 0.90, 0.98),
    )
    cold_v9, cold_v9_reb = trace_backtest(
        close,
        factor,
        market_equity,
        start="20250101",
        end=BACKTEST_END,
        exposure_mode="fixed98",
    )
    cold_fixed95, cold_fixed95_reb = trace_backtest(
        close,
        factor,
        market_equity,
        start="20250101",
        end=BACKTEST_END,
        exposure_mode="fixed95",
    )
    cold_ramp_809098, cold_ramp_809098_reb = trace_backtest(
        close,
        factor,
        market_equity,
        start="20250101",
        end=BACKTEST_END,
        exposure_mode="fixed98",
        exposure_ramp=(0.80, 0.90, 0.98),
    )
    cold_ramp_708598, cold_ramp_708598_reb = trace_backtest(
        close,
        factor,
        market_equity,
        start="20250101",
        end=BACKTEST_END,
        exposure_mode="fixed98",
        exposure_ramp=(0.70, 0.85, 0.98),
    )

    cases = {
        "warm_full_v9_total": full_v9,
        "warm_full_v9_slice_2025": slice_with_anchor(full_v9, "20250101", BACKTEST_END),
        "warm_full_v10_ramp_total": full_v10_ramp,
        "warm_full_v10_ramp_slice_2025": slice_with_anchor(full_v10_ramp, "20250101", BACKTEST_END),
        "cold_2025_v9_immediate": cold_v9,
        "cold_2025_v9_fixed95": cold_fixed95,
        "cold_2025_v10_ramp_80_90_98": cold_ramp_809098,
        "cold_2025_v10_ramp_70_85_98": cold_ramp_708598,
    }
    rows = []
    for name, daily in cases.items():
        rows.append({"case": name, **metric_from_daily(daily)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_CSV, index=False)

    rebalance_frames = []
    for case, reb in [
        ("warm_full_v9", full_v9_reb),
        ("warm_full_v10_ramp", full_v10_ramp_reb),
        ("cold_2025_v9_immediate", cold_v9_reb),
        ("cold_2025_v9_fixed95", cold_fixed95_reb),
        ("cold_2025_v10_ramp_80_90_98", cold_ramp_809098_reb),
        ("cold_2025_v10_ramp_70_85_98", cold_ramp_708598_reb),
    ]:
        part = reb.copy()
        part["case"] = case
        rebalance_frames.append(part)
    rebalances = pd.concat(rebalance_frames, ignore_index=True)
    rebalances.to_csv(OUT_REB, index=False)

    warm_2025_reb = full_v9_reb[full_v9_reb["date"].between("2025-01-01", "2025-12-31")]
    cold_first_targets = cold_v9_reb.iloc[0]["targets"].split(",") if not cold_v9_reb.empty else []
    warm_first_2025_targets = warm_2025_reb.iloc[0]["targets"].split(",") if not warm_2025_reb.empty else []
    overlap = sorted(set(cold_first_targets).intersection(warm_first_2025_targets))

    show_cols = [
        "case",
        "total_return",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "avg_exposure",
        "avg_cash",
        "avg_holdings",
        "start_value",
        "end_value",
    ]
    lines = [
        "# v9 2025 Path Sensitivity",
        "",
        f"- Cached JoinQuant pool coverage: {len(symbols)}/{len(pool_all)}",
        "- Purpose: mechanism check for warm-path vs cold-start differences.",
        "- Caveat: local cached pool may differ from the real JoinQuant 152-stock run.",
        "",
        "## Metrics",
        "",
        metrics[show_cols].to_markdown(index=False),
        "",
        "## Rebalance Phase",
        "",
        f"- Warm full-run v9 rebalances inside 2025: {', '.join(warm_2025_reb['date'].dt.strftime('%Y-%m-%d'))}",
        f"- Cold 2025 v9 rebalances: {', '.join(cold_v9_reb['date'].dt.strftime('%Y-%m-%d'))}",
        f"- First cold-start targets overlap with first warm-2025 rebalance targets: {len(overlap)}/{len(cold_first_targets)}",
        f"- Overlap: {', '.join(overlap) if overlap else 'none'}",
        "",
        "## Cold-Start Interpretation",
        "",
        "The same factor can produce a different 2025 result because v9 is stateful:",
        "carried positions, the 60-trading-day rebalance phase, and the 1.5x holding buffer all depend on when the strategy starts.",
        "The ramp variants are not new alpha. They are live-start risk controls that reduce the capital committed before the first few rebalances have confirmed the path.",
        "",
        "## Outputs",
        "",
        f"- {OUT_CSV.name}",
        f"- {OUT_REB.name}",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_REB}")
    print(f"[ok] wrote {OUT_MD}")
    print(metrics[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
