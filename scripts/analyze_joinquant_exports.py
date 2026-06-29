#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze JoinQuant exported transaction/position/log files.

The script intentionally depends only on the exported CSV/log files, so it can
be reused for v6, v7 and later strategy versions without touching JoinQuant.

Example:
    PYTHONPATH=. python scripts/analyze_joinquant_exports.py jointquant/v6 v6
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPITAL = 60_000.0


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_transaction(jq_dir: Path) -> pd.DataFrame:
    trx = read_csv(jq_dir / "transaction.csv")
    trx["date"] = pd.to_datetime(trx["日期"])
    trx["code"] = trx["标的"].astype(str).str.extract(r"\((\d+\.XS(?:HG|HE))\)")[0]
    trx["name"] = trx["标的"].astype(str).str.replace(r"\(\d+\.XS(?:HG|HE)\)", "", regex=True)
    trx["qty"] = trx["成交数量"].astype(str).str.replace("股", "", regex=False).astype(float)
    price_col = "成交均价" if "成交均价" in trx.columns else "成交价"
    trx["price"] = pd.to_numeric(trx[price_col], errors="coerce")
    trx["amount"] = pd.to_numeric(trx["成交额"], errors="coerce")
    trx["fee"] = pd.to_numeric(trx["手续费"], errors="coerce").fillna(0.0)
    trx["pnl"] = pd.to_numeric(trx["平仓盈亏"], errors="coerce").fillna(0.0)
    trx["side"] = trx["交易类型"].astype(str)
    return trx


def parse_position(jq_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = read_csv(jq_dir / "position.csv").reset_index()
    pos.columns = [
        "date",
        "asset_type",
        "target",
        "side",
        "qty",
        "available",
        "close",
        "value",
        "float_pnl",
        "open_avg",
        "fut_avg",
        "margin",
        "day_pnl",
        "today_qty",
        "pnl_pct",
        "total_value",
        "weight_pct",
    ]
    pos["date"] = pd.to_datetime(pos["date"])
    for col in ["close", "value", "float_pnl", "day_pnl", "total_value"]:
        pos[col] = pd.to_numeric(pos[col], errors="coerce")
    pos["code"] = pos["target"].astype(str).str.extract(r"\((\d+\.XS(?:HG|HE))\)")[0]
    pos["name"] = pos["target"].astype(str).str.replace(r"\(\d+\.XS(?:HG|HE)\)", "", regex=True)
    stock = pos[pos["asset_type"].eq("股票")].copy()
    cash = pos[pos["target"].eq("Cash")].copy()
    return stock, cash


def build_daily(stock: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    stock_daily = stock.groupby("date").agg(
        stock_value=("value", "sum"),
        holdings=("code", "nunique"),
        day_pnl=("day_pnl", "sum"),
        max_weight=(
            "weight_pct",
            lambda s: pd.to_numeric(
                s.astype(str).str.replace("%", "", regex=False),
                errors="coerce",
            ).max(),
        ),
    )
    total_daily = stock.groupby("date")["total_value"].max().rename("total_value")
    cash_daily = cash.groupby("date")["close"].last().rename("cash")
    daily = pd.concat([stock_daily, total_daily, cash_daily], axis=1).sort_index()
    daily["cash"] = daily["cash"].fillna(daily["total_value"] - daily["stock_value"])
    daily["exposure"] = daily["stock_value"] / daily["total_value"]
    daily["cash_ratio"] = daily["cash"] / daily["total_value"]
    daily["ret"] = daily["total_value"].pct_change().fillna(0.0)
    daily["equity"] = daily["total_value"] / CAPITAL
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    return daily


def max_drawdown_span(daily: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    dd_end = daily["drawdown"].idxmin()
    dd_start = daily.loc[:dd_end, "equity"].idxmax()
    return dd_start, dd_end, float(-daily.loc[dd_end, "drawdown"])


def parse_rebalance_log(jq_dir: Path, version: str) -> pd.DataFrame:
    lines = (jq_dir / "log.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    factor_rows = []
    for line in lines:
        escaped_version = re.escape(version)
        m = re.search(
            rf"(\d{{4}}-\d{{2}}-\d{{2}}) .*{escaped_version} rebalance .*?factors=([^ ]+) "
            rf"candidates=(\d+) scored=(\d+).*?targets=([0-9A-Z.,]+)",
            line,
        )
        if m:
            targets = m.group(5).split(",")
            rows.append(
                {
                    "date": pd.to_datetime(m.group(1)),
                    "factors": m.group(2),
                    "candidate_count": int(m.group(3)),
                    "score_count": int(m.group(4)),
                    "targets": ",".join(targets),
                    "target_count": len(targets),
                }
            )
        fm = re.search(
            rf"(\d{{4}}-\d{{2}}-\d{{2}}) .*{escaped_version} factor columns before scoring: (.*)$",
            line,
        )
        if fm:
            factor_rows.append({"date": pd.to_datetime(fm.group(1)), "factor_columns": fm.group(2)})
    reb = pd.DataFrame(rows)
    if not reb.empty and factor_rows:
        fac = pd.DataFrame(factor_rows).drop_duplicates("date")
        reb = reb.merge(fac, on="date", how="left")
    return reb


def summarize_by_code(trx: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    sell = trx[trx["side"].eq("卖")].copy()
    by = sell.groupby("code").agg(
        name=("name", "last"),
        sell_trades=("code", "count"),
        realized_pnl=("pnl", "sum"),
        sell_amount=("amount", lambda s: -s.sum()),
        fee=("fee", "sum"),
    )
    by["realized_pnl_rate"] = by["realized_pnl"] / by["sell_amount"].replace(0, np.nan)
    held_days = stock.groupby("code").agg(
        held_days=("date", "nunique"),
        avg_value=("value", "mean"),
        last_float_pnl=("float_pnl", "last"),
    )
    by = by.join(held_days, how="outer")
    by["name"] = by["name"].fillna(stock.groupby("code")["name"].last())
    by = by.sort_values("realized_pnl")
    return by


def annualized_return(equity: pd.Series) -> float:
    years = max(len(equity) / 252.0, 1e-9)
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: analyze_joinquant_exports.py <jointquant/version-dir> <version>")
    jq_dir = (PROJECT_ROOT / sys.argv[1]).resolve() if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1])
    version = sys.argv[2]
    if not jq_dir.exists():
        raise FileNotFoundError(jq_dir)

    out_md = jq_dir / f"{version}_deep_analysis.md"
    out_daily = jq_dir / f"{version}_daily_metrics.csv"
    out_code = jq_dir / f"{version}_code_contribution.csv"
    out_rebalance = jq_dir / f"{version}_rebalance_targets.csv"

    trx = parse_transaction(jq_dir)
    stock, cash = parse_position(jq_dir)
    daily = build_daily(stock, cash)
    reb = parse_rebalance_log(jq_dir, version)
    by_code = summarize_by_code(trx, stock)

    trade_by_date = trx.groupby("date").agg(
        orders=("code", "count"),
        buy_amount=("amount", lambda s: s[s > 0].sum()),
        sell_amount=("amount", lambda s: -s[s < 0].sum()),
        fee=("fee", "sum"),
        realized_pnl=("pnl", "sum"),
    )
    trade_by_date["gross_turnover"] = (
        trade_by_date["buy_amount"] + trade_by_date["sell_amount"]
    ) / daily["total_value"].reindex(trade_by_date.index)

    dd_start, dd_end, max_dd = max_drawdown_span(daily)
    total_return = float(daily["equity"].iloc[-1] - 1.0)
    ann = annualized_return(daily["equity"])
    sharpe = float(np.sqrt(252) * daily["ret"].mean() / daily["ret"].std())
    sell = trx[trx["side"].eq("卖")]

    out_daily.write_text(daily.to_csv(), encoding="utf-8")
    by_code.to_csv(out_code, encoding="utf-8")
    reb.to_csv(out_rebalance, index=False, encoding="utf-8")

    worst = by_code.sort_values("realized_pnl").head(12)
    best = by_code.sort_values("realized_pnl", ascending=False).head(12)
    period = daily.assign(period=daily.index.to_period("Y")).groupby("period").agg(
        start_value=("total_value", "first"),
        end_value=("total_value", "last"),
        min_drawdown=("drawdown", "min"),
        avg_exposure=("exposure", "mean"),
        avg_cash=("cash_ratio", "mean"),
        avg_holdings=("holdings", "mean"),
        avg_max_weight=("max_weight", "mean"),
    )
    period["return"] = period["end_value"] / period["start_value"] - 1.0

    lines = [
        f"# JoinQuant {version} Deep Analysis",
        "",
        "## Summary",
        "",
        f"- Range: {daily.index.min().date()} ~ {daily.index.max().date()}",
        f"- Total return: {total_return:.2%}",
        f"- Annualized return: {ann:.2%}",
        f"- Raw daily Sharpe: {sharpe:.2f}",
        f"- Max drawdown: {max_dd:.2%}, {dd_start.date()} ~ {dd_end.date()}",
        f"- Average exposure: {daily['exposure'].mean():.2%}",
        f"- Average cash: {daily['cash_ratio'].mean():.2%}",
        f"- Average holdings: {daily['holdings'].mean():.2f}",
        f"- Average max single-stock weight: {daily['max_weight'].mean():.2f}%",
        f"- Trade dates: {len(trade_by_date)}",
        f"- Orders: {len(trx)}",
        f"- Total fee: {trx['fee'].sum():.2f}",
        f"- Sell win rate: {(sell['pnl'] > 0).mean():.2%}",
        "",
        "## Rebalance Diagnostics",
        "",
        f"- Rebalance count: {len(reb)}",
        f"- Candidate count range: {reb['candidate_count'].min() if not reb.empty else np.nan} ~ {reb['candidate_count'].max() if not reb.empty else np.nan}",
        f"- Score count range: {reb['score_count'].min() if not reb.empty else np.nan} ~ {reb['score_count'].max() if not reb.empty else np.nan}",
        f"- Factor columns: {reb['factor_columns'].dropna().iloc[-1] if not reb.empty and reb['factor_columns'].notna().any() else 'N/A'}",
        "",
        "## Yearly Path",
        "",
        period.to_markdown(),
        "",
        "## Worst Realized PnL By Code",
        "",
        worst[["name", "sell_trades", "realized_pnl", "realized_pnl_rate", "held_days", "last_float_pnl"]].to_markdown(),
        "",
        "## Best Realized PnL By Code",
        "",
        best[["name", "sell_trades", "realized_pnl", "realized_pnl_rate", "held_days", "last_float_pnl"]].to_markdown(),
        "",
        "## Outputs",
        "",
        f"- {out_daily.name}",
        f"- {out_code.name}",
        f"- {out_rebalance.name}",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {out_md}")
    print(f"[ok] wrote {out_daily}")
    print(f"[ok] wrote {out_code}")
    print(f"[ok] wrote {out_rebalance}")
    print("\n".join(lines[:36]))


if __name__ == "__main__":
    main()
