# -*- coding: utf-8 -*-
"""Build JoinQuant strategy-version diagnostics for the HTML dashboard.

The dashboard needs more than return/Sharpe. This script calculates or records:
alpha, beta, information ratio, win rate, profit/loss ratio, profit factor,
volatility, drawdown and benchmark-relative metrics for the A-share JoinQuant
strategy versions.

Two data tiers are intentionally separated:

1. local_* rows: fully recalculated from local akshare-style data. Benchmark is
   the local equal-weight stock pool because a CSI300 daily benchmark is not
   cached in this project.
2. jq_* rows: JoinQuant reported/exported rows. For v3/v5 only screenshots are
   available, so their JoinQuant alpha/beta are recorded from screenshots and
   marked as such. Once the user exports daily/transaction files, these rows can
   be replaced with fully calculated rows.
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
from quant.data.industry import industry_series
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.universe import DEFAULT_POOL
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize


OUT_DIR = PROJECT_ROOT / "jointquant" / "version_metrics"
OUT_CSV = OUT_DIR / "joinquant_strategy_metrics.csv"
OUT_MD = OUT_DIR / "joinquant_strategy_metrics.md"

FACTOR_START = "20180101"
BACKTEST_START = "20190101"
BACKTEST_END = "20251231"
CAPITAL = 60_000.0
MAX_EXPOSURE = 0.95
TRADING_DAYS = 252


def _max_drawdown(equity: pd.Series) -> float:
    return float((1.0 - equity / equity.cummax()).max())


def _annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / len(equity)) - 1.0)


def _sharpe(ret: pd.Series) -> float:
    std = ret.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(ret.mean() / std * np.sqrt(TRADING_DAYS))


def _sortino(ret: pd.Series) -> float:
    downside = ret[ret < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(ret.mean() / downside * np.sqrt(TRADING_DAYS))


def _beta(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([strategy_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    bvar = aligned.iloc[:, 1].var()
    if bvar == 0 or np.isnan(bvar):
        return np.nan
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / bvar)


def _metric_row(
    version: str,
    source: str,
    equity: pd.Series,
    strategy_ret: pd.Series,
    bench_ret: pd.Series,
    trade_pnl: pd.Series | None = None,
    note: str = "",
) -> dict:
    equity = equity.dropna()
    strategy_ret = strategy_ret.reindex(equity.index).fillna(0.0)
    bench_ret = bench_ret.reindex(equity.index).fillna(0.0)
    bench_equity = (1.0 + bench_ret).cumprod()

    active = strategy_ret - bench_ret
    beta = _beta(strategy_ret, bench_ret)
    alpha_ann = float((strategy_ret.mean() - beta * bench_ret.mean()) * TRADING_DAYS) if np.isfinite(beta) else np.nan
    tracking_error = float(active.std() * np.sqrt(TRADING_DAYS))
    info_ratio = float(active.mean() / active.std() * np.sqrt(TRADING_DAYS)) if active.std() > 0 else np.nan

    win = strategy_ret[strategy_ret > 0]
    loss = strategy_ret[strategy_ret < 0]
    daily_pl = float(win.mean() / abs(loss.mean())) if len(win) and len(loss) else np.nan
    daily_win_rate = float((strategy_ret > 0).mean())

    row = {
        "version": version,
        "source": source,
        "metric_basis": "daily_calculated",
        "start": equity.index.min().date().isoformat(),
        "end": equity.index.max().date().isoformat(),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "annualized_return": _annualized_return(equity),
        "benchmark_return": float(bench_equity.iloc[-1] / bench_equity.iloc[0] - 1.0),
        "benchmark_annualized": _annualized_return(bench_equity),
        "excess_return": float((equity.iloc[-1] / equity.iloc[0] - 1.0) - (bench_equity.iloc[-1] / bench_equity.iloc[0] - 1.0)),
        "alpha_ann": alpha_ann,
        "beta": beta,
        "sharpe": _sharpe(strategy_ret),
        "benchmark_sharpe": _sharpe(bench_ret),
        "sortino": _sortino(strategy_ret),
        "information_ratio": info_ratio,
        "tracking_error": tracking_error,
        "volatility": float(strategy_ret.std() * np.sqrt(TRADING_DAYS)),
        "max_drawdown": _max_drawdown(equity),
        "calmar": float(_annualized_return(equity) / _max_drawdown(equity)) if _max_drawdown(equity) > 0 else np.nan,
        "daily_win_rate": daily_win_rate,
        "daily_pl_ratio": daily_pl,
        "trade_win_rate": np.nan,
        "trade_pl_ratio": np.nan,
        "profit_factor": np.nan,
        "note": note,
    }

    if trade_pnl is not None and len(trade_pnl) > 0:
        pos = trade_pnl[trade_pnl > 0]
        neg = trade_pnl[trade_pnl < 0]
        row["trade_win_rate"] = float((trade_pnl > 0).mean())
        row["trade_pl_ratio"] = float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else np.nan
        row["profit_factor"] = float(pos.sum() / abs(neg.sum())) if len(pos) and len(neg) else np.nan
        row["metric_basis"] = "daily_plus_trade_export"
    return row


def _benchmark_returns(close: pd.DataFrame, start: str, end: str) -> pd.Series:
    sub = close.loc[start:end]
    ret = sub.pct_change().fillna(0.0)
    weights = sub.notna().astype(float).div(sub.notna().sum(axis=1), axis=0).fillna(0.0)
    return (weights * ret).sum(axis=1)


def _build_factor_library(close: pd.DataFrame, amount: pd.DataFrame, value: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ind = industry_series(list(close.columns))
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "quality_roe": F.quality_roe(value["pe_ttm"], value["pb"]),
    }
    return {name: neutralize(factor, industry=ind, log_mv=log_mv, mode="full") for name, factor in raw.items()}


def _weighted_composite(lib: dict[str, pd.DataFrame], names: list[str], weights: list[float] | None = None) -> pd.DataFrame:
    if weights is None:
        return combine_factors(*(lib[n] for n in names))
    ranked = [lib[n].rank(axis=1, pct=True) * w for n, w in zip(names, weights)]
    return sum(ranked) / sum(weights)


def _passes_v5_filter(stock: str, full_idx: int, close_full: pd.DataFrame) -> bool:
    prices = close_full[stock].iloc[: full_idx + 1].dropna()
    if len(prices) < 130:
        return True
    mom120 = prices.iloc[-1] / prices.iloc[-121] - 1.0
    return bool(mom120 > -0.10)


def _select_targets(
    scores: pd.Series,
    prices: pd.Series,
    industry: dict[str, str],
    top_n: int,
    industry_cap: int,
    slot_value: float,
    close_full: pd.DataFrame,
    full_idx: int,
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
        if filter_mode == "v5_mom120" and not _passes_v5_filter(stock, full_idx, close_full):
            continue
        ind = industry.get(stock, "其他")
        if industry_cap > 0 and counts.get(ind, 0) >= industry_cap:
            continue
        selected.append(stock)
        counts[ind] = counts.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def _local_lot_series(
    close_full: pd.DataFrame,
    factor_full: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
    industry_cap: int,
    filter_mode: str = "none",
) -> tuple[pd.Series, pd.Series]:
    close_full = close_full.sort_index()
    factor_full = factor_full.reindex_like(close_full)
    close = close_full.loc[BACKTEST_START:BACKTEST_END]
    industry = industry_series(list(close.columns)).to_dict()
    full_dates = close_full.index

    cash = CAPITAL
    shares = pd.Series(0.0, index=close.columns)
    values = []
    returns = []
    prev_value = CAPITAL

    for i, date in enumerate(close.index):
        prices = close.loc[date]
        total_value = cash + float((shares * prices.fillna(0.0)).sum())
        if i % rebalance_every == 0:
            full_idx = full_dates.get_loc(date)
            scores = factor_full.iloc[full_idx - 1].dropna().sort_values(ascending=False)
            scores = scores[scores.index.isin(prices.dropna().index)]
            slot = total_value * MAX_EXPOSURE / top_n
            selected = _select_targets(scores, prices, industry, top_n, industry_cap, slot, close_full, full_idx - 1, filter_mode)

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

        values.append(total_value)
        returns.append(total_value / prev_value - 1.0 if len(values) > 1 else 0.0)
        prev_value = total_value

    equity = pd.Series(values, index=close.index) / CAPITAL
    ret = pd.Series(returns, index=close.index)
    return equity, ret


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _jq_export_row(version: str, folder: Path, bench_ret_full: pd.Series) -> dict | None:
    pos_file = folder / "position.csv"
    trx_file = folder / "transaction.csv"
    if not pos_file.exists() or not trx_file.exists():
        return None

    pos = _read_csv(pos_file).reset_index()
    pos.columns = [
        "date", "asset_type", "target", "side", "qty", "available", "close", "value",
        "float_pnl", "open_avg", "fut_avg", "margin", "day_pnl", "today_qty",
        "pnl_pct", "total_value", "weight_pct",
    ]
    pos["date"] = pd.to_datetime(pos["date"])
    for col in ["value", "total_value"]:
        pos[col] = pd.to_numeric(pos[col], errors="coerce")
    stock = pos[pos["asset_type"].eq("股票")].copy()
    daily_value = stock.groupby("date")["total_value"].max().dropna()
    equity = daily_value / CAPITAL
    ret = daily_value.pct_change().fillna(0.0)

    trx = _read_csv(trx_file)
    trx["pnl"] = pd.to_numeric(trx["平仓盈亏"], errors="coerce").fillna(0.0)
    trade_pnl = trx.loc[trx["交易类型"].astype(str).eq("卖"), "pnl"]
    bench = bench_ret_full.reindex(ret.index).fillna(0.0)
    return _metric_row(version, "jq_export_local_pool_benchmark", equity, ret, bench, trade_pnl=trade_pnl, note="聚宽导出日净值；Alpha/Beta 用本地等权池近似基准重算")


def _jq_screenshot_rows() -> list[dict]:
    rows = [
        {
            "version": "v2.1", "source": "jq_screenshot", "metric_basis": "joinquant_reported",
            "start": "2018-01-01", "end": "2025-12-31", "total_return": 0.8828,
            "annualized_return": 0.0849, "benchmark_return": 0.1486, "benchmark_annualized": np.nan,
            "excess_return": 0.6391, "alpha_ann": 0.061, "beta": 0.744, "sharpe": 0.269,
            "benchmark_sharpe": np.nan, "sortino": 0.386, "information_ratio": 0.685,
            "tracking_error": np.nan, "volatility": 0.166, "max_drawdown": 0.3270,
            "calmar": np.nan, "daily_win_rate": 0.519, "daily_pl_ratio": 2.218,
            "trade_win_rate": np.nan, "trade_pl_ratio": np.nan, "profit_factor": np.nan,
            "note": "聚宽截图录入；非本地重算",
        },
        {
            "version": "v3", "source": "jq_screenshot", "metric_basis": "joinquant_reported",
            "start": "2019-01-01", "end": "2025-12-31", "total_return": 1.7764,
            "annualized_return": 0.1621, "benchmark_return": 0.5379, "benchmark_annualized": np.nan,
            "excess_return": 0.8054, "alpha_ann": 0.103, "beta": 0.737, "sharpe": 0.727,
            "benchmark_sharpe": np.nan, "sortino": 1.029, "information_ratio": 0.919,
            "tracking_error": np.nan, "volatility": 0.168, "max_drawdown": 0.2122,
            "calmar": np.nan, "daily_win_rate": 0.518, "daily_pl_ratio": 2.708,
            "trade_win_rate": np.nan, "trade_pl_ratio": np.nan, "profit_factor": np.nan,
            "note": "聚宽截图录入；当前聚宽实测最优",
        },
        {
            "version": "v4", "source": "jq_screenshot", "metric_basis": "joinquant_reported",
            "start": "2019-01-01", "end": "2025-12-31", "total_return": 1.3809,
            "annualized_return": 0.1361, "benchmark_return": 0.5379, "benchmark_annualized": np.nan,
            "excess_return": 0.5482, "alpha_ann": 0.077, "beta": 0.751, "sharpe": 0.516,
            "benchmark_sharpe": np.nan, "sortino": 0.762, "information_ratio": 0.551,
            "tracking_error": np.nan, "volatility": 0.186, "max_drawdown": 0.2611,
            "calmar": np.nan, "daily_win_rate": 0.493, "daily_pl_ratio": 2.144,
            "trade_win_rate": np.nan, "trade_pl_ratio": np.nan, "profit_factor": np.nan,
            "note": "聚宽截图录入；v4 导出另有可重算行",
        },
        {
            "version": "v5", "source": "jq_screenshot", "metric_basis": "joinquant_reported",
            "start": "2019-01-01", "end": "2025-12-31", "total_return": 1.3990,
            "annualized_return": 0.1374, "benchmark_return": 0.5379, "benchmark_annualized": np.nan,
            "excess_return": 0.5599, "alpha_ann": 0.079, "beta": 0.726, "sharpe": 0.591,
            "benchmark_sharpe": np.nan, "sortino": 0.844, "information_ratio": 0.693,
            "tracking_error": np.nan, "volatility": 0.165, "max_drawdown": 0.1653,
            "calmar": np.nan, "daily_win_rate": 0.523, "daily_pl_ratio": 2.754,
            "trade_win_rate": np.nan, "trade_pl_ratio": np.nan, "profit_factor": np.nan,
            "note": "聚宽截图录入；降低回撤但收益未超过 v3",
        },
    ]
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = build_ohlcv_panels(DEFAULT_POOL, start=FACTOR_START, end=BACKTEST_END)
    close = panels["close"]
    amount = panels["amount"]
    value = build_value_panels(DEFAULT_POOL, start=FACTOR_START, end=BACKTEST_END, align_to=close)
    bench_ret = _benchmark_returns(close, BACKTEST_START, BACKTEST_END)
    bench_ret_full = _benchmark_returns(close, "20180101", BACKTEST_END)
    lib = _build_factor_library(close, amount, value)

    v3_factor = _weighted_composite(lib, ["earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud"])
    v4_factor = v3_factor
    v5_factor = _weighted_composite(
        lib,
        ["earnings_yield", "cashflow_yield", "sales_yield", "growth_peg", "amihud", "quality_roe"],
        [1, 1, 1, 1, 1, 0.5],
    )

    rows = []
    for version, factor, top_n, rebalance, industry_cap, filt, note in [
        ("v3", v3_factor, 10, 40, 1, "none", "本地复算；v3 聚宽策略形态"),
        ("v4", v4_factor, 6, 30, 2, "none", "本地复算；v4 集中化版本"),
        ("v5", v5_factor, 10, 60, 1, "v5_mom120", "本地复算；质量半权重+动量过滤"),
    ]:
        equity, ret = _local_lot_series(close, factor, top_n, rebalance, industry_cap, filter_mode=filt)
        rows.append(_metric_row(version, "local_recalc_equal_weight_benchmark", equity, ret, bench_ret, note=note))

    for version, folder in [
        ("v2.1", PROJECT_ROOT / "jointquant" / "v2"),
        ("v4", PROJECT_ROOT / "jointquant" / "v4"),
    ]:
        row = _jq_export_row(version, folder, bench_ret_full)
        if row:
            rows.append(row)

    rows.extend(_jq_screenshot_rows())

    df = pd.DataFrame(rows)
    order = {
        "jq_screenshot": 0,
        "jq_export_local_pool_benchmark": 1,
        "local_recalc_equal_weight_benchmark": 2,
    }
    df["_source_order"] = df["source"].map(order).fillna(99)
    df = df.sort_values(["_source_order", "version"]).drop(columns=["_source_order"])
    df.to_csv(OUT_CSV, index=False)

    show_cols = [
        "version", "source", "total_return", "annualized_return", "benchmark_return",
        "excess_return", "alpha_ann", "beta", "sharpe", "information_ratio",
        "max_drawdown", "daily_win_rate", "daily_pl_ratio", "trade_win_rate",
        "trade_pl_ratio", "profit_factor", "note",
    ]
    lines = [
        "# JoinQuant Strategy Version Metrics",
        "",
        "Alpha/Beta rows from `jq_screenshot` use JoinQuant's reported benchmark metrics.",
        "`local_recalc_equal_weight_benchmark` rows use the local equal-weight stock-pool benchmark.",
        "",
        df[show_cols].to_markdown(index=False),
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_MD}")
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
