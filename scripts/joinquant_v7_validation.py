#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""joinquant_v7_validation —— v7 聚宽策略候选本地验证。

v7 基于聚宽 v6 真实日志做两处改动：
  1. 明确关闭 holder 因子。v6 在聚宽里 holder_coverage=0，真实执行并未用到筹码因子；
  2. 尝试“分数倾斜持仓 + 市场状态仓位”。前者提高高分股票资金效率，
     后者在 2022 这类系统性下跌期降低 beta。

运行：
    conda activate quant
    PYTHONPATH=. python scripts/joinquant_v7_validation.py
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.backtest.cn_cost import cn_trade_cost_yuan
from quant.backtest.metrics import summary
from quant.config import RAW_DATA_DIR
from quant.data.industry import industry_series
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.universe import DEFAULT_POOL
from quant.factor import factors as F

from scripts.joinquant_v6_validation import (
    BACKTEST_END,
    BACKTEST_START,
    CAPITAL,
    FACTOR_START,
    _neutralize_many,
    _passes_filter,
    weighted_rank_composite,
)

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = PROJECT_ROOT / "jointquant" / "v7"
OUT_CSV = OUT_DIR / "v7_validation.csv"
OUT_MD = OUT_DIR / "v7_validation.md"


def build_v7_factor(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    value: dict[str, pd.DataFrame],
    ind: pd.Series,
    *,
    quality_weight: float = 0.5,
    lowvol_weight: float = 0.5,
) -> pd.DataFrame:
    """v7 真实可执行因子：不依赖聚宽覆盖不足的 holder 字段。"""
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    value_raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
    }
    value_neut = _neutralize_many(value_raw, ind, log_mv)
    value_blend = weighted_rank_composite(
        value_neut,
        {"earnings_yield": 1.0, "cashflow_yield": 1.0, "sales_yield": 1.0},
    )

    raw = {
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "quality_roe": F.quality_roe(value["pe_ttm"], value["pb"]),
        "low_vol_60": F.low_volatility(close, 60),
    }
    lib = {"value_blend": value_blend}
    lib.update(_neutralize_many(raw, ind, log_mv))
    return weighted_rank_composite(
        lib,
        {
            "value_blend": 1.0,
            "growth_peg": 1.0,
            "amihud": 1.0,
            "quality_roe": quality_weight,
            "low_vol_60": lowvol_weight,
        },
    )


def build_market_equity(close: pd.DataFrame) -> pd.Series:
    """用本地股票池等权收益近似市场状态，避免验证脚本额外联网取指数。"""
    ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    bench_ret = ret.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + bench_ret).cumprod()


def exposure_for(mode: str, market_equity: pd.Series, date_idx: int) -> float:
    if mode == "fixed95":
        return 0.95
    if mode == "fixed98":
        return 0.98
    hist = market_equity.iloc[: date_idx + 1].dropna()
    if len(hist) < 220:
        return 0.95 if mode != "fixed98" else 0.98

    px = hist.iloc[-1]
    ma60 = hist.tail(60).mean()
    ma120 = hist.tail(120).mean()
    ma200 = hist.tail(200).mean()
    mom60 = px / hist.iloc[-61] - 1.0
    mom120 = px / hist.iloc[-121] - 1.0
    dd120 = px / hist.tail(120).max() - 1.0

    if mode == "regime_mild":
        if px < ma200 and mom120 < -0.10:
            return 0.78
        if px < ma120 and mom60 < -0.03:
            return 0.88
        if px > ma60 and mom60 > 0.06:
            return 0.98
        return 0.95
    if mode == "regime_soft":
        if px < ma200 and mom120 < -0.12:
            return 0.68
        if px < ma120 and (mom60 < -0.05 or dd120 < -0.15):
            return 0.80
        if px > ma60 and mom60 > 0.08:
            return 0.98
        return 0.93
    if mode == "regime_hard":
        if px < ma200 and mom120 < -0.12:
            return 0.55
        if px < ma120 and (mom60 < -0.05 or dd120 < -0.15):
            return 0.72
        if px > ma60 and mom60 > 0.08:
            return 0.98
        return 0.90
    raise ValueError(mode)


def select_targets(
    scores: pd.Series,
    prices: pd.Series,
    industry: dict[str, str],
    *,
    top_n: int,
    industry_cap: int,
    slot_value: float,
    close_full: pd.DataFrame,
    date_idx: int,
    filter_mode: str,
) -> list[str]:
    ranked = scores.dropna().sort_values(ascending=False)
    selected: list[str] = []
    counts: dict[str, int] = {}
    for stock in ranked.index:
        price = prices.get(stock, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        if price * 100 > slot_value * 1.15:
            continue
        if not _passes_filter(stock, date_idx, close_full, filter_mode):
            continue
        ind_name = industry.get(stock, "其他")
        if counts.get(ind_name, 0) >= industry_cap:
            continue
        selected.append(stock)
        counts[ind_name] = counts.get(ind_name, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def score_weights(
    selected: list[str],
    scores: pd.Series,
    *,
    exposure: float,
    mode: str,
) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    if mode == "equal":
        return pd.Series(exposure / len(selected), index=selected)

    tilt_map = {
        "tilt065": 0.65,
        "tilt100": 1.00,
        "tilt130": 1.30,
    }
    if mode not in tilt_map:
        raise ValueError(mode)
    tilt = tilt_map[mode]
    sc = scores.reindex(selected)
    rank = sc.rank(pct=True)
    raw = 1.0 + tilt * (rank - rank.mean())
    raw = raw.clip(lower=0.45, upper=1.55)
    weights = raw / raw.sum() * exposure

    # 小资金实盘不让单票过于膨胀；先轻约束，再归一化。
    avg = exposure / len(selected)
    weights = weights.clip(lower=avg * 0.58, upper=min(0.145, avg * 1.45))
    return weights / weights.sum() * exposure


def lot_backtest_v7(
    close_full: pd.DataFrame,
    factor_full: pd.DataFrame,
    market_equity: pd.Series,
    *,
    top_n: int,
    rebalance_every: int,
    industry_cap: int,
    filter_mode: str,
    exposure_mode: str,
    weight_mode: str,
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
    exposures: list[float] = []
    skipped_slots: list[int] = []
    prev_value = CAPITAL

    for i, date in enumerate(close.index):
        prices = close.loc[date]
        total_value = cash + float((shares * prices.fillna(0.0)).sum())
        if i % rebalance_every == 0:
            date_idx = full_dates.get_loc(date)
            prev_idx = max(date_idx - 1, 0)
            exposure = exposure_for(exposure_mode, market_equity, prev_idx)
            scores = factor_full.iloc[prev_idx].dropna().sort_values(ascending=False)
            scores = scores[scores.index.isin(prices.dropna().index)]
            slot = total_value * exposure / top_n
            selected = select_targets(
                scores,
                prices,
                industry,
                top_n=top_n,
                industry_cap=industry_cap,
                slot_value=slot,
                close_full=close_full,
                date_idx=prev_idx,
                filter_mode=filter_mode,
            )
            weights = score_weights(selected, scores, exposure=exposure, mode=weight_mode)

            target_shares = pd.Series(0.0, index=close.columns)
            for stock, weight in weights.items():
                target_value = total_value * weight
                target_shares[stock] = math.floor(target_value / prices[stock] / 100.0) * 100.0

            delta = target_shares - shares
            notional = (delta.abs() * prices).fillna(0.0)
            fee = cn_trade_cost_yuan(notional, slippage=0.0005)
            cash = total_value - float((target_shares * prices.fillna(0.0)).sum()) - fee
            shares = target_shares
            total_value = cash + float((shares * prices.fillna(0.0)).sum())
            turnovers.append(float(notional.sum() / max(total_value, 1.0)))
            holdings.append(int((shares > 0).sum()))
            fees.append(float(fee))
            cash_ratios.append(cash / total_value if total_value > 0 else 0.0)
            exposures.append(float((shares * prices.fillna(0.0)).sum() / total_value))
            skipped_slots.append(top_n - len(selected))

        values.append(total_value)
        returns.append(total_value / prev_value - 1.0 if len(values) > 1 else 0.0)
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
            "avg_exposure": float(np.mean(exposures)),
            "avg_skipped_slots": float(np.mean(skipped_slots)),
            "rebalance_count": len(turnovers),
        }
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [
        s for s in DEFAULT_POOL
        if (RAW_DATA_DIR / f"{s}.parquet").exists()
        and (RAW_DATA_DIR / f"{s}_value.parquet").exists()
    ]
    print(f"A-share pool: {len(symbols)} stocks")
    panels = build_ohlcv_panels(symbols, start=FACTOR_START, end=BACKTEST_END)
    close, amount = panels["close"], panels["amount"]
    value = build_value_panels(symbols, start=FACTOR_START, end=BACKTEST_END, align_to=close)
    ind = industry_series(list(close.columns))

    print(f"Price: {close.shape[0]}d x {close.shape[1]}")
    factors = {
        "q05_lv05": build_v7_factor(close, amount, value, ind, quality_weight=0.5, lowvol_weight=0.5),
        "q05_lv10": build_v7_factor(close, amount, value, ind, quality_weight=0.5, lowvol_weight=1.0),
        "q00_lv00": build_v7_factor(close, amount, value, ind, quality_weight=0.0, lowvol_weight=0.0),
    }
    market_equity = build_market_equity(close)

    rows = []
    baseline = lot_backtest_v7(
        close,
        factors["q05_lv05"],
        market_equity,
        top_n=10,
        rebalance_every=60,
        industry_cap=2,
        filter_mode="mom120_gt_neg10",
        exposure_mode="fixed95",
        weight_mode="equal",
    )
    rows.append(
        {
            "kind": "baseline_like_v6_jq",
            "factor_name": "q05_lv05",
            "top_n": 10,
            "rebalance_days": 60,
            "industry_cap": 2,
            "filter_mode": "mom120_gt_neg10",
            "exposure_mode": "fixed95",
            "weight_mode": "equal",
            **baseline,
        }
    )

    for factor_name, factor in factors.items():
        print(f"searching {factor_name} ...")
        for top_n in (8, 10, 12):
            for rebalance in (40, 50, 60):
                industry_cap = 2
                for filt in ("none", "mom120_gt_neg10"):
                    for exposure_mode in ("fixed95", "fixed98", "regime_mild", "regime_soft"):
                        for weight_mode in ("equal", "tilt065", "tilt100"):
                            m = lot_backtest_v7(
                                close,
                                factor,
                                market_equity,
                                top_n=top_n,
                                rebalance_every=rebalance,
                                industry_cap=industry_cap,
                                filter_mode=filt,
                                exposure_mode=exposure_mode,
                                weight_mode=weight_mode,
                            )
                            rows.append(
                                {
                                    "kind": "search",
                                    "factor_name": factor_name,
                                    "top_n": top_n,
                                    "rebalance_days": rebalance,
                                    "industry_cap": industry_cap,
                                    "filter_mode": filt,
                                    "exposure_mode": exposure_mode,
                                    "weight_mode": weight_mode,
                                    **m,
                                }
                            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["sharpe", "annualized_return"], ascending=False)
    df.to_csv(OUT_CSV, index=False)

    show_cols = [
        "kind", "factor_name", "top_n", "rebalance_days", "industry_cap",
        "filter_mode", "exposure_mode", "weight_mode",
        "total_return", "annualized_return", "sharpe", "max_drawdown",
        "avg_cash", "avg_exposure", "avg_turnover", "avg_holdings",
    ]
    top_sharpe = df.head(30)
    top_return = df.sort_values(["total_return", "sharpe"], ascending=False).head(20)
    robust = df[
        (df["max_drawdown"] <= 0.22)
        & (df["avg_holdings"] >= 9.0)
        & (df["factor_name"].isin(["q05_lv05", "q05_lv10"]))
    ].sort_values(["annualized_return", "sharpe"], ascending=False).head(20)

    chosen = df[
        (df["factor_name"].eq("q05_lv05"))
        & (df["top_n"].eq(10))
        & (df["rebalance_days"].eq(50))
        & (df["industry_cap"].eq(2))
        & (df["filter_mode"].eq("none"))
        & (df["exposure_mode"].eq("fixed95"))
        & (df["weight_mode"].eq("tilt065"))
    ]

    lines = [
        "# JoinQuant v7 Validation",
        "",
        f"- Window: {BACKTEST_START} ~ {BACKTEST_END}",
        f"- Pool: DEFAULT_POOL cached A-share large/mid caps ({len(symbols)} names)",
        "- Cost: 60k CNY, 100-share lots, min commission, stamp duty, transfer fee, slippage",
        "- v7 premise: holder disabled because JoinQuant v6 log showed holder_coverage=0.",
        "",
        "## Baseline Like Real v6 JQ",
        "",
        df[df["kind"].eq("baseline_like_v6_jq")][show_cols].to_markdown(index=False),
        "",
        "## Top 30 By Sharpe",
        "",
        top_sharpe[show_cols].to_markdown(index=False),
        "",
        "## Top 20 By Total Return",
        "",
        top_return[show_cols].to_markdown(index=False),
        "",
        "## Robust High-Return Candidates",
        "",
        robust[show_cols].to_markdown(index=False),
    ]
    if not chosen.empty:
        lines.extend(["", "## Suggested v7 Default", "", chosen[show_cols].to_markdown(index=False)])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_MD}")
    print(top_sharpe[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
