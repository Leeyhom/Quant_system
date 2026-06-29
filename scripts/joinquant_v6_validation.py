#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""joinquant_v6_validation —— v6 A股聚宽策略的本地小资金验证。

v6 不是继续微调 v4-ssot 的参数，而是修几个结构问题：
  1. 三个价值因子先合成一个 value_blend，避免价值重复投票；
  2. 加入正交信号：筹码集中度、质量 ROE、60 日低波；
  3. 120 日趋势只做风险过滤，避免便宜股价值陷阱；
  4. 用 6 万本金、100 股整数手、最低 5 元佣金的真实小资金成本验证。

运行：
    conda activate quant
    PYTHONPATH=. python scripts/joinquant_v6_validation.py
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
from quant.config import RAW_DATA_DIR
from quant.data import cn_holder_loader
from quant.data.industry import industry_series
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.universe import DEFAULT_POOL
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize


OUT_DIR = PROJECT_ROOT / "jointquant" / "v6"
OUT_CSV = OUT_DIR / "v6_validation.csv"
OUT_MD = OUT_DIR / "v6_validation.md"

FACTOR_START = "20180101"
BACKTEST_START = "20190101"
BACKTEST_END = "20251231"
CAPITAL = 60_000.0
MAX_EXPOSURE = 0.95


def build_holder_panels_cached(symbols: list[str], align_to: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """只用本地缓存构建股东户数面板，避免验证脚本意外联网。"""
    loaded = {}
    for sym in symbols:
        try:
            df = cn_holder_loader.load_parquet(sym)
        except FileNotFoundError:
            continue
        if df is not None and len(df) > 0:
            loaded[sym] = df
    if not loaded:
        return {}

    panels = {}
    for field in ["holder_num", "change_ratio", "avg_hold_num", "avg_market_cap"]:
        series = {}
        for sym, df in loaded.items():
            if field not in df.columns:
                continue
            s = (
                df[["notice_date", field]]
                .dropna(subset=["notice_date"])
                .drop_duplicates(subset=["notice_date"], keep="last")
                .set_index("notice_date")[field]
                .sort_index()
            )
            s.name = sym
            series[sym] = s
        if series:
            raw = pd.concat(series, axis=1).sort_index()
            union_idx = raw.index.union(align_to.index)
            panels[field] = (
                raw.reindex(union_idx)
                .ffill()
                .reindex(index=align_to.index, columns=align_to.columns)
            )
    return panels


def _neutralize_many(raw: dict[str, pd.DataFrame], ind: pd.Series, log_mv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: neutralize(fac, industry=ind, log_mv=log_mv, mode="full") for name, fac in raw.items()}


def weighted_rank_composite(factors: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """按因子名权重合成；缺失因子自动跳过。"""
    parts = []
    total_weight = None
    for name, fac in factors.items():
        w = float(weights.get(name, 1.0))
        r = fac.rank(axis=1, pct=True)
        parts.append(r * w)
        valid = r.notna().astype(float) * w
        total_weight = valid if total_weight is None else total_weight + valid
    score = sum(parts) / total_weight.replace(0, np.nan)
    return score


def build_v4_ssot_factor(close, amount, value, ind) -> pd.DataFrame:
    """v4-ssot 当前 5 因子基线：三个价值因子仍各占一票。"""
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
    }
    lib = _neutralize_many(raw, ind, log_mv)
    return combine_factors(*lib.values())


def build_v5_factor(close, amount, value, ind) -> pd.DataFrame:
    """v5 本地口径：五因子 + quality_roe 半权重。"""
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "quality_roe": F.quality_roe(value["pe_ttm"], value["pb"]),
    }
    lib = _neutralize_many(raw, ind, log_mv)
    return weighted_rank_composite(
        lib,
        {
            "earnings_yield": 1,
            "cashflow_yield": 1,
            "sales_yield": 1,
            "growth_peg": 1,
            "amihud": 1,
            "quality_roe": 0.5,
        },
    )


def build_v6_factor(close, amount, value, holder, panels, ind, *, include_holder: bool = True) -> pd.DataFrame:
    """v6 因子：价值降维 + 正交信号扩容。"""
    log_mv = np.log(value["total_mv"].replace(0, np.nan))

    value_raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
    }
    value_neut = _neutralize_many(value_raw, ind, log_mv)
    value_blend = combine_factors(*value_neut.values())

    raw = {
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "quality_roe": F.quality_roe(value["pe_ttm"], value["pb"]),
        "low_vol_60": F.low_volatility(close, 60),
    }
    if include_holder and "change_ratio" in holder:
        raw["holder_concentration"] = F.holder_concentration(holder["change_ratio"])

    lib = {"value_blend": value_blend}
    lib.update(_neutralize_many(raw, ind, log_mv))
    return weighted_rank_composite(
        lib,
        {
            "value_blend": 1.0,
            "growth_peg": 1.0,
            "amihud": 1.0,
            "holder_concentration": 1.0,
            "quality_roe": 0.5,
            "low_vol_60": 0.5,
        },
    )


def build_weighted_separate_factor(
    close,
    amount,
    value,
    holder,
    ind,
    *,
    quality_weight: float = 0.0,
    lowvol_weight: float = 0.0,
    holder_weight: float = 0.0,
) -> pd.DataFrame:
    """保留三价值因子的收益贡献，在其上叠加正交风险/质量信号。"""
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
    }
    weights = {
        "earnings_yield": 1.0,
        "cashflow_yield": 1.0,
        "sales_yield": 1.0,
        "growth_peg": 1.0,
        "amihud": 1.0,
    }
    if quality_weight > 0:
        raw["quality_roe"] = F.quality_roe(value["pe_ttm"], value["pb"])
        weights["quality_roe"] = quality_weight
    if lowvol_weight > 0:
        raw["low_vol_60"] = F.low_volatility(close, 60)
        weights["low_vol_60"] = lowvol_weight
    if holder_weight > 0 and "change_ratio" in holder:
        raw["holder_concentration"] = F.holder_concentration(holder["change_ratio"])
        weights["holder_concentration"] = holder_weight
    lib = _neutralize_many(raw, ind, log_mv)
    return weighted_rank_composite(lib, weights)


def _passes_filter(stock: str, date_idx: int, close_full: pd.DataFrame, mode: str) -> bool:
    if mode == "none":
        return True
    prices = close_full[stock].iloc[: date_idx + 1].dropna()
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
    current: list[str],
    hold_multiplier: float,
) -> list[str]:
    ranked = scores.dropna().sort_values(ascending=False)
    selected: list[str] = []
    counts: dict[str, int] = {}
    max_keep_rank = max(top_n, int(math.ceil(top_n * hold_multiplier)))

    def eligible(stock: str) -> bool:
        price = prices.get(stock, np.nan)
        if not np.isfinite(price) or price <= 0:
            return False
        if price * 100 > slot_value * 1.15:
            return False
        return _passes_filter(stock, date_idx, close_full, filter_mode)

    # 先保留仍在较高分位的老持仓，降低“微小分数差”带来的无效换手。
    if hold_multiplier > 1.0:
        rank_pos = {stock: i for i, stock in enumerate(ranked.index, start=1)}
        for stock in sorted(current, key=lambda s: rank_pos.get(s, 10**9)):
            if stock not in ranked.index or rank_pos.get(stock, 10**9) > max_keep_rank:
                continue
            if not eligible(stock):
                continue
            ind_name = industry.get(stock, "其他")
            if counts.get(ind_name, 0) >= industry_cap:
                continue
            selected.append(stock)
            counts[ind_name] = counts.get(ind_name, 0) + 1
            if len(selected) >= top_n:
                return selected

    for stock in ranked.index:
        if stock in selected or not eligible(stock):
            continue
        ind_name = industry.get(stock, "其他")
        if counts.get(ind_name, 0) >= industry_cap:
            continue
        selected.append(stock)
        counts[ind_name] = counts.get(ind_name, 0) + 1
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
    hold_multiplier: float = 1.0,
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
            prev_idx = max(date_idx - 1, 0)
            scores = factor_full.iloc[prev_idx].dropna().sort_values(ascending=False)
            scores = scores[scores.index.isin(prices.dropna().index)]
            slot = total_value * MAX_EXPOSURE / top_n
            current = shares[shares > 0].index.tolist()
            selected = select_targets(
                scores,
                prices,
                industry,
                top_n,
                industry_cap,
                slot,
                close_full,
                prev_idx,
                filter_mode,
                current,
                hold_multiplier,
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
    holder = build_holder_panels_cached(symbols, align_to=close)
    ind = industry_series(list(close.columns))

    print(f"Price: {close.shape[0]}d x {close.shape[1]} | holder cached columns: {len(holder.get('change_ratio', pd.DataFrame()).columns)}")

    factors = {
        "v4_ssot_5f": build_v4_ssot_factor(close, amount, value, ind),
        "v5_quality_half": build_v5_factor(close, amount, value, ind),
        "v6_value_blend": build_v6_factor(close, amount, value, holder, panels, ind, include_holder=True),
        "v6_value_blend_no_holder": build_v6_factor(close, amount, value, holder, panels, ind, include_holder=False),
        "v6_quality_lowvol_holder": build_weighted_separate_factor(
            close, amount, value, holder, ind,
            quality_weight=0.5, lowvol_weight=0.5, holder_weight=0.5,
        ),
        "v6_quality_lowvol": build_weighted_separate_factor(
            close, amount, value, holder, ind,
            quality_weight=0.5, lowvol_weight=0.5, holder_weight=0.0,
        ),
        "v6_lowvol_half": build_weighted_separate_factor(
            close, amount, value, holder, ind,
            quality_weight=0.0, lowvol_weight=0.5, holder_weight=0.0,
        ),
        "v6_quality_holder": build_weighted_separate_factor(
            close, amount, value, holder, ind,
            quality_weight=0.5, lowvol_weight=0.0, holder_weight=0.5,
        ),
    }

    rows = []
    # Baselines close to actual deployed shapes.
    baseline_configs = [
        ("v4_ssot_5f", 10, 40, 1, "none", 1.0),
        ("v5_quality_half", 10, 60, 1, "mom120_gt_neg10", 1.0),
    ]
    for factor_name, top_n, rebalance, industry_cap, filt, hold in baseline_configs:
        m = lot_backtest(close, factors[factor_name], top_n, rebalance, industry_cap, filt, hold)
        rows.append(
            {
                "kind": "baseline",
                "factor_name": factor_name,
                "top_n": top_n,
                "rebalance_days": rebalance,
                "industry_cap": industry_cap,
                "filter_mode": filt,
                "hold_multiplier": hold,
                **m,
            }
        )

    for factor_name in [
        "v6_value_blend",
        "v6_value_blend_no_holder",
        "v6_quality_lowvol_holder",
        "v6_quality_lowvol",
        "v6_lowvol_half",
        "v6_quality_holder",
    ]:
        for top_n in (8, 10, 12):
            for rebalance in (40, 50, 60):
                for industry_cap in (1, 2):
                    for filt in ("none", "mom120_gt_neg10", "not_deep_downtrend", "trend_combo"):
                        for hold in (1.0, 1.5):
                            m = lot_backtest(close, factors[factor_name], top_n, rebalance, industry_cap, filt, hold)
                            rows.append(
                                {
                                    "kind": "search",
                                    "factor_name": factor_name,
                                    "top_n": top_n,
                                    "rebalance_days": rebalance,
                                    "industry_cap": industry_cap,
                                    "filter_mode": filt,
                                    "hold_multiplier": hold,
                                    **m,
                                }
                            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["sharpe", "annualized_return"], ascending=False)
    df.to_csv(OUT_CSV, index=False)

    show_cols = [
        "kind", "factor_name", "top_n", "rebalance_days", "industry_cap",
        "filter_mode", "hold_multiplier", "total_return", "annualized_return",
        "sharpe", "max_drawdown", "avg_cash", "avg_turnover", "avg_holdings",
    ]
    top = df.head(25)
    baseline = df[df["kind"].eq("baseline")]
    chosen = df[
        (df["factor_name"].eq("v6_value_blend"))
        & (df["top_n"].eq(10))
        & (df["rebalance_days"].eq(60))
        & (df["industry_cap"].eq(2))
        & (df["filter_mode"].eq("mom120_gt_neg10"))
        & (df["hold_multiplier"].eq(1.0))
    ]

    lines = [
        "# JoinQuant v6 Validation",
        "",
        f"- Window: {BACKTEST_START} ~ {BACKTEST_END}",
        f"- Pool: DEFAULT_POOL cached A-share large/mid caps ({len(symbols)} names)",
        "- Cost: 60k CNY, 100-share lots, min commission, stamp duty, transfer fee, slippage",
        "",
        "## Baselines",
        "",
        baseline[show_cols].to_markdown(index=False),
        "",
        "## Top 25 By Sharpe",
        "",
        top[show_cols].to_markdown(index=False),
    ]
    if not chosen.empty:
        lines.extend(["", "## Suggested v6 Default", "", chosen[show_cols].to_markdown(index=False)])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_MD}")
    print(top[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
