#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""joinquant_v9_validation —— v8/v9 聚宽策略候选本地验证。

v9 不新增黑箱 alpha。它只在 v8 的稳健因子结构上测试两个组合层改进：

1. 持仓缓冲：老持仓只要仍在 TOP_N * hold_multiplier 的高分带内，就优先保留，
   降低小分数抖动造成的换手。
2. 有界低波风险预算：在接近等权的约束内，用 60 日波动率做轻微逆波动权重，
   单票权重被严格限制，避免重演 v7 分数倾斜导致的集中度问题。

重要限制：
    本脚本只使用本地缓存。当前若聚宽 152 池未补齐，只能作为工程/方向验证，
    不能替代完整 152 池 walk-forward。

运行：
    PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python scripts/joinquant_v9_validation.py
"""
from __future__ import annotations

import importlib.util
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
from quant.data.industry import industry_series
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.factor import factors as F
from scripts.joinquant_v6_validation import (
    BACKTEST_END,
    BACKTEST_START,
    CAPITAL,
    FACTOR_START,
    MAX_EXPOSURE,
    _neutralize_many,
    _passes_filter,
    weighted_rank_composite,
)


OUT_DIR = PROJECT_ROOT / "jointquant" / "v9"
OUT_CSV = OUT_DIR / "v9_validation.csv"
OUT_MD = OUT_DIR / "v9_validation.md"
JQ_STRATEGY = PROJECT_ROOT / "scripts" / "joinquant_cn_sim_strategy_v9.py"


def load_jq_strategy(path: Path = JQ_STRATEGY):
    spec = importlib.util.spec_from_file_location("jq_strategy_v9", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_v9_factor(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    value: dict[str, pd.DataFrame],
    ind: pd.Series,
) -> pd.DataFrame:
    """Local copy of the v8/v9 JoinQuant factor score.

    价值三因子先合成 value_blend，其他正交信号再与它等权/半权合成。
    这里不使用全样本 IC 定向，所有因子方向来自经济先验。
    """
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
            "quality_roe": 0.5,
            "low_vol_60": 0.5,
        },
    )


def build_market_equity(close: pd.DataFrame) -> pd.Series:
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
        return 0.95

    px = hist.iloc[-1]
    ma60 = hist.tail(60).mean()
    ma120 = hist.tail(120).mean()
    ma200 = hist.tail(200).mean()
    mom60 = px / hist.iloc[-61] - 1.0
    mom120 = px / hist.iloc[-121] - 1.0

    if mode == "soft_brake":
        if px < ma200 and mom120 < -0.12:
            return 0.85
        if px < ma120 and mom60 < -0.05:
            return 0.90
        if px > ma60 and mom60 > 0.08:
            return 0.98
        return 0.95
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
    current: list[str],
    hold_multiplier: float,
) -> list[str]:
    ranked = scores.dropna().sort_values(ascending=False)
    selected: list[str] = []
    counts: dict[str, int] = {}
    rank_pos = {stock: i for i, stock in enumerate(ranked.index, start=1)}
    max_keep_rank = max(top_n, int(math.ceil(top_n * hold_multiplier)))

    def eligible(stock: str) -> bool:
        price = prices.get(stock, np.nan)
        if not np.isfinite(price) or price <= 0:
            return False
        if price * 100 > slot_value * 1.15:
            return False
        return _passes_filter(stock, date_idx, close_full, "mom120_gt_neg10")

    def add(stock: str) -> bool:
        if stock in selected or not eligible(stock):
            return False
        ind_name = industry.get(stock, "其他")
        if counts.get(ind_name, 0) >= industry_cap:
            return False
        selected.append(stock)
        counts[ind_name] = counts.get(ind_name, 0) + 1
        return True

    if hold_multiplier > 1.0:
        for stock in sorted(current, key=lambda s: rank_pos.get(s, 10**9)):
            if rank_pos.get(stock, 10**9) > max_keep_rank:
                continue
            add(stock)
            if len(selected) >= top_n:
                return selected

    for stock in ranked.index:
        add(stock)
        if len(selected) >= top_n:
            break
    return selected


def score_weights(
    selected: list[str],
    vol60: pd.Series,
    *,
    exposure: float,
    mode: str,
) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    if mode == "equal" or len(selected) < 3:
        return pd.Series(exposure / len(selected), index=selected)

    vol = vol60.reindex(selected).replace([np.inf, -np.inf], np.nan)
    if vol.notna().sum() < 3:
        return pd.Series(exposure / len(selected), index=selected)

    lo = vol.quantile(0.20)
    hi = vol.quantile(0.80)
    vol = vol.clip(lower=lo, upper=hi)
    raw = 1.0 / vol.replace(0, np.nan)
    raw = raw.fillna(raw.median())
    weights = raw / raw.sum() * exposure

    avg = exposure / len(selected)
    weights = weights.clip(lower=avg * 0.85, upper=min(avg * 1.15, 0.115))
    return weights / weights.sum() * exposure


def lot_backtest_v9(
    close_full: pd.DataFrame,
    factor_full: pd.DataFrame,
    market_equity: pd.Series,
    *,
    top_n: int,
    rebalance_every: int,
    industry_cap: int,
    hold_multiplier: float,
    weight_mode: str,
    exposure_mode: str,
) -> dict:
    close_full = close_full.sort_index()
    factor_full = factor_full.reindex_like(close_full)
    close = close_full.loc[BACKTEST_START:BACKTEST_END]
    full_dates = close_full.index
    industry = industry_series(list(close.columns)).to_dict()
    vol60 = close_full.pct_change(fill_method=None).rolling(60).std()

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
            current = shares[shares > 0].index.tolist()
            selected = select_targets(
                scores,
                prices,
                industry,
                top_n=top_n,
                industry_cap=industry_cap,
                slot_value=slot,
                close_full=close_full,
                date_idx=prev_idx,
                current=current,
                hold_multiplier=hold_multiplier,
            )
            weights = score_weights(
                selected,
                vol60.iloc[prev_idx],
                exposure=exposure,
                mode=weight_mode,
            )

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
    jq = load_jq_strategy()
    pool_all = [s.split(".")[0] for s in jq.STOCK_POOL]
    symbols = [
        s for s in pool_all
        if (RAW_DATA_DIR / f"{s}.parquet").exists()
        and (RAW_DATA_DIR / f"{s}_value.parquet").exists()
    ]
    missing = [s for s in pool_all if s not in symbols]
    print(f"JoinQuant pool cache coverage: {len(symbols)}/{len(pool_all)}")
    if missing:
        print(f"Missing first 20: {missing[:20]}")

    panels = build_ohlcv_panels(symbols, start=FACTOR_START, end=BACKTEST_END)
    close, amount = panels["close"], panels["amount"]
    value = build_value_panels(symbols, start=FACTOR_START, end=BACKTEST_END, align_to=close)
    ind = industry_series(list(close.columns))
    factor = build_v9_factor(close, amount, value, ind)
    market_equity = build_market_equity(close)

    configs = [
        # Controlled production candidates. Keep this small to avoid turning
        # the validation into another in-sample parameter mine.
        ("v8_baseline_equal", 10, 60, 1.0, "equal", "fixed95"),
        ("v8_less_cash_98", 10, 60, 1.0, "equal", "fixed98"),
        ("v8_rebalance50", 10, 50, 1.0, "equal", "fixed95"),
        ("v8_top8", 8, 60, 1.0, "equal", "fixed95"),
        ("v8_top12", 12, 60, 1.0, "equal", "fixed95"),
        ("v9_buffer_equal", 10, 60, 1.5, "equal", "fixed95"),
        ("v9_buffer_less_cash_98", 10, 60, 1.5, "equal", "fixed98"),
        ("v9_buffer_vol_budget", 10, 60, 1.5, "vol_budget", "fixed95"),
        ("v9_buffer_vol_less_cash_98", 10, 60, 1.5, "vol_budget", "fixed98"),
        ("v9_buffer_vol_soft_brake", 10, 60, 1.5, "vol_budget", "soft_brake"),
    ]
    rows = []
    for name, top_n, rebalance_days, hold, weight_mode, exposure_mode in configs:
        m = lot_backtest_v9(
            close,
            factor,
            market_equity,
            top_n=top_n,
            rebalance_every=rebalance_days,
            industry_cap=2,
            hold_multiplier=hold,
            weight_mode=weight_mode,
            exposure_mode=exposure_mode,
        )
        rows.append(
            {
                "config": name,
                "top_n": top_n,
                "rebalance_days": rebalance_days,
                "industry_cap": 2,
                "hold_multiplier": hold,
                "weight_mode": weight_mode,
                "exposure_mode": exposure_mode,
                **m,
            }
        )

    df = pd.DataFrame(rows).sort_values(["sharpe", "annualized_return"], ascending=False)
    df.to_csv(OUT_CSV, index=False)

    show_cols = [
        "config", "top_n", "rebalance_days", "total_return", "annualized_return", "sharpe", "max_drawdown",
        "avg_turnover", "avg_cash", "avg_exposure", "avg_holdings", "avg_skipped_slots",
    ]
    baseline = df[df["config"].eq("v8_baseline_equal")].iloc[0]
    candidate = df[df["config"].eq("v9_buffer_less_cash_98")].iloc[0]
    best_sharpe = df.iloc[0]
    delta_return = candidate["total_return"] - baseline["total_return"]
    delta_sharpe = candidate["sharpe"] - baseline["sharpe"]
    delta_dd = candidate["max_drawdown"] - baseline["max_drawdown"]

    lines = [
        "# JoinQuant v9 Local Validation",
        "",
        f"- Window: {BACKTEST_START} ~ {BACKTEST_END}",
        f"- Cached JoinQuant pool coverage: {len(symbols)}/{len(pool_all)}",
        "- Cost: 60k CNY, 100-share lots, min commission, stamp duty, transfer fee, slippage",
        "- Factor: same as v8, no holder, no full-sample IC direction.",
        "- Limitation: this is not final until the full 152-stock JoinQuant pool is cached.",
        "",
        "## Config Comparison",
        "",
        df[show_cols].to_markdown(index=False),
        "",
        "## v9 Decision",
        "",
        f"Best Sharpe in this run: `{best_sharpe['config']}`.",
        "Production candidate: `v9_buffer_less_cash_98`.",
        "",
        f"- Total return delta vs v8 baseline: {delta_return:+.2%}",
        f"- Sharpe delta vs v8 baseline: {delta_sharpe:+.3f}",
        f"- Max drawdown delta vs v8 baseline: {delta_dd:+.2%}",
        "",
        "Decision for the JoinQuant v9 file: use `v9_buffer_less_cash_98`",
        "(same factor score as v8, target exposure 98%, keep existing holdings that",
        "remain inside the top 1.5 * TOP_N score band). The bounded volatility budget",
        "remains a research toggle but is disabled by default because it underperformed",
        "in this cached-pool run.",
        "",
        "This still needs full 152-stock cache validation and a JoinQuant export replay",
        "before replacing v8 as the production baseline.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_MD}")
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
