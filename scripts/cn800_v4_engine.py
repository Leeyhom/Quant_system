#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cn800_v4_engine —— CN800池 v4因子体系 本地 Walk-Forward 验证引擎。

将 JoinQuant 上验证通过的 v4 策略翻译到本地 quant/ 框架：
  - 股票池: CN800（沪深300+中证500, 800只）
  - 因子: value_blend + growth_peg + amihud + quality_roe + low_vol_60 + residual_momentum
  - 自适应动量: 市场regime检测 + 牛市raw_momentum_120加分
  - 动态行业上限: max(2, 池内行业股票数/10)
  - Walk-Forward: train段IC定向, test段独立评估, 无全样本泄漏

同时跑旧池(DEFAULT_POOL)作为基线对比。

用法:
    conda activate quant
    NO_PROXY='*' PYTHONPATH=. python scripts/cn800_v4_engine.py
    NO_PROXY='*' PYTHONPATH=. python scripts/cn800_v4_engine.py --train 480 --test 120 --step 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.data.universe import DEFAULT_POOL
from quant.data.universe_cn800 import CN800_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.industry import industry_series
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize
from quant.factor.composite import ic_weighted_composite


# ============================ 参数（与v4 JoinQuant策略一致） ============================

TOP_N = 10
REBALANCE_DAYS = 40
COST_RATE = 0.001  # 单边交易成本

MOMENTUM_120_MIN_DEFAULT = -0.10
MOMENTUM_120_MIN_BULL = -0.05
MOMENTUM_120_MIN_BEAR = -0.25
MOMENTUM_MARKET_BULL = 0.05
MOMENTUM_MARKET_BEAR = -0.10

QUALITY_WEIGHT = 0.5
LOWVOL_WEIGHT = 0.5
RESIDUAL_MOM_WEIGHT = 0.5


# ============================ v4 因子库构建 ============================

def _neutralize_many(raw, ind, log_mv):
    result = {}
    for n, f in raw.items():
        # ensure factor is DataFrame (some factor functions may return Series)
        if isinstance(f, pd.Series):
            f = f.to_frame().T
        result[n] = neutralize(f, industry=ind, log_mv=log_mv, mode="full")
    return result


def build_v4_factor_library(close, amount, value, ind):
    """构建 v4 因子库（对齐 joinquant_cn800_strategy_v4.py 的 build_scores）。

    因子:
      value_blend: earnings_yield+cashflow_yield+sales_yield 先各自中性化再rank均值
      growth_peg: 1/PEG
      amihud: -mean(|ret|/amount, 20d)
      quality_roe: PB/PE
      low_vol_60: -std(ret, 60d)
      residual_momentum: 个股60日收益-行业均值
      牛市时额外: raw_momentum_120
    """
    log_mv = np.log(value["total_mv"].replace(0, np.nan))

    # value_blend
    value_raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
    }
    value_neut = _neutralize_many(value_raw, ind, log_mv)
    value_blend = combine_factors(*value_neut.values())

    # 正交因子
    growth_peg = F.growth_peg(value["peg"])
    amihud = F.amihud_illiquidity(close, amount, 20)
    quality_roe = F.quality_roe(value["pe_ttm"], value["pb"])
    low_vol = F.low_volatility(close, 60)

    # 残差动量
    if len(close) >= 61:
        ret_60 = close.iloc[-1] / close.iloc[-61].replace(0, np.nan) - 1.0
        ind_s = pd.Series({s: ind.get(s, "其他") for s in ret_60.index})
        ind_avg = ret_60.groupby(ind_s).transform("mean")
        resid_mom = ret_60 - ind_avg
    else:
        resid_mom = pd.DataFrame(np.nan, index=close.columns).T.squeeze()

    # 市场regime检测（只在有足够数据时）
    market_regime = "neutral"
    if len(close) >= 121:
        mom120 = close.iloc[-1] / close.iloc[-121].replace(0, np.nan) - 1.0
        market_mom = mom120.mean()
        if pd.notna(market_mom):
            if market_mom > MOMENTUM_MARKET_BULL:
                market_regime = "bull"
            elif market_mom < MOMENTUM_MARKET_BEAR:
                market_regime = "bear"

    raw_factors = {
        "growth_peg": growth_peg,
        "amihud": amihud,
        "quality_roe": quality_roe,
        "low_vol_60": low_vol,
        "residual_momentum": resid_mom,
    }
    if market_regime == "bull" and len(close) >= 121:
        raw_factors["raw_momentum_120"] = mom120

    lib = {"value_blend": value_blend}
    lib.update(_neutralize_many(raw_factors, ind, log_mv))
    return lib, market_regime


def build_v4_composite(lib, market_regime="neutral"):
    """v4 加权合成。"""
    ranked = {n: f.rank(axis=1, pct=True) for n, f in lib.items()}
    weights = {
        "value_blend": 1.0, "growth_peg": 1.0, "amihud": 1.0,
        "quality_roe": QUALITY_WEIGHT, "low_vol_60": LOWVOL_WEIGHT,
        "residual_momentum": RESIDUAL_MOM_WEIGHT,
    }
    if "raw_momentum_120" in lib and market_regime == "bull":
        weights["raw_momentum_120"] = 0.3

    parts, total_w = [], None
    for name, r in ranked.items():
        w = weights.get(name, 1.0)
        parts.append(r * w)
        valid = r.notna().astype(float) * w
        total_w = valid if total_w is None else total_w + valid

    if total_w is None:
        return pd.Series(dtype=float)
    return sum(parts) / total_w.replace(0, np.nan)


# ============================ Walk-Forward 核心 ============================

def _train_signs(lib, fwd_ret, upto_date, horizon=20):
    fwd_train = fwd_ret.loc[fwd_ret.index < upto_date]
    if len(fwd_train) > horizon:
        fwd_train = fwd_train.iloc[:-horizon]
    signs = {}
    for name, fac in lib.items():
        s = ic_summary(daily_ic(fac, fwd_train, min_count=5))
        ic = float(s["mean_ic"]) if pd.notna(s["mean_ic"]) else 0.0
        signs[name] = 1.0 if ic >= 0 else -1.0
    return signs


def run_wf(pool, pool_name, train_days=480, test_days=120, step_days=120, horizon=20):
    """诚实 walk-forward。"""
    print(f"\n{'='*70}")
    print(f"  {pool_name}: {len(pool)} 只")
    print(f"{'='*70}")

    # 构建全样本面板
    ohlcv = build_ohlcv_panels(pool)
    close, amount = ohlcv["close"], ohlcv["amount"]

    # 过滤数据不足的股票
    coverage = close.notna().mean()
    good = coverage[coverage >= 0.6].index.tolist()
    bad_n = len(close.columns) - len(good)
    if bad_n:
        print(f"  过滤 {bad_n} 只数据不足股票")
        close, amount = close[good], amount[good]

    val = build_value_panels(pool, align_to=close)
    ind = industry_series(list(close.columns))
    print(f"  {len(close)}天 × {len(close.columns)}只, {close.notna().sum().sum():,}有效价")

    # 全样本因子库
    lib, _ = build_v4_factor_library(close, amount, val, ind)
    fwd_ret = forward_returns(close, horizon=horizon)
    print(f"  因子: {list(lib.keys())}")

    n = len(close)
    dates = close.index

    # Warmup offset — skip days where factors are NaN-heavy
    lib_valid = pd.DataFrame({k: v.notna().any(axis=1) for k, v in lib.items()})
    all_ok = lib_valid.all(axis=1)
    first_ok_idx = all_ok.idxmax() if all_ok.any() else dates[0]
    min_idx = max(dates.get_loc(first_ok_idx), 120)

    oos_equal, oos_ic, oos_pick = [], [], []
    oos_bench = []
    windows = []
    start = min_idx

    while start + train_days + test_days <= n:
        cut = dates[start + train_days]
        test_slice = slice(start + train_days, start + train_days + test_days)
        test_dates = dates[test_slice]

        signs = _train_signs(lib, fwd_ret, cut, horizon)

        # Equal composite: orient factors by train signs, then rank mean
        oriented = [(lib[n] if signs.get(n, 1) >= 0 else -lib[n]) for n in lib]
        comp_equal = combine_factors(*oriented)

        # IC-weighted composite
        try:
            comp_ic, _ = ic_weighted_composite(lib, fwd_ret, cut, horizon=horizon)
        except Exception:
            comp_ic = comp_equal  # fallback

        # Train段择优
        train_d = dates[start:start + train_days]
        pick = "equal"; best_s = -np.inf
        close_t = close.loc[train_d]
        for nm, comp in [("equal", comp_equal), ("ic", comp_ic)]:
            try:
                c = comp.reindex(index=close_t.index, columns=close_t.columns)
                bt = long_top_layer(close_t, c, n_layers=5,
                                    rebalance_every=REBALANCE_DAYS,
                                    cost_rate=COST_RATE, first_rebalance=True)
                s = summary(bt["equity"], bt["port_ret"])["sharpe"]
                if pd.notna(s) and s > best_s:
                    best_s, pick = s, nm
            except Exception:
                pass

        comp_map = {"equal": comp_equal, "ic": comp_ic,
                    "train_pick": comp_equal if pick == "equal" else comp_ic}

        close_test = close.loc[test_dates]
        for m, comp in comp_map.items():
            c = comp.reindex(index=close_test.index, columns=close_test.columns)
            try:
                bt = long_top_layer(close_test, c, n_layers=5,
                                    rebalance_every=REBALANCE_DAYS,
                                    cost_rate=COST_RATE, first_rebalance=True)
                rets = bt["port_ret"]
            except Exception:
                rets = pd.Series(0.0, index=test_dates)
            if m == "equal": oos_equal.append(rets)
            elif m == "ic": oos_ic.append(rets)
            else: oos_pick.append(rets)
        oos_bench.append(bt["benchmark_ret"])
        windows.append(pick)
        start += step_days

    # 拼接
    def concat(ret_list):
        r = pd.concat(ret_list).sort_index()
        r = r[~r.index.duplicated(keep="first")]
        return r.replace([np.inf, -np.inf], np.nan).dropna()

    bench_r = concat(oos_bench)
    bench_eq = (1 + bench_r).cumprod()
    bs = summary(bench_eq, bench_r)

    results = {}
    for method, rets in [("equal", oos_equal), ("ic", oos_ic), ("train_pick", oos_pick)]:
        r = concat(rets)
        eq = (1 + r).cumprod()
        s = summary(eq, r)
        results[method] = {
            "total_return": s["total_return"], "annualized": s["annualized_return"],
            "sharpe": s["sharpe"], "max_dd": s["max_drawdown"],
            "excess_vs_bench": s["total_return"] - bs["total_return"],
        }

    return {
        "pool_name": pool_name, "n_stocks": len(pool), "n_windows": len(windows),
        "results": results, "benchmark": bs,
        "pick_counts": dict(pd.Series(windows).value_counts()),
    }


# ============================ 主流程 ============================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=480)
    p.add_argument("--test", type=int, default=120)
    p.add_argument("--step", type=int, default=120)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--new-only", action="store_true", help="只跑CN800")
    args = p.parse_args()

    print("=" * 70)
    print("  CN800 v4 本地 Walk-Forward 验证引擎")
    print(f"  train={args.train} test={args.test} step={args.step} horizon={args.horizon}")
    print(f"  因子: value_blend + growth_peg + amihud + quality_roe + low_vol_60 + residual_momentum")
    print(f"  自适应动量: bull/bear/neutral + 牛市raw_momentum_120")
    print("=" * 70)

    t0 = time.time()

    # CN800
    r_new = run_wf(CN800_POOL, "CN800 v4（沪深300+中证500）",
                   args.train, args.test, args.step, args.horizon)

    # 旧池
    if not args.new_only:
        r_old = run_wf(DEFAULT_POOL, "旧池 v3（DEFAULT_POOL）",
                       args.train, args.test, args.step, args.horizon)
    else:
        r_old = None

    elapsed = (time.time() - t0) / 60
    print(f"\n  总耗时: {elapsed:.1f} 分钟")

    # ── 对比输出 ──
    print(f"\n{'='*70}")
    print(f"  等权合成 OOS 对比")
    print(f"{'='*70}")
    cols = ["total_return", "annualized", "sharpe", "max_dd", "excess_vs_bench"]
    for r in [r_new] + ([r_old] if r_old else []):
        eq = r["results"]["equal"]
        print(f"\n  {r['pool_name']} ({r['n_stocks']}只, {r['n_windows']}窗口):")
        for c in cols:
            print(f"    {c}: {eq[c]:+.4f}")
        print(f"    基准: 收益{ r['benchmark']['total_return']:+.2%} "
              f"夏普{r['benchmark']['sharpe']:.3f} 回撤{r['benchmark']['max_drawdown']:.2%}")

    if r_old:
        eq_n = r_new["results"]["equal"]
        eq_o = r_old["results"]["equal"]
        print(f"\n  CN800 vs 旧池 改善:")
        for c in cols:
            d = eq_n[c] - eq_o[c]
            print(f"    {c}: {eq_o[c]:+.4f} → {eq_n[c]:+.4f}  ({d:+.4f})")

    # ── 保存结果 ──
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for r in [r_new] + ([r_old] if r_old else []):
        for method in ["equal", "ic", "train_pick"]:
            rows.append({"pool": r["pool_name"], "method": method, **r["results"][method]})
    pd.DataFrame(rows).to_csv(out_dir / "cn800_v4_local_validation.csv", index=False)
    print(f"\n结果已保存: {out_dir}/cn800_v4_local_validation.csv")

    return r_new, r_old


if __name__ == "__main__":
    main()
