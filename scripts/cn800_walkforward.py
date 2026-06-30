#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cn800_walkforward —— CN800 扩展池 Walk-Forward 验证（v9因子，诚实OOS）。

核心问题：800只沪深300+中证500 vs 152只旧池，多因子截面策略是否显著改善？

严格纪律（与 cn_walkforward_honest.py 一致）：
  ① 因子方向只用每个窗口 train 段 IC 决定，test 段零信息泄漏
  ② train IC 裁掉末尾 horizon 天（防未来收益重叠污染）
  ③ v9 因子口径: value_blend + growth_peg + amihud + quality_roe + low_vol_60
  ④ 同时比较三种合成方式，不事后择优
  ⑤ 同时跑旧池(152)和新池(800)，同口径对比

用法:
    NO_PROXY='*' PYTHONPATH=. python scripts/cn800_walkforward.py
    NO_PROXY='*' PYTHONPATH=. python scripts/cn800_walkforward.py --train 480 --test 120 --step 120
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
from quant.factor.composite import ic_weighted_composite
from quant.factor.neutralize import neutralize


# ───────────────────────── v9 因子构建（对齐聚宽生产策略） ─────────────────────────

def _neutralize_many(
    raw: dict[str, pd.DataFrame], ind: pd.Series, log_mv: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """行业+市值双中性化。"""
    return {
        name: neutralize(fac, industry=ind, log_mv=log_mv, mode="full")
        for name, fac in raw.items()
    }


def build_v9_factor_library(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    value: dict[str, pd.DataFrame],
    ind: pd.Series,
) -> dict[str, pd.DataFrame]:
    """构建 v8/v9 生产策略因子库（无 holder，5因子）。

    与 joinquant_cn_sim_strategy_v9.py 的 build_scores_v9 逐口径对齐：
      - value_blend: earnings_yield + cashflow_yield + sales_yield 先各自中性化,
        再 rank(pct) 横向均值合成（三价值合并为一票）
      - growth_peg: 1/PEG (日频倒数, =增速/PE)
      - amihud: -mean(|ret|/amount, 20d) (非流动性取负)
      - quality_roe: PB/PE (ROE 代理, 半权重)
      - low_vol_60: -std(ret, 60d) (低波偏好, 半权重)
    """
    log_mv = np.log(value["total_mv"].replace(0, np.nan))

    # value_blend: 三个价值因子各自中性化后 rank 均值合成
    value_raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
    }
    value_neut = _neutralize_many(value_raw, ind, log_mv)
    value_blend = combine_factors(*value_neut.values())  # rank mean

    # 正交因子
    raw = {
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "quality_roe": F.quality_roe(value["pe_ttm"], value["pb"]),
        "low_vol_60": F.low_volatility(close, 60),
    }

    lib = {"value_blend": value_blend}
    lib.update(_neutralize_many(raw, ind, log_mv))
    return lib


def build_v9_composite(
    lib: dict[str, pd.DataFrame],
    signs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """v9 加权合成: value_blend(1.0) + growth_peg(1.0) + amihud(1.0)
       + quality_roe(0.5) + low_vol_60(0.5)

    signs: 可选方向符号（walk-forward train段IC决定），None=使用先验方向(全正向)
    """
    weights = {
        "value_blend": 1.0,
        "growth_peg": 1.0,
        "amihud": 1.0,
        "quality_roe": 0.5,
        "low_vol_60": 0.5,
    }
    if signs is None:
        signs = {k: 1.0 for k in lib}

    parts = []
    total_valid = None
    for name, fac in lib.items():
        w = float(weights.get(name, 1.0))
        s = signs.get(name, 1.0)
        oriented = fac if s >= 0 else -fac
        r = oriented.rank(axis=1, pct=True)
        parts.append(r * w)
        valid = r.notna().astype(float) * w
        total_valid = valid if total_valid is None else total_valid + valid

    return sum(parts) / total_valid.replace(0, np.nan)


# ───────────────────────── Walk-Forward 核心 ─────────────────────────

def _train_signs(
    lib: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    upto_date: pd.Timestamp,
    horizon: int = 20,
) -> dict[str, float]:
    """只用 train 段（裁掉末尾 horizon 重叠）IC 给每个因子定方向。"""
    fwd_train = fwd_ret.loc[fwd_ret.index < upto_date]
    if len(fwd_train) > horizon:
        fwd_train = fwd_train.iloc[:-horizon]
    signs = {}
    for name, fac in lib.items():
        s = ic_summary(daily_ic(fac, fwd_train, min_count=5))
        ic = float(s["mean_ic"]) if pd.notna(s["mean_ic"]) else 0.0
        signs[name] = 1.0 if ic >= 0 else -1.0
    return signs


def _equal_composite(lib: dict, signs: dict) -> pd.DataFrame:
    """等权 rank 平均（方向由 train IC 决定）。"""
    oriented = [(lib[name] if signs[name] >= 0 else -lib[name]) for name in lib]
    return combine_factors(*oriented)


def _train_pick(
    close_train: pd.DataFrame,
    comp_equal: pd.DataFrame,
    comp_ic: pd.DataFrame,
    rebalance_days: int = 40,
    cost_rate: float = 0.001,
) -> str:
    """在 train 段比较 equal vs ic 的夏普，返回更好的方法名。"""
    best, best_sharpe = "equal", -np.inf
    for name, comp in [("equal", comp_equal), ("ic", comp_ic)]:
        try:
            bt = long_top_layer(
                close_train,
                comp.reindex_like(close_train),
                n_layers=5,
                rebalance_every=rebalance_days,
                cost_rate=cost_rate,
                first_rebalance=True,
            )
            s = summary(bt["equity"], bt["port_ret"])["sharpe"]
        except Exception:
            s = -np.inf
        if s > best_sharpe:
            best, best_sharpe = name, s
    return best


def run_walkforward(
    pool: list[str],
    pool_name: str,
    train_days: int = 480,
    test_days: int = 120,
    step_days: int = 120,
    horizon: int = 20,
    rebalance_days: int = 40,
    cost_rate: float = 0.001,
) -> dict:
    """对给定股票池跑诚实 walk-forward。

    返回 dict 含: oos_returns, bench_returns, windows, pick_distribution, metrics_df
    """
    t0 = time.time()
    print(f"\n{'─'*70}")
    print(f"  {pool_name}: {len(pool)} 只股票")
    print(f"{'─'*70}")

    # 构建面板
    print("  构建面板...", end=" ", flush=True)
    ohlcv = build_ohlcv_panels(pool)
    close, amount = ohlcv["close"], ohlcv["amount"]

    # ── 数据质量过滤：剔除价格覆盖不足的股票 ──
    # 早期（2018年）未上市或数据缺失的股票会在回测中产生 NaN/inf，
    # 这里预过滤：要求每只股票在 close 面板中至少有 60% 的有效价格
    coverage = close.notna().mean()
    good_stocks = coverage[coverage >= 0.6].index.tolist()
    bad_stocks = [s for s in close.columns if s not in good_stocks]
    if bad_stocks:
        print(f"\n    过滤掉 {len(bad_stocks)} 只数据不足的股票（覆盖<60%）")
        close = close[good_stocks]
        amount = amount[good_stocks]

    val = build_value_panels(pool, align_to=close)
    ind = industry_series(list(close.columns))
    print(f"{len(close)}天 × {len(close.columns)}只, {close.notna().sum().sum():,}有效价格点")

    # 因子库（全样本面板，但方向/权重由每窗 train 决定）
    lib = build_v9_factor_library(close, amount, val, ind)
    print(f"  因子: {list(lib.keys())}")
    fwd_ret = forward_returns(close, horizon=horizon)

    n = len(close)
    dates = close.index
    oos = {"equal": [], "ic": [], "train_pick": []}
    oos_bench = []
    windows = []

    # 找第一个所有因子都有效的日期（跳过 warmup 期）
    # low_vol_60 需要 61 天，加上估值数据可能有早期缺口
    lib_sample = pd.DataFrame({k: v.notna().any(axis=1) for k, v in lib.items()})
    all_valid = lib_sample.all(axis=1)
    first_valid_idx = all_valid.idxmax() if all_valid.any() else dates[0]
    min_start_idx = dates.get_loc(first_valid_idx)
    # 额外加 20 天缓冲，保证 train 窗口有足够有效数据
    min_start_idx = max(min_start_idx, 120)
    print(f"  因子warmup后首个全有效日: {first_valid_idx.date()}, 实际起始偏移: {min_start_idx}")

    start = min_start_idx
    while start + train_days + test_days <= n:
        cut = dates[start + train_days]
        test_end = start + train_days + test_days
        test_slice = slice(start + train_days, test_end)
        test_dates = dates[test_slice]

        # train 段定向
        signs = _train_signs(lib, fwd_ret, cut, horizon)

        # 三种合成
        comp_equal = _equal_composite(lib, signs)
        comp_ic, _ = ic_weighted_composite(lib, fwd_ret, cut, horizon=horizon)

        # train 择优
        train_dates_slice = dates[start : start + train_days]
        pick = _train_pick(
            close.loc[train_dates_slice], comp_equal, comp_ic,
            rebalance_days=rebalance_days, cost_rate=cost_rate,
        )

        comp_map = {
            "equal": comp_equal,
            "ic": comp_ic,
            "train_pick": comp_equal if pick == "equal" else comp_ic,
        }

        # test 段回测
        close_test = close.loc[test_dates]
        for m, comp in comp_map.items():
            bt = long_top_layer(
                close_test,
                comp.loc[test_dates],
                n_layers=5,
                rebalance_every=rebalance_days,
                cost_rate=cost_rate,
                first_rebalance=True,
            )
            oos[m].append(bt["port_ret"])
        oos_bench.append(bt["benchmark_ret"])
        windows.append((dates[start].date(), cut.date(), test_dates[-1].date(), pick))

        start += step_days

    elapsed = time.time() - t0
    print(f"  窗口数: {len(windows)}, 耗时: {elapsed:.0f}s")

    # 拼接 OOS
    bench_ret = pd.concat(oos_bench).sort_index()
    bench_ret = bench_ret[~bench_ret.index.duplicated(keep="first")]
    bench_bad = np.isinf(bench_ret).sum() + bench_ret.isna().sum()
    bench_ret = bench_ret.replace([np.inf, -np.inf], np.nan).dropna()
    bench_eq = (1 + bench_ret).cumprod()
    bs = summary(bench_eq, bench_ret)

    rows = []
    for m in ["equal", "ic", "train_pick"]:
        r = pd.concat(oos[m]).sort_index()
        r = r[~r.index.duplicated(keep="first")]
        n_raw = len(r)
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        n_filtered = n_raw - len(r)
        eq = (1 + r).cumprod()
        s = summary(eq, r)
        rows.append({
            "method": m,
            "total_return": s["total_return"],
            "annualized": s["annualized_return"],
            "sharpe": s["sharpe"],
            "max_dd": s["max_drawdown"],
            "excess_vs_bench": s["total_return"] - bs["total_return"],
            "n_filtered": n_filtered,
        })

    pick_counts = pd.Series([w[3] for w in windows]).value_counts().to_dict()

    return {
        "pool_name": pool_name,
        "n_stocks": len(pool),
        "n_windows": len(windows),
        "metrics": pd.DataFrame(rows),
        "benchmark": bs,
        "pick_counts": pick_counts,
        "oos": oos,
        "oos_bench": oos_bench,
    }


# ───────────────────────── 主流程 ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="CN800 Walk-Forward 验证")
    p.add_argument("--train", type=int, default=480)
    p.add_argument("--test", type=int, default=120)
    p.add_argument("--step", type=int, default=120)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--rebalance", type=int, default=40)
    p.add_argument("--cost", type=float, default=0.001,
                   help="单边交易成本（费率+滑点），默认0.1%")
    p.add_argument("--new-only", action="store_true",
                   help="只跑 CN800，跳过旧池对比")
    args = p.parse_args()

    cost_rate = args.cost

    print("=" * 70)
    print("  CN800 扩展池 Walk-Forward 验证")
    print(f"  train={args.train} test={args.test} step={args.step}")
    print(f"  horizon={args.horizon} rebalance={args.rebalance} cost={cost_rate:.3%}")
    print(f"  因子: value_blend + growth_peg + amihud + quality_roe(0.5) + low_vol_60(0.5)")
    print("=" * 70)

    # ── 跑 CN800 ──
    result_new = run_walkforward(
        CN800_POOL, "CN800（沪深300+中证500）",
        train_days=args.train, test_days=args.test, step_days=args.step,
        horizon=args.horizon, rebalance_days=args.rebalance, cost_rate=cost_rate,
    )

    # ── 跑旧池对比 ──
    if not args.new_only:
        result_old = run_walkforward(
            DEFAULT_POOL, "旧池（DEFAULT_POOL）",
            train_days=args.train, test_days=args.test, step_days=args.step,
            horizon=args.horizon, rebalance_days=args.rebalance, cost_rate=cost_rate,
        )
    else:
        result_old = None

    # ── 汇总对比 ──
    print("\n" + "=" * 70)
    print("  对比汇总：等权合成（equal）OOS 绩效")
    print("=" * 70)
    cols = ["total_return", "annualized", "sharpe", "max_dd", "excess_vs_bench"]

    compare_rows = []
    for r in [result_new] + ([result_old] if result_old else []):
        eq_row = r["metrics"][r["metrics"]["method"] == "equal"].iloc[0]
        compare_rows.append({
            "pool": r["pool_name"],
            "n_stocks": r["n_stocks"],
            "n_windows": r["n_windows"],
            **{c: eq_row[c] for c in cols},
        })
    df_compare = pd.DataFrame(compare_rows)
    print(df_compare.to_string(index=False))

    # 详细对比
    if result_old:
        print(f"\n  等权合成对比:")
        eq_new = result_new["metrics"][result_new["metrics"]["method"] == "equal"].iloc[0]
        eq_old = result_old["metrics"][result_old["metrics"]["method"] == "equal"].iloc[0]
        for c in cols:
            delta = eq_new[c] - eq_old[c]
            direction = "↑" if delta > 0 else "↓"
            if c in ("max_dd",):
                direction = "↓" if delta < 0 else "↑"  # 回撤降低是好事
            print(f"    {c}: {eq_old[c]:+.4f} → {eq_new[c]:+.4f}  ({delta:+.4f}) {direction}")

    # 完整指标表
    for r in [result_new] + ([result_old] if result_old else []):
        print(f"\n{'─'*70}")
        print(f"  {r['pool_name']} 完整指标 ({r['n_stocks']}只, {r['n_windows']}窗口)")
        print(f"{'─'*70}")
        print(r["metrics"].to_string(index=False))
        print(f"  基准: 收益{r['benchmark']['total_return']:+.2%} "
              f"年化{r['benchmark']['annualized_return']:+.2%} "
              f"夏普{r['benchmark']['sharpe']:.3f} "
              f"回撤{r['benchmark']['max_drawdown']:.2%}")
        print(f"  train_pick 选择分布: {r['pick_counts']}")

    # ── IC 统计补充 ──
    print(f"\n{'─'*70}")
    print(f"  CN800 全样本因子 IC 统计（仅参考，非 walk-forward 定向依据）")
    print(f"{'─'*70}")
    ohlcv = build_ohlcv_panels(CN800_POOL)
    close, amount = ohlcv["close"], ohlcv["amount"]
    val = build_value_panels(CN800_POOL, align_to=close)
    ind = industry_series(list(close.columns))
    lib = build_v9_factor_library(close, amount, val, ind)
    fwd = forward_returns(close, horizon=args.horizon)

    ic_rows = []
    for name, fac in lib.items():
        s = ic_summary(daily_ic(fac, fwd, min_count=5))
        ic_rows.append({
            "factor": name,
            "mean_ic": s["mean_ic"],
            "ic_std": s["ic_std"],
            "icir": s["ic_ir"],
            "ic_t": s["ic_t"],
            "pos_rate": s["pos_rate"],
        })
    df_ic = pd.DataFrame(ic_rows)
    print(df_ic.to_string(index=False))

    # ── 保存结果 ──
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    result_new["metrics"].to_csv(out_dir / "cn800_wf_metrics.csv", index=False)
    if result_old:
        result_old["metrics"].to_csv(out_dir / "oldpool_wf_metrics.csv", index=False)
    df_compare.to_csv(out_dir / "cn800_vs_oldpool_comparison.csv", index=False)
    df_ic.to_csv(out_dir / "cn800_factor_ic.csv", index=False)

    print(f"\n结果已保存至: {out_dir}/")
    print(f"  cn800_wf_metrics.csv")
    print(f"  cn800_factor_ic.csv")
    if result_old:
        print(f"  oldpool_wf_metrics.csv")
        print(f"  cn800_vs_oldpool_comparison.csv")


if __name__ == "__main__":
    main()
