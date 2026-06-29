#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cn_walkforward_honest —— A股 v3 五因子「诚实」Walk-Forward（M20，去作弊）。

为什么写这个脚本（对症 docs/AUDIT 报告的 HIGH 级偏差）：
   此前本地宣传的 314%/夏普1.04 用了**全样本 IC 定向**——在 2018 建仓时就用到了
   2025 的全期因子方向，等于作弊；joinquant_v5_risk_filter_search 又从上百个参数
   组合里挑夏普最高的，叠加选择偏差。结果本地数字远高于聚宽实测，根本无法复现。

本脚本严格遵守 walk-forward 纪律，产出**可复现、可对齐聚宽**的诚实数字：
   ① 因子方向只用每个窗口 train 段的 IC 决定，test 段零信息泄漏；
   ② train IC 计算复用 composite.py 已验证的防重叠逻辑（裁掉末尾 horizon 天）；
   ③ 因子口径用 M20 对齐后的 SSOT（cn_factor_spec），成长因子=净利润同比/PE（聚宽口径）；
   ④ 同时比较三种合成方式，输出各自 OOS 表现，不事后择优：
        - equal:   等权 rank 平均（M14 结论：方向先验稳时等权胜）
        - ic:      train 段 |IC| 加权（composite.ic_weighted_composite）
        - train_pick: 在 train 段比较 equal vs ic，选 train 上更好的应用到 test
          （docs/AUDIT 建议的诚实做法：择优也只在 train 段做）

用法:
    NO_PROXY='*' PYTHONPATH=. python scripts/cn_walkforward_honest.py
    可选: --train 480 --test 120 --step 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels, build_cn_quarterly_panels
from quant.data.industry import industry_series
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.factor.composite import ic_weighted_composite
from quant.factor.factors import combine_factors
from quant.strategy import cn_factor_spec as SPEC


def _train_signs(lib: dict, fwd_ret: pd.DataFrame, upto_date, horizon: int = 20) -> dict:
    """只用 train 段（裁掉末尾 horizon 重叠）IC 给每个因子定方向符号。"""
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
    """按 train 段符号定向后等权 rank 平均。"""
    oriented = [(lib[name] if signs[name] >= 0 else -lib[name]) for name in lib]
    return combine_factors(*oriented)


def run(train_days=480, test_days=120, step_days=120, horizon=20, growth_mode="yoy"):
    print("=" * 78)
    print(f"  A股 v3 五因子 诚实 Walk-Forward（train段定向，test段评估，无全样本泄漏）[growth={growth_mode}]")
    print("=" * 78)

    pool = DEFAULT_POOL
    ohlcv = build_ohlcv_panels(pool)
    close, amount = ohlcv["close"], ohlcv["amount"]
    val = build_value_panels(pool, align_to=close)
    q = build_cn_quarterly_panels(pool, align_to=close)
    ind = industry_series(list(close.columns))

    lib = SPEC.build_factor_library(close, amount, val, q, ind, growth_mode=growth_mode)
    fwd_ret = forward_returns(close, horizon=horizon)

    n = len(close)
    dates = close.index
    # 收集每种方法的 OOS 日收益拼接
    oos = {m: [] for m in ["equal", "ic", "train_pick"]}
    oos_bench = []
    windows = []

    start = 0
    while start + train_days + test_days <= n:
        cut = dates[start + train_days]              # train/test 切分日
        test_end = start + train_days + test_days
        test_slice = slice(start + train_days, test_end)
        test_dates = dates[test_slice]

        # train 段定向（只用 cut 之前）
        signs = _train_signs(lib, fwd_ret, cut, horizon)

        # 三种合成因子（全样本面板，但方向/权重只由 train 决定）
        comp_equal = _equal_composite(lib, signs)
        comp_ic, _ = ic_weighted_composite(lib, fwd_ret, cut, horizon=horizon)

        # train 段择优：在 train 上分别回测 equal/ic，选更好的
        train_dates = dates[start:start + train_days]
        pick = _train_pick(close.loc[train_dates], comp_equal, comp_ic)

        comp_map = {"equal": comp_equal, "ic": comp_ic, "train_pick": comp_equal if pick == "equal" else comp_ic}

        # test 段回测每种方法
        close_test = close.loc[test_dates]
        for m, comp in comp_map.items():
            bt = long_top_layer(
                close_test, comp.loc[test_dates],
                n_layers=5, rebalance_every=SPEC.REBALANCE_DAYS,
                cost_rate=SPEC.EFFECTIVE_COMMISSION_RATE + SPEC.STAMP_DUTY_RATE / 2 + SPEC.SLIPPAGE,
                first_rebalance=True,
            )
            oos[m].append(bt["port_ret"])
        oos_bench.append(bt["benchmark_ret"])
        windows.append((dates[start].date(), cut.date(), test_dates[-1].date(), pick))

        start += step_days

    print(f"\n滚动窗口数: {len(windows)}  (train={train_days} test={test_days} step={step_days})")
    print(f"窗口示例: {windows[0]} ... {windows[-1]}")
    pick_counts = pd.Series([w[3] for w in windows]).value_counts().to_dict()
    print(f"train_pick 选择分布: {pick_counts}")

    # 拼接 OOS 日收益，计算整体绩效
    bench_ret = pd.concat(oos_bench).sort_index()
    bench_ret = bench_ret[~bench_ret.index.duplicated(keep="first")]
    bench_eq = (1 + bench_ret).cumprod()
    bs = summary(bench_eq, bench_ret)

    print("\n" + "=" * 78)
    print(f"  拼接 OOS 绩效（{bench_ret.index[0].date()} ~ {bench_ret.index[-1].date()}）")
    print("=" * 78)
    rows = []
    for m in ["equal", "ic", "train_pick"]:
        r = pd.concat(oos[m]).sort_index()
        r = r[~r.index.duplicated(keep="first")]
        eq = (1 + r).cumprod()
        s = summary(eq, r)
        rows.append({
            "method": m,
            "total_return": s["total_return"],
            "annualized": s["annualized_return"],
            "sharpe": s["sharpe"],
            "max_dd": s["max_drawdown"],
            "excess_vs_bench": s["total_return"] - bs["total_return"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n等权基准: 收益{bs['total_return']:+.2%} 年化{bs['annualized_return']:+.2%} 夏普{bs['sharpe']:.3f} 回撤{bs['max_drawdown']:.2%}")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "cn_walkforward_honest.csv", index=False)
    print(f"\n已保存: {out_dir / 'cn_walkforward_honest.csv'}")
    return df, bs


def _train_pick(close_train: pd.DataFrame, comp_equal, comp_ic) -> str:
    """在 train 段分别回测两种合成，返回 train 上夏普更高的方法名。"""
    best, best_sharpe = "equal", -np.inf
    for name, comp in [("equal", comp_equal), ("ic", comp_ic)]:
        try:
            bt = long_top_layer(
                close_train, comp.reindex_like(close_train),
                n_layers=5, rebalance_every=SPEC.REBALANCE_DAYS,
                cost_rate=SPEC.EFFECTIVE_COMMISSION_RATE + SPEC.STAMP_DUTY_RATE / 2 + SPEC.SLIPPAGE,
                first_rebalance=True,
            )
            s = summary(bt["equity"], bt["port_ret"])["sharpe"]
        except Exception:
            s = -np.inf
        if s > best_sharpe:
            best, best_sharpe = name, s
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=480)
    p.add_argument("--test", type=int, default=120)
    p.add_argument("--step", type=int, default=120)
    p.add_argument("--growth", choices=["yoy", "peg"], default="peg",
                   help="成长因子口径：peg=日频PEG倒数(SSOT生产默认,覆盖94%/OOS优)，yoy=季报同比/PE(覆盖68%,仅对比)")
    args = p.parse_args()
    run(args.train, args.test, args.step, growth_mode=args.growth)


if __name__ == "__main__":
    main()
