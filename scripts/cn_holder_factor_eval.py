#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cn_holder_factor_eval —— 筹码集中度因子的诚实增量验证（M21）。

目的（决定是否把 holder_concentration 纳入 v3→v4 因子集）：
   1. 方向稳定性：全样本 IC 符号 + 滚动窗口 posRate（方向是否一贯）；
   2. 正交性：与现有 5 因子的横截面相关（目标 |corr|<0.7，否则信息冗余）；
   3. 增量价值：诚实 walk-forward 下「5因子 vs 6因子(加筹码)」的 OOS 收益/夏普/回撤对比，
      重点看**回撤是否被压低**（筹码因子的设计初衷是规避价值陷阱）。

严格遵守 walk-forward 纪律：方向只用 train 段 IC，test 段零泄漏（复用 honest 脚本逻辑）。

用法:
    NO_PROXY='*' PYTHONPATH=. python scripts/cn_holder_factor_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.data.universe import DEFAULT_POOL
from quant.data.panel import (
    build_ohlcv_panels, build_value_panels,
    build_cn_quarterly_panels, build_cn_holder_panels,
)
from quant.data.industry import industry_series
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer, simple_tradable_mask
from quant.backtest.metrics import summary
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize
from quant.strategy import cn_factor_spec as SPEC


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


def _equal_composite(lib, signs):
    oriented = [(lib[n] if signs[n] >= 0 else -lib[n]) for n in lib]
    return combine_factors(*oriented)


def main():
    print("=" * 78)
    print("  筹码集中度因子 诚实增量验证（IC稳定性 / 正交性 / 回撤改善）")
    print("=" * 78)

    pool = DEFAULT_POOL
    ohlcv = build_ohlcv_panels(pool)
    close, amount = ohlcv["close"], ohlcv["amount"]
    val = build_value_panels(pool, align_to=close)
    q = build_cn_quarterly_panels(pool, align_to=close)
    hp = build_cn_holder_panels(pool, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(val["total_mv"].replace(0, np.nan))

    # 轻量执行约束掩码（M22：停牌/涨跌停，调仓日不可交易则排除）
    tradable = simple_tradable_mask(ohlcv["open"], ohlcv["high"], ohlcv["low"], close, ohlcv["volume"])
    tradable_ratio = tradable.mean().mean()
    print(f"\n【执行约束统计】每日平均可交易: {tradable_ratio:.1%} (pool={len(pool)}只)")

    # 5 因子库（SSOT）+ 筹码因子（同样中性化）
    lib5 = SPEC.build_factor_library(close, amount, val, q, ind)
    holder = neutralize(
        F.holder_concentration(hp["change_ratio"]),
        industry=ind, log_mv=log_mv, mode="full",
    )
    lib6 = {**lib5, "holder_concentration": holder}

    fwd_ret = forward_returns(close, horizon=20)

    # ── 1. 方向稳定性：全样本 IC + 滚动 posRate ──
    print("\n【1. 筹码因子方向稳定性】")
    ic_full = daily_ic(holder, fwd_ret, min_count=5)
    s = ic_summary(ic_full)
    pos_rate = (ic_full > 0).mean()
    print(f"  全样本 meanIC={s['mean_ic']:+.4f}  t={s['t_stat']:+.2f}  posRate(IC>0)={pos_rate:.2%}")

    # ── 2. 正交性：与 5 因子的横截面相关（多日平均） ──
    print("\n【2. 与现有5因子的横截面相关（绝对值，目标<0.7）】")
    sample_dates = close.index[::120]  # 每120日采一个截面
    for name in SPEC.V3_FACTORS:
        corrs = []
        for d in sample_dates:
            a, b = holder.loc[d], lib5[name].loc[d]
            common = a.dropna().index.intersection(b.dropna().index)
            if len(common) >= 15:
                corrs.append(a[common].corr(b[common], method="spearman"))
        mc = np.nanmean(corrs) if corrs else np.nan
        flag = "✓正交" if abs(mc) < 0.7 else "✗冗余"
        print(f"  vs {name:16s} 平均相关={mc:+.3f}  {flag}")

    # ── 3. 增量价值：诚实 walk-forward 5因子 vs 6因子，有约束 vs 无约束 ──
    print("\n【3. 诚实 Walk-Forward：因子与执行约束的影响（train480/test120）】")
    train_days, test_days, step_days, horizon = 480, 120, 120, 20
    n = len(close); dates = close.index
    oos = {
        "v3 5因子(无约束)": [],
        "v4 6因子(加筹码无约束)": [],
        "v3 5因子(加约束)": [],
        "v4 6因子(加筹码加约束)": [],
    }
    oos_bench = []

    start = 0
    while start + train_days + test_days <= n:
        cut = dates[start + train_days]
        test_end = start + train_days + test_days
        test_dates = dates[start + train_days:test_end]

        for label, lib, do_mask in [
            ("v3 5因子(无约束)", lib5, False),
            ("v4 6因子(加筹码无约束)", lib6, False),
            ("v3 5因子(加约束)", lib5, True),
            ("v4 6因子(加筹码加约束)", lib6, True),
        ]:
            signs = _train_signs(lib, fwd_ret, cut, horizon)
            comp = _equal_composite(lib, signs)
            mask = tradable.loc[test_dates] if do_mask else None
            bt = long_top_layer(
                close.loc[test_dates], comp.loc[test_dates],
                n_layers=5, rebalance_every=SPEC.REBALANCE_DAYS,
                cost_rate=SPEC.EFFECTIVE_COMMISSION_RATE + SPEC.STAMP_DUTY_RATE / 2 + SPEC.SLIPPAGE,
                first_rebalance=True,
                tradable_mask=mask,
            )
            oos[label].append(bt["port_ret"])
        oos_bench.append(bt["benchmark_ret"])
        start += step_days

    bench_ret = pd.concat(oos_bench).sort_index()
    bench_ret = bench_ret[~bench_ret.index.duplicated(keep="first")]
    bs = summary((1 + bench_ret).cumprod(), bench_ret)

    rows = []
    for label in ["v3 5因子(无约束)", "v4 6因子(加筹码无约束)", "v3 5因子(加约束)", "v4 6因子(加筹码加约束)"]:
        r = pd.concat(oos[label]).sort_index()
        r = r[~r.index.duplicated(keep="first")]
        m = summary((1 + r).cumprod(), r)
        rows.append({
            "组合": label,
            "总收益": f"{m['total_return']:+.2%}",
            "年化": f"{m['annualized_return']:+.2%}",
            "夏普": f"{m['sharpe']:.3f}",
            "最大回撤": f"{m['max_drawdown']:.2%}",
            "超额": f"{m['total_return']-bs['total_return']:+.2%}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n  等权基准: 收益{bs['total_return']:+.2%} 夏普{bs['sharpe']:.3f} 回撤{bs['max_drawdown']:.2%}")

    out = PROJECT_ROOT / "results" / "cn_holder_factor_eval.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
