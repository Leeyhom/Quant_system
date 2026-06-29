"""strategy_evolution —— 策略进化对比（Stage 2c 进化版）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/strategy_evolution.py

对比维度（同一套因子+数据，只变组合构造方法）：
    ① 基线：等权分层 L5（top 20% 等权）
    ② Score-Weighted：因子分最高的 20% 股票，按 z-score 配权重
    ③ Score-Weighted + Vol Target：在②基础上叠加 15% 年化波动率缩放
    ④ 扩展因子等权：加入 value_ey（4因子等权）
    ⑤ 扩展因子 Score-Weighted + Vol Target：4因子 + 分数加权 + 波动率缩放

输出：综合对比表 + 净值曲线对比图
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant.config import RAW_DATA_DIR
from quant.data import us_loader
from quant.data.universe_us_expanded import EXPANDED_US_POOL
from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
from quant.data.industry_us import industry_series
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize as neut_fn
from quant.backtest.layered import long_top_layer, score_weighted_portfolio, vol_targeted
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary

REBALANCE = 20
US_COST_FN = make_layered_cost_fn()


def _summarize(df: pd.DataFrame) -> dict:
    m = summary(df["equity"], df["port_ret"])
    return {
        "total_return": m["total_return"],
        "annual_return": m["annualized_return"],
        "sharpe": m["sharpe"],
        "max_dd": m["max_drawdown"],
        "calmar": m["annualized_return"] / max(m["max_drawdown"], 0.001),
    }


def _print_row(label, s):
    print(f"  {label:40s} {s['total_return']:>+8.1%} {s['annual_return']:>+8.1%} "
          f"{s['sharpe']:>+6.2f} {s['max_dd']:>+7.1%} {s['calmar']:>+6.2f}")


def main():
    print("=" * 80)
    print("  策略进化对比（同一因子库，不同组合构造方法）")
    print("=" * 80)

    # —— 数据准备 ——
    print("\n[1] 构建面板...", flush=True)
    panels = build_ohlcv_panels(EXPANDED_US_POOL, loader=us_loader)
    close = panels["close"]
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    ind = industry_series(list(close.columns))
    fwd = forward_returns(close, horizon=20)

    # 因子：3 个稳定 + value_ey（虽然 ICIR 低但 posRate 51% 接近阈值，分数加权下可用）
    raw_3f = {
        "quality_roe": F.us_quality_roe(fund["roe"]),
        "quality_gm": F.us_quality_roe(fund["gross_margin"]),
        "growth_rev": F.us_growth(fund["rev_yoy"]),
    }
    raw_4f = {
        **raw_3f,
        "value_ey": F.us_earnings_yield(fund["eps_ttm"], close),
    }

    # 行业中性化
    facs_3f = {n: neut_fn(f, industry=ind, mode="industry") for n, f in raw_3f.items()}
    facs_4f = {n: neut_fn(f, industry=ind, mode="industry") for n, f in raw_4f.items()}

    eq_3f = combine_factors(*facs_3f.values())
    eq_4f = combine_factors(*facs_4f.values())

    # —— 各方案运行 ——
    print("\n[2] 运行 5 种组合方案...\n", flush=True)

    results = {}

    # ① 基线：等权分层 L5（3因子）
    bt = long_top_layer(close, eq_3f, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    results["① 等权分层L5 (3因子)"] = _summarize(bt)

    # ② Score-Weighted（3因子）
    bt = score_weighted_portfolio(close, eq_3f, top_frac=0.20, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    results["② Score-Weighted (3因子)"] = _summarize(bt)

    # ③ Score-Weighted + Vol Target（3因子）
    bt_vt = vol_targeted(bt.copy(), target_vol=0.15)
    results["③ Score-Weighted + VolTarget (3因)"] = _summarize(bt_vt)

    # ④ 扩展因子等权分层（4因子）
    bt = long_top_layer(close, eq_4f, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    results["④ 等权分层L5 (4因子含value)"] = _summarize(bt)

    # ⑤ 扩展因子 Score-Weighted + Vol Target（4因子）
    bt = score_weighted_portfolio(close, eq_4f, top_frac=0.20, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt = vol_targeted(bt, target_vol=0.15)
    results["⑤ Score-Weighted+VolTarget (4因)"] = _summarize(bt)

    # —— 基准 ——
    ret = close.pct_change().fillna(0.0)
    available = close.notna().astype(float)
    bench_w = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    bench_ret = (bench_w * ret).sum(axis=1)
    bench_eq = (1.0 + bench_ret).cumprod()
    bench_s = summary(bench_eq, bench_ret)
    bench = {"total_return": bench_s["total_return"], "annual_return": bench_s["annualized_return"],
             "sharpe": bench_s["sharpe"], "max_dd": bench_s["max_drawdown"],
             "calmar": bench_s["annualized_return"] / max(bench_s["max_drawdown"], 0.001)}

    # —— 打印对比表 ——
    print(f"  {'方案':40s} {'累计收益':>8s} {'年化收益':>8s} {'夏普':>6s} {'回撤':>7s} {'Calmar':>6s}")
    print(f"  {'-'*80}")
    for name, s in results.items():
        _print_row(name, s)
    _print_row("📊 等权全持有基准", bench)
    print()

    # 超额收益
    print(f"  {'方案':40s} {'超额收益(pct)':>14s} {'超额夏普':>10s}")
    print(f"  {'-'*70}")
    for name, s in results.items():
        excess = s["total_return"] - bench["total_return"]
        exc_sharpe = s["sharpe"] - bench["sharpe"]
        print(f"  {name:40s} {excess:>+14.0%} {exc_sharpe:>+10.2f}")

    # —— 净值曲线对比 ——
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ["steelblue", "darkorange", "green", "purple", "red"]
    for (name, s), c in zip(results.items(), colors):
        bt = None
        if "Score-Weighted+VolTarget (4因)" in name:
            bt_tmp = score_weighted_portfolio(close, eq_4f, top_frac=0.20, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
            bt = vol_targeted(bt_tmp, target_vol=0.15)
        elif "Score-Weighted + VolTarget (3因)" in name:
            bt_tmp = score_weighted_portfolio(close, eq_3f, top_frac=0.20, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
            bt = vol_targeted(bt_tmp, target_vol=0.15)
        elif "Score-Weighted (3因子)" in name:
            bt = score_weighted_portfolio(close, eq_3f, top_frac=0.20, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
        elif "等权分层L5 (4因子" in name:
            bt = long_top_layer(close, eq_4f, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
        else:
            bt = long_top_layer(close, eq_3f, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
        ax.plot(bt["equity"].index, bt["equity"], label=name, lw=1.8, color=c)

    ax.plot(bench_eq.index, bench_eq, label="Benchmark", lw=1.0, ls="--", color="gray")
    ax.set_title("Strategy Evolution: Portfolio Net Value Comparison (2018-2025)")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = RAW_DATA_DIR / "strategy_evolution.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n  净值曲线: {png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
