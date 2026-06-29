"""phase3a_pricevolume —— 补量价因子 + IC方向筛选（Phase 3a）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/phase3a_pricevolume.py

逻辑：
    ① 在扩展池(454只)上重新计算M16的6个量价因子IC。
    ② 用「IC方向稳定性」筛选：不要求posRate≥52%，只要求meanIC方向明确(|t|>2)。
       ——如果量价因子IC持续为负，就反向使用（orient_by_sign），不浪费信号。
    ③ 将方向稳定且与基本面低相关的量价因子纳入，与3个基本面因子合并。
    ④ 对比「纯基本面3因子」vs「基本面+量价N因子」的等权组合绩效。
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
from quant.factor.composite import factor_correlation
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer, score_weighted_portfolio
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
US_COST_FN = make_layered_cost_fn()

# IC方向筛选阈值：|t| > 2 即可（比 select_stable_positive 的 ICIR≥0.06 更宽松，
# 因为我们不要求IC为正——只要方向明确，负的就反向用）
DIR_T_THRESHOLD = 2.0


def build_pricevolume_factors(panels: dict) -> dict:
    """构建 6 个量价因子（与M16完全一致）。"""
    close, amount = panels["close"], panels["amount"]
    high, low = panels["high"], panels["low"]
    return {
        "pv_momentum60": F.momentum(close, 60),
        "pv_reversal20": F.reversal(close, 20),
        "pv_lowvol20": F.low_volatility(close, 20),
        "pv_amihud": F.amihud_illiquidity(close, amount, 20),
        "pv_maslope20": F.ma_slope(close, 20),
        "pv_parkinson": F.parkinson_volatility(high, low, 20),
    }


def orient_factor(factor: pd.DataFrame, mean_ic: float) -> pd.DataFrame:
    """如果全样本meanIC为负，翻转因子方向（使「越高越好」与IC方向一致）。"""
    if mean_ic < 0:
        return -factor
    return factor


def _perf_line(ret, label):
    eq = (1.0 + ret).cumprod()
    m = summary(eq, ret)
    return (f"{label:35s} {m['total_return']:>+8.1%} {m['sharpe']:>+6.2f} "
            f"{m['max_drawdown']:>+7.1%}")


def main():
    print("=" * 72)
    print("  Phase 3a: 量价因子 + IC方向筛选")
    print("=" * 72)

    # —— 数据准备 ——
    print("\n[1] 构建面板...", flush=True)
    panels = build_ohlcv_panels(EXPANDED_US_POOL, loader=us_loader)
    close = panels["close"]
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    ind = industry_series(list(close.columns))
    fwd = forward_returns(close, horizon=HORIZON)

    # —— 基本面因子（基线3因子） ——
    print("\n[2] 构建基本面因子（3因子基线）...", flush=True)
    raw_fund_facs = {
        "quality_roe": F.us_quality_roe(fund["roe"]),
        "quality_gm": F.us_quality_roe(fund["gross_margin"]),
        "growth_rev": F.us_growth(fund["rev_yoy"]),
    }
    fund_facs = {n: neut_fn(f, industry=ind, mode="industry") for n, f in raw_fund_facs.items()}
    eq_fund = combine_factors(*fund_facs.values())

    # —— 量价因子 + IC方向筛选 ——
    print("\n[3] 量价因子IC诊断 + 方向筛选...", flush=True)
    pv_raw = build_pricevolume_factors(panels)
    # 量价因子也做行业中性化（剥离行业beta后看纯alpha方向）
    pv_facs = {n: neut_fn(f, industry=ind, mode="industry") for n, f in pv_raw.items()}

    print(f"      {'Factor':20s} {'meanIC':>8s} {'t-stat':>7s} {'IC方向':>8s} {'判定':>6s}")
    print(f"      {'-'*55}")
    pv_selected = {}
    pv_rejected = []
    for name, fac in pv_facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        t = s["t_stat"]
        mean_ic = s["mean_ic"]
        direction = "正向" if mean_ic > 0 else "负向"
        if abs(t) > DIR_T_THRESHOLD:
            # 方向明确，纳入。若IC为负则翻转因子
            oriented = orient_factor(fac, mean_ic)
            pv_selected[name] = oriented
            action = "✅纳入" if mean_ic > 0 else "✅纳入(翻)"
        else:
            pv_rejected.append(name)
            action = "❌方向不稳"
        print(f"      {name:20s} {mean_ic:>+8.4f} {t:>+7.2f} {direction:>8s} {action:>10s}")

    n_pv = len(pv_selected)
    print(f"\n      量价因子: {n_pv} 个通过方向筛选, {len(pv_rejected)} 个被拒绝")
    if pv_rejected:
        print(f"      拒绝: {pv_rejected}")

    if n_pv == 0:
        print("\n  → 无方向稳定的量价因子，M16结论在扩展池仍成立。")
        return

    # —— 基本面 vs 量价相关性 ——
    print(f"\n[4] 量价因子与基本面因子的相关性...", flush=True)
    all_facs_check = {**fund_facs, **pv_selected}
    corr = factor_correlation(all_facs_check)
    # 只看基本面 vs 量价的交叉相关
    fund_names = list(fund_facs.keys())
    pv_names = list(pv_selected.keys())
    cross_corr = corr.loc[fund_names, pv_names]
    print(cross_corr.round(3).to_string())
    avg_cross = cross_corr.abs().mean().mean()
    print(f"      平均绝对交叉相关: {avg_cross:.3f}")

    # —— 合并因子集 ——
    print(f"\n[5] 构建扩展因子集（基本面{len(fund_facs)} + 量价{len(pv_selected)}）...", flush=True)
    all_facs = {**fund_facs, **pv_selected}
    eq_all = combine_factors(*all_facs.values())
    print(f"      总因子数: {len(all_facs)}")

    # —— 对比回测 ——
    print(f"\n[6] 对比回测: 纯基本面 vs 基本面+量价...\n", flush=True)

    bt_fund = long_top_layer(close, eq_fund, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt_all = long_top_layer(close, eq_all, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bench_ret = bt_fund["benchmark_ret"]

    print(f"      {'方案':35s} {'累计收益':>8s} {'夏普':>6s} {'回撤':>7s}")
    print(f"      {'-'*60}")
    print(f"      {_perf_line(bt_fund['port_ret'], '纯基本面3因子')}")
    print(f"      {_perf_line(bt_all['port_ret'], f'基本面+量价{len(all_facs)}因子')}")
    print(f"      {_perf_line(bench_ret, '等权基准')}")

    # —— 滚动窗口验证 ——
    print(f"\n[7] 滚动 walk-forward 验证...", flush=True)
    TRAIN_SIZE, TEST_SIZE, STEP = 480, 120, 60
    for label, eq_fac in [("纯基本面3因子", eq_fund), (f"基本面+量价{len(all_facs)}因子", eq_all)]:
        periods = []
        start = 0
        n = len(close)
        while start + TRAIN_SIZE + TEST_SIZE <= n:
            train_end = start + TRAIN_SIZE
            test_end = train_end + TEST_SIZE
            test = close.iloc[train_end - 1:test_end]
            test_factor = eq_fac.iloc[train_end - 1:test_end]
            bt = long_top_layer(test, test_factor, n_layers=N_LAYERS,
                                rebalance_every=REBALANCE, first_rebalance=True,
                                cost_fn=US_COST_FN).iloc[1:]
            bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
            bt["benchmark"] = (1.0 + bt["benchmark_ret"]).cumprod()
            m = summary(bt["equity"], bt["port_ret"])
            bm = summary(bt["benchmark"], bt["benchmark_ret"])
            periods.append({"beat": m["sharpe"] > bm["sharpe"], "excess": m["sharpe"] - bm["sharpe"]})
            start += STEP
        beat = sum(p["beat"] for p in periods) / len(periods)
        med_excess = np.median([p["excess"] for p in periods])
        print(f"      {label:35s} {len(periods)}窗口 | 跑赢{beat:.0%} | 中位超额夏普{med_excess:+.2f}")

    # —— 净值对比图 ——
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt_fund["equity"].index, bt_fund["equity"], label="Fundamental 3F", lw=2.0, color="steelblue")
    ax.plot(bt_all["equity"].index, bt_all["equity"], label=f"Fund+PV {len(all_facs)}F", lw=2.0, color="darkorange")
    ax.plot(bt_fund["benchmark"].index, bt_fund["benchmark"], label="Benchmark", lw=1.0, ls="--", color="gray")
    ax.set_title(f"Phase 3a: Adding Price-Volume Factors ({len(pv_selected)} selected via IC direction)")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = RAW_DATA_DIR / "phase3a_pv_factors.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n      图: {png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
