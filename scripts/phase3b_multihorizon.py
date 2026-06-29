"""phase3b_multihorizon —— 多周期因子 + 基本面变动因子（Phase 3b）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/phase3b_multihorizon.py

逻辑：
    ① 量价因子多周期：5d(短期反转/波动)、60d(长期动量/趋势)，与20d正交
    ② 基本面变动因子：ROE YoY变化、毛利率YoY变化、营收增长加速度
    ③ IC方向筛选后纳入，与Phase 3a的8因子合并
    ④ 对比 8因子 vs 全因子 的绩效
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
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
US_COST_FN = make_layered_cost_fn()
DIR_T_THRESHOLD = 2.0


def orient_factor(fac, mean_ic):
    return -fac if mean_ic < 0 else fac


def _perf_line(ret, label):
    eq = (1.0 + ret).cumprod()
    m = summary(eq, ret)
    return f"{label:35s} {m['total_return']:>+8.1%} {m['sharpe']:>+6.2f} {m['max_drawdown']:>+7.1%}"


def screen_and_orient(facs: dict, fwd) -> dict:
    """IC方向筛选 + 定向。返回 {name: oriented_factor}。"""
    selected = {}
    for name, fac in facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        if abs(s["t_stat"]) > DIR_T_THRESHOLD:
            selected[name] = orient_factor(fac, s["mean_ic"])
            direction = "翻" if s["mean_ic"] < 0 else "正"
            print(f"        {name:22s} IC{s['mean_ic']:+.4f} t{s['t_stat']:+.2f} ✅纳入({direction})")
        else:
            print(f"        {name:22s} IC{s['mean_ic']:+.4f} t{s['t_stat']:+.2f} ❌")
    return selected


def main():
    print("=" * 72)
    print("  Phase 3b: 多周期因子 + 基本面变动因子")
    print("=" * 72)

    # —— 数据准备 ——
    print("\n[1] 构建面板...", flush=True)
    panels = build_ohlcv_panels(EXPANDED_US_POOL, loader=us_loader)
    close = panels["close"]
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    ind = industry_series(list(close.columns))
    fwd = forward_returns(close, horizon=HORIZON)

    # —— 基线：Phase 3a 的 8 因子 ——
    print("\n[2] 构建 Phase 3a 基线（8因子）...", flush=True)
    fund_facs = {
        "quality_roe": neut_fn(F.us_quality_roe(fund["roe"]), industry=ind, mode="industry"),
        "quality_gm": neut_fn(F.us_quality_roe(fund["gross_margin"]), industry=ind, mode="industry"),
        "growth_rev": neut_fn(F.us_growth(fund["rev_yoy"]), industry=ind, mode="industry"),
    }
    # PV 5因子（从Phase 3a确认的，做方向定向）
    pv_5f = {
        "pv_momentum60": -F.momentum(close, 60),      # 翻：IC负
        "pv_reversal20": F.reversal(close, 20),        # 正
        "pv_lowvol20": -F.low_volatility(close, 20),   # 翻：IC负
        "pv_amihud": -F.amihud_illiquidity(close, panels["amount"], 20),  # 翻
        "pv_parkinson": -F.parkinson_volatility(panels["high"], panels["low"], 20),  # 翻
    }
    pv_5f = {n: neut_fn(f, industry=ind, mode="industry") for n, f in pv_5f.items()}
    baseline_8f = {**fund_facs, **pv_5f}
    eq_baseline = combine_factors(*baseline_8f.values())

    # —— 多周期量价因子 ——
    print("\n[3] 多周期量价因子 IC诊断...", flush=True)
    amount = panels["amount"]
    high, low = panels["high"], panels["low"]

    multi_pv_raw = {
        # 短期（5日）—— 捕捉快速反转/微观结构
        "pv_reversal5": F.reversal(close, 5),
        "pv_lowvol5": F.low_volatility(close, 5),
        "pv_amihud5": F.amihud_illiquidity(close, amount, 5),
        # 长期（60日）—— 捕捉趋势/持续动量
        "pv_momentum120": F.momentum(close, 120),
        "pv_maslope60": F.ma_slope(close, 60),
        "pv_lowvol60": F.low_volatility(close, 60),
    }
    multi_pv_raw = {n: neut_fn(f, industry=ind, mode="industry") for n, f in multi_pv_raw.items()}
    print("      ▸ 多周期量价因子:")
    multi_pv = screen_and_orient(multi_pv_raw, fwd)

    # —— 基本面变动因子 ——
    print("\n[4] 基本面变动因子（从已加载面板零成本导出）...", flush=True)
    # ROE YoY变化：4个季度前的ROE vs 现在
    roe_panel = fund["roe"]
    roe_4q_ago = roe_panel.shift(252)  # ~1年交易日 ≈ 4个季度
    roe_change = roe_panel - roe_4q_ago

    gm_panel = fund["gross_margin"]
    gm_change = gm_panel - gm_panel.shift(252)

    # 营收增长率加速度（growth_rev的二阶导）
    rev_yoy = fund["rev_yoy"]
    rev_yoy_4q_ago = rev_yoy.shift(252)
    rev_growth_accel = rev_yoy - rev_yoy_4q_ago

    change_facs_raw = {
        "roe_yoy_change": roe_change,
        "gm_yoy_change": gm_change,
        "rev_growth_accel": rev_growth_accel,
    }
    change_facs_raw = {n: neut_fn(f, industry=ind, mode="industry") for n, f in change_facs_raw.items()}
    print("      ▸ 基本面变动因子:")
    change_facs = screen_and_orient(change_facs_raw, fwd)

    # —— 合并所有因子 ——
    all_new = {**multi_pv, **change_facs}
    print(f"\n[5] 因子汇总: 基线8 + 新增{len(all_new)} = {8+len(all_new)}因子", flush=True)

    if len(all_new) == 0:
        print("  → 无新增因子通过方向筛选。Phase 3a的8因子即最优。")
        return

    all_facs = {**baseline_8f, **all_new}
    eq_all = combine_factors(*all_facs.values())

    # —— 相关性检查 ——
    print(f"\n      ▸ 新增因子与基线因子的平均绝对交叉相关:")
    new_names = list(all_new.keys())
    baseline_names = list(baseline_8f.keys())
    corr = factor_correlation(all_facs)
    cross = corr.loc[baseline_names, new_names].abs()
    print(f"        {cross.mean().mean():.3f}")

    # —— 对比回测 ——
    print(f"\n[6] 对比回测...\n", flush=True)
    bt_base = long_top_layer(close, eq_baseline, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt_all = long_top_layer(close, eq_all, rebalance_every=REBALANCE, cost_fn=US_COST_FN)

    print(f"      {'方案':35s} {'累计收益':>8s} {'夏普':>6s} {'回撤':>7s}")
    print(f"      {'-'*60}")
    print(f"      {_perf_line(bt_base['port_ret'], 'Phase 3a 8因子')}")
    print(f"      {_perf_line(bt_all['port_ret'], f'Phase 3b {len(all_facs)}因子')}")
    print(f"      {_perf_line(bt_base['benchmark_ret'], '等权基准')}")

    # —— 滚动验证 ——
    print(f"\n[7] 滚动 walk-forward...", flush=True)
    TRAIN, TEST, STEP = 480, 120, 60
    for label, eq_fac in [("Phase 3a 8因子", eq_baseline), (f"Phase 3b {len(all_facs)}因子", eq_all)]:
        periods = []
        start = 0
        n = len(close)
        while start + TRAIN + TEST <= n:
            te = start + TRAIN
            test_end = te + TEST
            test = close.iloc[te - 1:test_end]
            tf = eq_fac.iloc[te - 1:test_end]
            bt = long_top_layer(test, tf, n_layers=N_LAYERS, rebalance_every=REBALANCE,
                                first_rebalance=True, cost_fn=US_COST_FN).iloc[1:]
            bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
            bt["benchmark"] = (1.0 + bt["benchmark_ret"]).cumprod()
            m = summary(bt["equity"], bt["port_ret"])
            bm = summary(bt["benchmark"], bt["benchmark_ret"])
            periods.append({"beat": m["sharpe"] > bm["sharpe"], "excess": m["sharpe"] - bm["sharpe"]})
            start += STEP
        beat = sum(p["beat"] for p in periods) / max(len(periods), 1)
        med_excess = np.median([p["excess"] for p in periods])
        print(f"      {label:35s} {len(periods)}窗口 | 跑赢{beat:.0%} | 中位超额夏普{med_excess:+.2f}")

    # —— 图 ——
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt_base["equity"].index, bt_base["equity"], label="Phase 3a (8F)", lw=2.0, color="steelblue")
    ax.plot(bt_all["equity"].index, bt_all["equity"], label=f"Phase 3b ({len(all_facs)}F)", lw=2.0, color="darkorange")
    ax.plot(bt_base["benchmark"].index, bt_base["benchmark"], label="Benchmark", lw=1.0, ls="--", color="gray")
    ax.set_title(f"Phase 3b: Multi-Horizon + Fundamental Changes")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = RAW_DATA_DIR / "phase3b_multihorizon.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n      图: {png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
