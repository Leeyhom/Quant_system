"""us_longshort_demo —— 美股多空对冲验证（Step 4，Stage 2c 后半）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/us_longshort_demo.py              # 全扩展池
    NO_PROXY='*' python scripts/us_longshort_demo.py --limit 200  # 小步

目的（对症 M17「方向先验漂移」，验证多空能否对冲掉方向漂移）：
    M17 的核心诊断：方向先验随池子构成变化（小池价值主导，全池成长接力），
    select_stable_positive 全池只筛出 1~3 个稳定因子。

    多空对冲的逻辑：不赌「哪个方向这段有效」，赌「因子排序能力本身是否真实」。
    只要 L5 长期跑赢 L1（IC>0），做多 L5 + 做空 L1 就能提取纯 alpha——
    市场 beta、行业 beta、方向漂移全被对冲掉。

    本脚本跑四件事：
    ① 全样本 L5/L1 单调性检验（分层收益是否单调递增？）
    ② 单因子多空滚动 walk-forward（每个因子独立多空）
    ③ 等权合成多空（选中的稳定因子等权→再分层层→多空 L5-L1）
    ④ 与多头-only 对比：多空是否提升跑赢率/降低回撤？

与 us_expanded_demo 的差异：
    - 组合从 long_top_layer → long_short_portfolio
    - 基准从「等权全持有」→ 保留但仅作参考（多空不应跟多头基准比）
    - 额外输出多空净值的风险指标
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from quant.config import RAW_DATA_DIR
from quant.data import us_loader
from quant.data.universe_us_expanded import EXPANDED_US_POOL
from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import layered_backtest, long_short_portfolio, long_top_layer, layer_summary
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
TRAIN_SIZE = 480
TEST_SIZE = 120
STEP = 60

# 借券费率：年化 2%（覆盖多数美股，偏保守）
BORROW_RATE = 0.02

US_COST_FN = make_layered_cost_fn()


def parse_limit(argv: list[str]) -> int | None:
    for i, a in enumerate(argv):
        if a == "--limit" and i + 1 < len(argv):
            return int(argv[i + 1])
    return None


def select_stable_positive(facs: dict, fwd, min_posrate: float = 0.52, min_icir: float = 0.06) -> list[str]:
    """筛方向先验稳定净正向的因子（与 M14/M17 同口径）。"""
    kept = []
    for name, fac in facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        if s["positive_rate"] >= min_posrate and s["icir"] >= min_icir:
            kept.append(name)
    return kept


def build_factors(close: pd.DataFrame, fund: dict) -> dict:
    """复用扩展因子库（7 个，与 us_expanded_demo 一致）。"""
    out = {
        "value_ey": F.us_earnings_yield(fund["eps_ttm"], close),
        "quality_roe": F.us_quality_roe(fund["roe"]),
        "quality_gm": F.us_quality_roe(fund["gross_margin"]),
        "growth_rev": F.us_growth(fund["rev_yoy"]),
        "growth_profit": F.us_growth(fund["profit_yoy"]),
    }
    if "net_margin" in fund:
        out["quality_nm"] = F.us_quality_roe(fund["net_margin"])
    if "debt_ratio" in fund:
        out["safety_debt"] = -fund["debt_ratio"]
    return out


def _roll_single(close, fac, label, mode="long_short"):
    """单因子滚动验证（long-only 或 long-short）。"""
    periods = []
    start = 0
    n = len(close)
    while start + TRAIN_SIZE + TEST_SIZE <= n:
        train_end = start + TRAIN_SIZE
        test_end = train_end + TEST_SIZE
        test = close.iloc[train_end - 1:test_end]

        if mode == "long_short":
            bt = long_short_portfolio(
                test, fac.reindex_like(test),
                n_layers=N_LAYERS, rebalance_every=REBALANCE,
                first_rebalance=True, cost_fn=US_COST_FN,
                borrow_rate=BORROW_RATE,
            ).iloc[1:]
        else:
            bt = long_top_layer(
                test, fac.reindex_like(test),
                n_layers=N_LAYERS, rebalance_every=REBALANCE,
                first_rebalance=True, cost_fn=US_COST_FN,
            ).iloc[1:]

        bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
        bt["benchmark"] = (1.0 + bt["benchmark_ret"]).cumprod()
        m = summary(bt["equity"], bt["port_ret"])
        bm = summary(bt["benchmark"], bt["benchmark_ret"])
        periods.append({
            "test_start": bt.index[0], "test_end": bt.index[-1],
            "sharpe": m["sharpe"], "bench_sharpe": bm["sharpe"],
            "beat": m["sharpe"] > bm["sharpe"],
        })
        start += STEP

    if not periods:
        return {"periods": [], "n": 0, "beat_rate": float("nan"), "median_sharpe": float("nan"),
                "median_excess": float("nan")}
    sh = pd.Series([p["sharpe"] for p in periods])
    exc = pd.Series([p["sharpe"] - p["bench_sharpe"] for p in periods])
    print(f"  {label:32s} 窗口{len(periods):2d} 跑赢{float(pd.Series([p['beat'] for p in periods]).mean()):.0%} "
          f"中位夏普{sh.median():+.2f} 中位超额{exc.median():+.2f}", flush=True)
    return {"periods": periods, "n": len(periods),
            "beat_rate": float(pd.Series([p["beat"] for p in periods]).mean()),
            "median_sharpe": float(sh.median()), "median_excess": float(exc.median())}


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = EXPANDED_US_POOL[:limit] if limit else EXPANDED_US_POOL
    limit_str = f"{len(symbols)} 只" if limit else "全池"

    print(f"[1/4] 构建面板（{limit_str}）...", flush=True)
    panels = build_ohlcv_panels(symbols, loader=us_loader)
    close = panels["close"]
    n_stocks = close.notna().any().sum()
    span = f"{close.index.min().date()}~{close.index.max().date()}"
    print(f"      行情面板 {close.shape[0]}天×{close.shape[1]}只（有数据 {n_stocks} 只）| {span}", flush=True)

    print(f"\n[2/4] 构建季报基本面面板...", flush=True)
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_factors(close, fund)

    # —— 全样本分层单调性 ——
    print(f"\n[3/4] 全样本分层单调性检验（L1→L5 收益是否递增？含美股费用）", flush=True)
    for name, fac in facs.items():
        res = layered_backtest(close, fac, n_layers=N_LAYERS, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
        lsum = layer_summary(res, n_layers=N_LAYERS)
        lrets = [lsum[lsum["layer"] == f"L{i}"]["total_return"].iloc[0] for i in range(1, N_LAYERS + 1)]
        mono = all(lrets[i] <= lrets[i + 1] for i in range(len(lrets) - 1))
        print(f"  {name:14s} L1→L5: {[f'{r:+.1%}' for r in lrets]}  {'单调✅' if mono else '不单调❌'}", flush=True)

    # —— 稳定因子筛选 ——
    selected = select_stable_positive(facs, fwd)
    print(f"\n      等权分量筛选: {selected}（共 {len(selected)} 个）", flush=True)
    if not selected:
        print("  无稳定因子，终止。")
        return
    sel_facs = {n: facs[n] for n in selected}

    # —— 多空滚动验证 ——
    print(f"\n[4/4] 多空对冲 滚动 walk-forward（train{TRAIN_SIZE}/test{TEST_SIZE}/step{STEP}）", flush=True)
    print("      —— 做多 L5 + 做空 L1，借券费 2%/年 ——", flush=True)

    print("\n      ▸ 单因子多空：", flush=True)
    ls_single = {}
    for name in facs:
        ls_single[name] = _roll_single(close, facs[name], f"多空 {name}", mode="long_short")

    # 等权合成多空
    eq_factor = combine_factors(*sel_facs.values())
    print(f"      {'-'*60}", flush=True)
    ls_eq = _roll_single(close, eq_factor, f"等权多空 ({len(selected)}因子)", mode="long_short")

    # —— 对比多头-only ——
    print(f"\n      ▸ 对照：多头-only（用于对比多空是否提升）：", flush=True)
    long_eq = _roll_single(close, eq_factor, f"等权多头 ({len(selected)}因子)", mode="long_only")

    # —— 全样本多空净值 ——
    print(f"\n      —— 全样本多空净值 ——", flush=True)
    bt_ls = long_short_portfolio(
        close, eq_factor,
        n_layers=N_LAYERS, rebalance_every=REBALANCE,
        cost_fn=US_COST_FN, borrow_rate=BORROW_RATE,
    )
    bt_long = long_top_layer(
        close, eq_factor,
        n_layers=N_LAYERS, rebalance_every=REBALANCE, cost_fn=US_COST_FN,
    )
    m_ls = summary(bt_ls["equity"], bt_ls["port_ret"])
    m_long = summary(bt_long["equity"], bt_long["port_ret"])
    m_bench = summary(bt_ls["benchmark"], bt_ls["benchmark_ret"])

    print(f"      等权多空 ({len(selected)}因子) 收益 {m_ls['total_return']:+.2%} | "
          f"夏普 {m_ls['sharpe']:+.2f} | 回撤 {m_ls['max_drawdown']:.2%}")
    print(f"      等权多头 ({len(selected)}因子) 收益 {m_long['total_return']:+.2%} | "
          f"夏普 {m_long['sharpe']:+.2f} | 回撤 {m_long['max_drawdown']:.2%}")
    print(f"      等权基准            收益 {m_bench['total_return']:+.2%} | "
          f"夏普 {m_bench['sharpe']:+.2f} | 回撤 {m_bench['max_drawdown']:.2%}")

    # 图表：多空 vs 多头 vs 基准
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(bt_ls["equity"].index, bt_ls["equity"], label=f"long-short ({len(selected)} factors)", linewidth=1.6)
    ax1.plot(bt_long["equity"].index, bt_long["equity"], label=f"long-only ({len(selected)} factors)", linewidth=1.3)
    ax1.plot(bt_ls["benchmark"].index, bt_ls["benchmark"], label="benchmark (equal-hold)", linewidth=0.8, linestyle="--")
    ax1.set_title(f"US Expanded Pool ({n_stocks} stocks, {span}): Long-Short vs Long-Only")
    ax1.set_ylabel("net value"); ax1.legend()

    # 滚动窗口跑赢率对比
    ax2.bar(["多头-only", "多空对冲"],
            [long_eq["beat_rate"], ls_eq["beat_rate"]],
            color=["steelblue", "darkorange"])
    ax2.axhline(y=0.5, color="gray", linestyle="--", label="50% 基准")
    ax2.set_title(f"Rolling Win Rate ({ls_eq['n']} windows): Long-Only vs Long-Short")
    ax2.set_ylabel("beat rate"); ax2.legend()
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    png = RAW_DATA_DIR / "us_longshort_equity.png"
    fig.savefig(png, dpi=120)
    print(f"      图: {png}")

    # —— 诚实结论 ——
    print(f"\n———— 诚实结论 ————", flush=True)
    print(f"  多空等权: 跑赢 {ls_eq['beat_rate']:.0%} | 中位超额夏普 {ls_eq['median_excess']:+.2f}")
    print(f"  多头等权: 跑赢 {long_eq['beat_rate']:.0%} | 中位超额夏普 {long_eq['median_excess']:+.2f}")
    if ls_eq["beat_rate"] > long_eq["beat_rate"]:
        print(f"  ✅ 多空对冲提升了跑赢率（{ls_eq['beat_rate']:.0%} > {long_eq['beat_rate']:.0%}）")
    else:
        print(f"  → 多空对冲未提升跑赢率。可能是因为：")
        print(f"    ① L1 空头腿的 alpha 不足（底部股票本身不差，只是相对不够好）")
        print(f"    ② 借券费侵蚀了多空利差（2%/年 ≈ {BORROW_RATE/252*1e4:.1f}bp/天）")
        print(f"    ③ 多空利差本身的波动大，夏普不一定优于纯多头")
    print(f"完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
