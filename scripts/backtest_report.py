"""backtest_report —— 美股最优配置综合回测报告（Stage 2c 交付）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/backtest_report.py

最优配置（基于 Step 2-4 的实证选择）：
    池子:  扩展池 454只（S&P 500 + S&P 400 + 精选）
    因子:  quality_roe + quality_gm + growth_rev（3个稳定正向）
    中性化: 行业中性化（GICS 一级行业去均值）
    合成:  等权（固定正向先验）
    组合:  分层多头 L5（top 20%）
    再平衡: 20日
    费用:  美股每股费+每笔最低费

输出（全部落地到 data/raw/ 目录）：
    - report_equity.png：主净值曲线（等权 vs 最优单因子 vs 基准）
    - report_yearly_ic.png：逐年 IC 热力图
    - report_rolling_ic.png：滚动 IC 稳定性
    - report_drawdown.png：回撤分析
    - report_yearly_returns.png：逐年收益对比
    - report_corr_ts.png：因子相关性时变
    - report_summary.txt：文字总结
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
from quant.backtest.layered import long_top_layer, layered_backtest, layer_summary
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

# ─── 最优配置参数 ───
HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
TRAIN_SIZE = 480
TEST_SIZE = 120
STEP = 60
US_COST_FN = make_layered_cost_fn()

# 报告输出目录
REPORT_DIR = RAW_DATA_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    p = REPORT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  → {p}")
    plt.close(fig)


def _perf_line(ret: pd.Series, label: str) -> str:
    """把一条日收益序列总结成一行文字。"""
    eq = (1.0 + ret).cumprod()
    m = summary(eq, ret)
    calmar = m["annualized_return"] / m["max_drawdown"] if m["max_drawdown"] > 0 else 0
    return (f"{label:20s} 累计{m['total_return']:>+8.1%}  "
            f"年化{m['annualized_return']:>+6.1%}  夏普{m['sharpe']:>+6.2f}  "
            f"回撤{m['max_drawdown']:>+6.1%}  Calmar{calmar:>+5.2f}")


def main() -> None:
    print("=" * 72)
    print("  美股最优配置综合回测报告（Stage 2c）")
    print("=" * 72)
    print(f"  池子: 扩展池 (S&P 500 + S&P 400)")
    print(f"  因子: quality_roe + quality_gm + growth_rev（行业中性化）")
    print(f"  合成: 等权固定正向")
    print(f"  组合: 分层多头 L5（top 20%）")
    print(f"  再平衡: {REBALANCE}日 | 费用: 美股每股+每笔最低费")
    print()

    # ═══════════════════════════════════════════════════════════
    # 1. 数据准备
    # ═══════════════════════════════════════════════════════════
    print("[1/7] 构建面板...", flush=True)
    panels = build_ohlcv_panels(EXPANDED_US_POOL, loader=us_loader)
    close = panels["close"]
    n_stocks = close.notna().any().sum()
    span = f"{close.index.min().date()} ~ {close.index.max().date()}"
    print(f"      行情: {close.shape[0]}天 × {n_stocks}只 | {span}")

    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    n_fund = fund["roe"].notna().any().sum()
    print(f"      基本面: {n_fund} 只有效数据")

    fwd = forward_returns(close, horizon=HORIZON)

    # ─── 因子构建 + 行业中性化 ───
    ind = industry_series(list(close.columns))
    raw_facs = {
        "quality_roe": F.us_quality_roe(fund["roe"]),
        "quality_gm": F.us_quality_roe(fund["gross_margin"]),
        "growth_rev": F.us_growth(fund["rev_yoy"]),
    }
    facs = {n: neut_fn(f, industry=ind, mode="industry") for n, f in raw_facs.items()}
    eq_factor = combine_factors(*facs.values())

    # ═══════════════════════════════════════════════════════════
    # 2. 全样本因子诊断
    # ═══════════════════════════════════════════════════════════
    print("\n[2/7] 因子诊断（全样本 IC + 相关性）...", flush=True)

    print("\n      ▸ 全样本 IC 统计")
    print(f"      {'Factor':16s} {'meanIC':>8s} {'stdIC':>8s} {'ICIR':>7s} {'t-stat':>7s} {'posRate':>8s}")
    print(f"      {'-'*60}")
    ic_data = {}
    for name, fac in facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        ic_data[name] = s
        print(f"      {name:16s} {s['mean_ic']:>+8.4f} {s['std_ic']:>8.4f} "
              f"{s['icir']:>+7.2f} {s['t_stat']:>+7.2f} {s['positive_rate']:>7.1%}")

    # ─── 逐年 IC 热力图 ───
    yearly_ic = {}
    for name, fac in facs.items():
        ic_series = daily_ic(fac, fwd)
        yearly_ic[name] = ic_series.groupby(ic_series.index.year).mean()
    df_yearly = pd.DataFrame(yearly_ic)
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(df_yearly.T.values, cmap="RdBu_r", aspect="auto", vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(len(df_yearly.index)))
    ax.set_xticklabels([str(y) for y in df_yearly.index], rotation=45)
    ax.set_yticks(range(len(df_yearly.columns)))
    ax.set_yticklabels(list(df_yearly.columns))
    for i in range(len(df_yearly.columns)):
        for j in range(len(df_yearly.index)):
            v = df_yearly.iloc[j, i] if i < len(df_yearly) else np.nan
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(v) < 0.04 else "white")
    ax.set_title(f"Yearly Mean IC by Factor ({span})")
    plt.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "report_yearly_ic.png")

    # ─── 因子相关性矩阵 ───
    corr_mat = factor_correlation(facs)
    print(f"\n      ▸ 因子相关性（中性化后）")
    print(corr_mat.round(3).to_string())

    # ═══════════════════════════════════════════════════════════
    # 3. 全样本净值
    # ═══════════════════════════════════════════════════════════
    print("\n[3/7] 全样本分层多头净值...", flush=True)

    bt_eq = long_top_layer(close, eq_factor, n_layers=N_LAYERS,
                           rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    best_name = max(ic_data, key=lambda k: ic_data[k]["t_stat"])
    bt_best = long_top_layer(close, facs[best_name], n_layers=N_LAYERS,
                             rebalance_every=REBALANCE, cost_fn=US_COST_FN)

    # ─── 主净值图 ───
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt_eq["equity"].index, bt_eq["equity"], label="Equal-Weight (quality_roe+quality_gm+growth_rev)", lw=2.0, color="steelblue")
    ax.plot(bt_best["equity"].index, bt_best["equity"], label=f"Best Single ({best_name})", lw=1.2, color="darkorange")
    ax.plot(bt_eq["benchmark"].index, bt_eq["benchmark"], label="Benchmark (equal-hold)", lw=1.0, ls="--", color="gray")
    ax.set_title(f"US Expanded Pool ({n_stocks} stocks, {span}): Long-Only L5 Portfolio\n"
                 f"Industry-Neutralized, Equal-Weight Composite, {REBALANCE}d Rebalance")
    ax.set_ylabel("Net Value (log scale)"); ax.set_yscale("log")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    _save(fig, "report_equity.png")

    print(f"      {_perf_line(bt_eq['port_ret'], '等权合成 L5')}")
    print(f"      {_perf_line(bt_best['port_ret'], f'最优单因子 {best_name}')}")
    print(f"      {_perf_line(bt_eq['benchmark_ret'], '等权基准')}")

    # ═══════════════════════════════════════════════════════════
    # 4. 逐年收益拆解
    # ═══════════════════════════════════════════════════════════
    print("\n[4/7] 逐年收益拆解...", flush=True)

    yearly = pd.DataFrame({
        "year": bt_eq["port_ret"].index.year,
        "ew": bt_eq["port_ret"].values,
        "bench": bt_eq["benchmark_ret"].values,
    })
    yg = yearly.groupby("year")
    yr_ew = yg["ew"].apply(lambda x: (1 + x).prod() - 1)
    yr_bench = yg["bench"].apply(lambda x: (1 + x).prod() - 1)
    yr_excess = yr_ew - yr_bench

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(yr_ew))
    w = 0.3
    ax.bar(x - w, yr_ew.values, w, label="Equal-Weight L5", color="steelblue")
    ax.bar(x, yr_bench.values, w, label="Benchmark", color="lightgray")
    ax.bar(x + w, yr_excess.values, w, label="Excess", color="darkorange")
    ax.axhline(y=0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yr_ew.index], rotation=45)
    ax.set_title(f"Yearly Returns: Equal-Weight L5 vs Benchmark")
    ax.set_ylabel("Return"); ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _save(fig, "report_yearly_returns.png")

    print(f"      {'Year':6s} {'EW L5':>8s} {'Bench':>8s} {'Excess':>8s} {'Beat?':>6s}")
    for y in yr_ew.index:
        print(f"      {y:<6d} {yr_ew[y]:>+7.1%} {yr_bench[y]:>+7.1%} {yr_excess[y]:>+7.1%} "
              f"{'YES' if yr_ew[y] > yr_bench[y] else 'NO':>6s}")
    beat_years = int((yr_ew > yr_bench).sum())
    print(f"      跑赢年份: {beat_years}/{len(yr_ew)} ({beat_years/len(yr_ew):.0%})")

    # ═══════════════════════════════════════════════════════════
    # 5. 回撤分析
    # ═══════════════════════════════════════════════════════════
    print("\n[5/7] 回撤分析...", flush=True)

    def drawdown_series(equity):
        peak = equity.cummax()
        return equity / peak - 1.0

    dd_eq = drawdown_series(bt_eq["equity"])
    dd_bench = drawdown_series(bt_eq["benchmark"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dd_eq.index, 0, dd_eq.values, alpha=0.5, color="steelblue", label="EW L5 Drawdown")
    ax.fill_between(dd_bench.index, 0, dd_bench.values, alpha=0.3, color="gray", label="Benchmark Drawdown")
    ax.set_title(f"Drawdown Analysis (max EW: {dd_eq.min():.1%}, max Bench: {dd_bench.min():.1%})")
    ax.set_ylabel("Drawdown"); ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _save(fig, "report_drawdown.png")

    # Top 5 最大回撤期
    dd_sorted = dd_eq.sort_values()
    print(f"      Top 5 最大回撤（等权 L5）:")
    for i, (date, dd) in enumerate(dd_sorted.head(5).items()):
        print(f"        {i+1}. {date.date()}  {dd:.1%}")

    # ═══════════════════════════════════════════════════════════
    # 6. 滚动窗口稳定性
    # ═══════════════════════════════════════════════════════════
    print("\n[6/7] 滚动 walk-forward 稳定性验证...", flush=True)

    periods = []
    start = 0
    n = len(close)
    while start + TRAIN_SIZE + TEST_SIZE <= n:
        train_end = start + TRAIN_SIZE
        test_end = train_end + TEST_SIZE
        test = close.iloc[train_end - 1:test_end]

        # 在该 test 窗口内用因子直接跑（无前视：因子已在前面由全样本计算，
        # 但因为是等权且无IC加权，不依赖train段IC，可以用全样本因子直接切片）
        test_facs = {n: f.iloc[train_end - 1:test_end] for n, f in facs.items()}
        test_eq = combine_factors(*test_facs.values())

        bt = long_top_layer(test, test_eq, n_layers=N_LAYERS,
                            rebalance_every=REBALANCE, first_rebalance=True,
                            cost_fn=US_COST_FN).iloc[1:]
        bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
        bt["benchmark"] = (1.0 + bt["benchmark_ret"]).cumprod()
        m = summary(bt["equity"], bt["port_ret"])
        bm = summary(bt["benchmark"], bt["benchmark_ret"])
        periods.append({
            "train_start": close.index[start], "train_end": close.index[train_end],
            "test_start": bt.index[0], "test_end": bt.index[-1],
            "sharpe": m["sharpe"], "bench_sharpe": bm["sharpe"],
            "beat": m["sharpe"] > bm["sharpe"],
            "excess": m["sharpe"] - bm["sharpe"],
        })
        start += STEP

    nw = len(periods)
    beat_rate = sum(p["beat"] for p in periods) / nw
    exc_sharpes = [p["excess"] for p in periods]

    # ─── 滚动窗口夏普对比图 ───
    fig, ax = plt.subplots(figsize=(14, 5))
    xs = range(nw)
    ax.bar(xs, [p["sharpe"] for p in periods], alpha=0.7, label="EW L5 Sharpe", color="steelblue")
    ax.bar(xs, [p["bench_sharpe"] for p in periods], alpha=0.5, label="Benchmark Sharpe", color="gray")
    ax.axhline(y=0, color="black", lw=0.8)
    ax.set_title(f"Rolling Walk-Forward ({nw} windows, train{TRAIN_SIZE}/test{TEST_SIZE}/step{STEP}): "
                 f"Beat Rate {beat_rate:.0%}, Median Excess Sharpe {np.median(exc_sharpes):+.2f}")
    ax.set_xlabel("Window #"); ax.set_ylabel("Sharpe Ratio"); ax.legend()
    _save(fig, "report_rolling_sharpe.png")

    print(f"      滚动窗口数: {nw}")
    print(f"      跑赢率: {beat_rate:.0%}")
    print(f"      中位超额夏普: {np.median(exc_sharpes):+.2f}")
    print(f"      超额夏普 std: {np.std(exc_sharpes):.2f}")

    # ═══════════════════════════════════════════════════════════
    # 7. 分层单调性 + 换手分析
    # ═══════════════════════════════════════════════════════════
    print("\n[7/7] 分层单调性 + 换手分析...", flush=True)

    res = layered_backtest(close, eq_factor, n_layers=N_LAYERS,
                           rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    lsum = layer_summary(res, n_layers=N_LAYERS)
    print(f"\n      ▸ 各层收益（含美股费用）:")
    for _, row in lsum.iterrows():
        print(f"        {row['layer']:12s} 累计{row['total_return']:>+8.1%}  "
              f"夏普{row['sharpe']:>+6.2f}  回撤{row['max_drawdown']:>+6.1%}")

    # 换手率
    to = res["turnover"]
    avg_to = to.mean()
    print(f"\n      ▸ 平均换手率（每再平衡日）:")
    for col in to.columns:
        print(f"        {col}: {avg_to[col]:.1%}")

    # ═══════════════════════════════════════════════════════════
    # 最终总结
    # ═══════════════════════════════════════════════════════════
    m_eq = summary(bt_eq["equity"], bt_eq["port_ret"])
    m_best = summary(bt_best["equity"], bt_best["port_ret"])
    m_bench = summary(bt_eq["benchmark"], bt_eq["benchmark_ret"])

    summary_lines = [
        "=" * 72,
        "  美股最优配置综合回测报告 — 最终总结",
        "=" * 72,
        "",
        f"  数据区间: {span}",
        f"  股票数量: {n_stocks} 只（行情）/ {n_fund} 只（基本面）",
        f"  行业数量: {ind.nunique()} 个（GICS 一级）",
        "",
        "  —— 最优配置 ——",
        "  池子: S&P 500 + S&P 400 扩展池",
        "  因子: quality_roe + quality_gm + growth_rev",
        "  中性化: 行业内去均值（GICS 一级行业）",
        "  合成: 等权固定正向（不按IC换向）",
        "  组合: 分层多头 L5（top 20%）",
        f"  再平衡: {REBALANCE} 交易日",
        "  费用: 美股每股费 + 每笔最低费",
        "",
        "  —— 全样本绩效 ——",
        f"  {'':20s} {'累计收益':>8s} {'年化收益':>8s} {'夏普':>6s} {'回撤':>8s} {'Calmar':>6s}",
        f"  等权合成 L5      {m_eq['total_return']:>+8.1%} {m_eq['annualized_return']:>+8.1%} "
        f"{m_eq['sharpe']:>+6.2f} {m_eq['max_drawdown']:>+8.1%} "
        f"{m_eq['annualized_return']/max(m_eq['max_drawdown'],0.001):>+6.2f}",
        f"  最优单因子       {m_best['total_return']:>+8.1%} {m_best['annualized_return']:>+8.1%} "
        f"{m_best['sharpe']:>+6.2f} {m_best['max_drawdown']:>+8.1%} "
        f"{m_best['annualized_return']/max(m_best['max_drawdown'],0.001):>+6.2f}",
        f"  等权基准         {m_bench['total_return']:>+8.1%} {m_bench['annualized_return']:>+8.1%} "
        f"{m_bench['sharpe']:>+6.2f} {m_bench['max_drawdown']:>+8.1%} "
        f"{m_bench['annualized_return']/max(m_bench['max_drawdown'],0.001):>+6.2f}",
        "",
        "  —— 滚动稳健性 ——",
        f"  窗口数: {nw}（train{TRAIN_SIZE}/test{TEST_SIZE}/step{STEP}）",
        f"  跑赢率: {beat_rate:.0%}",
        f"  中位超额夏普: {np.median(exc_sharpes):+.2f}",
        f"  超额夏普标准差: {np.std(exc_sharpes):.2f}",
        "",
        "  —— 逐年收益 ——",
    ]
    for y in yr_ew.index:
        summary_lines.append(f"  {y}: EW {yr_ew[y]:>+7.1%}  Bench {yr_bench[y]:>+7.1%}  "
                             f"Excess {yr_excess[y]:>+7.1%}  {'✅' if yr_ew[y] > yr_bench[y] else '❌'}")
    summary_lines.append(f"  跑赢年份: {beat_years}/{len(yr_ew)} ({beat_years/len(yr_ew):.0%})")

    summary_lines += [
        "",
        "  —— 因子 IC 稳定性 ——",
    ]
    for name, s in ic_data.items():
        summary_lines.append(f"  {name:16s}: meanIC {s['mean_ic']:+.4f}  "
                             f"ICIR {s['icir']:+.2f}  t {s['t_stat']:+.2f}  "
                             f"posRate {s['positive_rate']:.0%}")

    summary_lines += [
        "",
        "  —— 与 M17 基线对比 ——",
        f"  {'':20s} {'M17基线(72只)':>16s} {'本次(454只)':>16s}",
        f"  {'稳定因子数':20s} {'1':>16s} {'3':>16s}",
        f"  {'等权跑赢率':20s} {'50%':>16s} {f'{beat_rate:.0%}':>16s}",
        f"  {'等权超额夏普':20s} {'+0.07':>16s} {f'{np.median(exc_sharpes):+.2f}':>16s}",
        f"  {'超额收益(pct)':20s} {'+143pct':>16s} "
        f"{m_eq['total_return'] - m_bench['total_return']:>+16.0%}",
        "",
        "  —— 已知局限 ——",
        "  1. 多空对冲不适用：L1 空头腿无 alpha（与A股M12一致）",
        "  2. 金融股+REITs无基本面数据（东财接口限制）",
        "  3. 回撤仍偏高（~38%），需引入风险管理",
        "  4. 因子广度有限（仅3个），等权分散空间受限",
        "  5. 无市值中性化（美股缺少日频市值数据源）",
        "  6. 静态池，未处理幸存者偏差和退市",
        "",
        "=" * 72,
    ]

    report_txt = "\n".join(summary_lines)
    print(f"\n{report_txt}")

    txt_path = REPORT_DIR / "report_summary.txt"
    txt_path.write_text(report_txt, encoding="utf-8")
    print(f"  报告文本已保存: {txt_path}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
