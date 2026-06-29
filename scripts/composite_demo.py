"""composite_demo —— 多因子 IC 加权合成验证（M13）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/composite_demo.py            # 全池(~90只)
    NO_PROXY='*' python scripts/composite_demo.py --limit 20 # 小步验证

回答 M12 留下的问题（见 docs/13）：单因子即使中性化也躲不过 regime 依赖
（2025Q3 价值因子单季 IC 转负）。把多个**已中性化、OOS 同号、低相关**的因子
合成，能否分散 regime 风险，把「稳定 IC」兑现成「稳定超额」？

三件事对比：
  ① 相关矩阵去冗余（高相关的同类价值因子只留代表）。
  ② OOS IC 稳定性：最优单因子 vs 等权合成 vs IC加权合成。
  ③ 分层多头滚动 walk-forward：跑赢等权基准的窗口比例谁更高（权重每窗 OOS 重算）。
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
from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.industry import industry_series
from quant.factor import factors as F
from quant.factor.neutralize import neutralize
from quant.factor.composite import factor_correlation, ic_weighted_composite, weighted_composite
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary, cumulative_ic
from quant.backtest.factor_validation import ic_train_test, build_oriented_composite
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

from scripts.factor_research_demo import parse_limit

HORIZON = 20
TRAIN_RATIO = 0.7
N_LAYERS = 5
REBALANCE = 20
CORR_THRESHOLD = 0.7  # 相关高于此值视作冗余，同类只留一个代表
N_CUTS = 5            # 多切分点加权：在 train 后半段取几个 expanding 切分点平均


def build_candidate_factors(panels: dict, value_panels: dict, ind, log_mv) -> dict:
    """构建 M12 验证过 OOS 同号的核心因子，并各自做行业+市值双中性。

    earnings_yield/sales_yield/book_to_price 是价值同类（彼此高相关，靠相关矩阵筛）；
    amihud（流动性）、momentum20（动量）提供正交信息，是分散 regime 的关键。
    """
    close, volume, amount = panels["close"], panels["volume"], panels["amount"]
    raw = {
        "earnings_yield": F.earnings_yield(value_panels["pe_ttm"]),
        "sales_yield": F.sales_yield(value_panels["ps"]),
        "book_to_price": F.book_to_price(value_panels["pb"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "momentum20": F.momentum(close, 20),
    }
    # 全部双中性化（M12）：合成的前提是各分量已剥掉行业/市值 beta。
    return {n: neutralize(f, industry=ind, log_mv=log_mv, mode="full") for n, f in raw.items()}


def select_low_corr(corr: pd.DataFrame, names: list[str], threshold: float) -> list[str]:
    """贪心去冗余：按给定顺序保留因子，若与已保留者相关 >threshold 则丢弃。

    names 的顺序即优先级（价值同类里把先验最稳的 earnings_yield 放前面）。
    """
    kept: list[str] = []
    for name in names:
        redundant = any(abs(corr.loc[name, k]) > threshold for k in kept)
        if redundant:
            print(f"      - 丢弃 {name}（与 {[k for k in kept if abs(corr.loc[name,k])>threshold]} 相关>{threshold}）")
        else:
            kept.append(name)
    return kept


def rolling_long_top_layer(
    close: pd.DataFrame,
    build_fn,
    train_size: int,
    test_size: int,
    step: int,
    cost_fn=None,
) -> dict:
    """对「分层多头」做滚动 walk-forward，每窗用 build_fn 现算因子（防前视）。

    build_fn(upto_date) -> 因子面板：对单因子忽略 upto_date 直接返回；对合成因子
    用 upto_date 之前的 IC 现算权重，保证每个窗口的合成权重都只用过去信息。

    cost_fn：可选费用回调（默认 None=比例成本，向后兼容 A股调用）；传入则透传给
        long_top_layer，用于美股「每股费+每笔最低费」模型（见 quant/backtest/us_cost.py）。

    每窗用 train 末日作预热行建仓，仅统计 test 段。返回每窗 test 夏普、基准夏普、
    是否跑赢，以及跨窗稳定性汇总。
    """
    periods = []
    start = 0
    n = len(close)
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size
        upto_date = close.index[train_end]
        factor = build_fn(upto_date)

        test = close.iloc[train_end - 1:test_end]  # 多带一行预热
        bt = long_top_layer(
            test, factor.reindex_like(test),
            n_layers=N_LAYERS, rebalance_every=REBALANCE, first_rebalance=True,
            cost_fn=cost_fn,
        ).iloc[1:].copy()  # 去掉预热行
        bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
        bt["benchmark"] = (1.0 + bt["benchmark_ret"]).cumprod()
        m = summary(bt["equity"], bt["port_ret"])
        bm = summary(bt["benchmark"], bt["benchmark_ret"])
        periods.append({
            "test_start": bt.index[0], "test_end": bt.index[-1],
            "sharpe": m["sharpe"], "bench_sharpe": bm["sharpe"],
            "beat": m["sharpe"] > bm["sharpe"],
        })
        start += step

    if not periods:
        return {"periods": [], "n": 0, "beat_rate": float("nan"), "median_sharpe": float("nan")}
    sharpes = pd.Series([p["sharpe"] for p in periods])
    return {
        "periods": periods,
        "n": len(periods),
        "beat_rate": float(pd.Series([p["beat"] for p in periods]).mean()),
        "median_sharpe": float(sharpes.median()),
    }


def _line(seg: dict) -> str:
    return (f"meanIC {seg['mean_ic']:+.3f} | stdIC {seg['std_ic']:.3f} | "
            f"ICIR {seg['icir']:+.2f} | t {seg['t_stat']:+.2f}")


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = DEFAULT_POOL[:limit] if limit else DEFAULT_POOL
    print(f"[1/6] 构建面板（{len(symbols)} 只）...")
    panels = build_ohlcv_panels(symbols)
    close = panels["close"]
    value_panels = build_value_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(value_panels["total_mv"].replace(0, np.nan))
    print(f"      面板 {close.shape[0]}天×{close.shape[1]}只 | 行业数 {ind.nunique()}")

    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_candidate_factors(panels, value_panels, ind, log_mv)

    # ───────── ① 相关矩阵去冗余 ─────────
    print(f"\n[2/6] 因子相关矩阵（双中性后；|corr|>{CORR_THRESHOLD} 视作冗余）")
    corr = factor_correlation(facs)
    print(corr.round(2).to_string())
    # 优先级顺序：价值同类把 earnings_yield 放前，再 amihud/momentum 这类正交因子。
    priority = ["earnings_yield", "sales_yield", "book_to_price", "amihud", "momentum20"]
    priority = [n for n in priority if n in facs]
    selected = select_low_corr(corr, priority, CORR_THRESHOLD)
    print(f"      去冗余后参与合成: {selected}")
    sel_facs = {n: facs[n] for n in selected}

    # ───────── ② OOS IC：单因子 vs 三种合成（等权 / |IC| / ICIR多切分）─────────
    cut = int(len(close) * TRAIN_RATIO)
    upto_date = close.index[cut]
    print(f"\n[3/6] OOS IC 稳定性（train_ratio={TRAIN_RATIO}，切分日 {upto_date.date()}）")
    print("      —— 看 test 段 ICIR/|t| 是否提升、stdIC 是否下降（分散见效）——")

    # 单因子（取双中性后 test ICIR 最高者作代表对照）
    single_test = {}
    best_single = None
    for name in selected:
        res = ic_train_test(sel_facs[name], fwd, train_ratio=TRAIN_RATIO, horizon=HORIZON)
        single_test[name] = res
        tag = "同号" if res["sign_consistent"] else "变号"
        print(f"      单因子 {name:<15} test[{_line(res['test'])}] {tag}")
        if best_single is None or abs(res["test"]["icir"]) > abs(single_test[best_single]["test"]["icir"]):
            best_single = name

    # 三种合成的「单点」构造器（用于 OOS IC 对比，切分日 = upto_date）。
    # build_fn(upto) -> 因子面板，供下方滚动 walk-forward 每窗 OOS 重算复用。
    builders = {
        "eq    ": lambda up: build_oriented_composite(sel_facs, fwd, up, horizon=HORIZON)[0],
        "ic     ": lambda up: ic_weighted_composite(sel_facs, fwd, up, horizon=HORIZON)[0],
        "icir×mc": lambda up: weighted_composite(
            sel_facs, fwd, up, scheme="icir", n_cuts=N_CUTS, horizon=HORIZON)[0],
    }
    comp_res = {}
    comps = {}
    for label, bfn in builders.items():
        comp = bfn(upto_date)
        comps[label] = comp
        comp_res[label] = ic_train_test(comp, fwd, train_ratio=TRAIN_RATIO, horizon=HORIZON)

    # 打印 ICIR 多切分版的最终权重（带符号），看方向不稳的因子是否被收缩。
    _, icir_w = weighted_composite(sel_facs, fwd, upto_date, scheme="icir", n_cuts=N_CUTS, horizon=HORIZON)

    print(f"      {'─'*60}")
    print(f"      最优单因子 {best_single:<11} test[{_line(single_test[best_single]['test'])}]")
    for label in builders:
        print(f"      合成 {label:<13} test[{_line(comp_res[label]['test'])}]")
    print(f"      ICIR×多切分权重(带符号): "
          + ", ".join(f"{n}{'+' if w>=0 else '-'}{abs(w):.3f}" for n, w in icir_w.items()))

    # ───────── ③ 分层多头滚动 walk-forward（权重每窗 OOS 重算）─────────
    print(f"\n[4/6] 分层多头滚动 walk-forward（每窗合成权重只用过去 IC 重算）")
    n = len(close)
    train_size = min(240, int(n * 0.5))
    test_size = min(60, max(20, int(n * 0.15)))
    step = test_size

    def build_single(_upto, _name=best_single):
        return sel_facs[_name]

    roll = {"single": rolling_long_top_layer(close, build_single, train_size, test_size, step)}
    for label, bfn in builders.items():
        roll[label] = rolling_long_top_layer(close, bfn, train_size, test_size, step)
    print(f"      窗口数 {roll['single']['n']}（train {train_size}/test {test_size}）")
    print(f"      最优单因子 {best_single:<11} 跑赢基准比例 {roll['single']['beat_rate']:.0%} | 中位夏普 {roll['single']['median_sharpe']:+.2f}")
    for label in builders:
        print(f"      合成 {label:<13} 跑赢基准比例 {roll[label]['beat_rate']:.0%} | 中位夏普 {roll[label]['median_sharpe']:+.2f}")

    # 主角合成 = ICIR×多切分（本轮打磨的目标方法）。
    champ_label = "icir×mc"
    champ_comp = comps[champ_label]
    champ_res = comp_res[champ_label]
    champ_roll = roll[champ_label]

    # ───────── 全样本分层多头净值（直观对照）─────────
    print(f"\n[5/6] 全样本分层多头净值（ICIR×多切分合成 vs 最优单因子 vs 基准）")
    bt_comp = long_top_layer(close, champ_comp, n_layers=N_LAYERS, rebalance_every=REBALANCE)
    bt_single = long_top_layer(close, sel_facs[best_single], n_layers=N_LAYERS, rebalance_every=REBALANCE)
    m_comp = summary(bt_comp["equity"], bt_comp["port_ret"])
    m_single = summary(bt_single["equity"], bt_single["port_ret"])
    m_bench = summary(bt_comp["benchmark"], bt_comp["benchmark_ret"])
    print(f"      ICIR×多切分 收益 {m_comp['total_return']:+.2%} | 夏普 {m_comp['sharpe']:+.2f} | 回撤 {m_comp['max_drawdown']:.2%}")
    print(f"      最优单因子  收益 {m_single['total_return']:+.2%} | 夏普 {m_single['sharpe']:+.2f} | 回撤 {m_single['max_drawdown']:.2%}")
    print(f"      等权基准    收益 {m_bench['total_return']:+.2%} | 夏普 {m_bench['sharpe']:+.2f} | 回撤 {m_bench['max_drawdown']:.2%}")

    # 出图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, factor in [(best_single, sel_facs[best_single]),
                          ("eq comp", comps["eq    "]),
                          ("icir×mc comp", champ_comp)]:
        ic = daily_ic(factor, fwd)
        ax1.plot(cumulative_ic(ic).index, cumulative_ic(ic).values, label=label, linewidth=1.2)
    ax1.set_title("Cumulative IC: best single vs composites")
    ax1.legend(); ax1.set_xlabel("date"); ax1.set_ylabel("cum IC")
    ax2.plot(bt_comp["equity"].index, bt_comp["equity"], label="icir×mc composite long", linewidth=1.5)
    ax2.plot(bt_single["equity"].index, bt_single["equity"], label=f"{best_single} long", linewidth=1.2)
    ax2.plot(bt_comp["benchmark"].index, bt_comp["benchmark"], label="benchmark", linewidth=1.2, linestyle="--")
    ax2.set_title("Long-top-layer: composite vs single vs benchmark")
    ax2.legend(); ax2.set_xlabel("date"); ax2.set_ylabel("net value")
    fig.tight_layout()
    png = RAW_DATA_DIR / "composite_compare.png"
    fig.savefig(png, dpi=120)
    print(f"      图: {png}")

    # ───────── 诚实结论 ─────────
    print(f"\n[6/6] ———— 诚实结论 ————")
    best_icir = single_test[best_single]["test"]["icir"]
    champ_icir = champ_res["test"]["icir"]
    ic_more_stable = abs(champ_icir) >= abs(best_icir)
    std_dropped = champ_res["test"]["std_ic"] <= single_test[best_single]["test"]["std_ic"]
    print(f"  (a) OOS IC 稳定性: ICIR×多切分 test ICIR {champ_icir:+.2f} "
          f"vs 最优单因子 {best_icir:+.2f}（{'提升' if ic_more_stable else '未提升'}），"
          f"stdIC {'下降' if std_dropped else '未降'}")
    icir_vs_ic = champ_icir - comp_res["ic     "]["test"]["icir"]
    print(f"  (b) ICIR×多切分 vs |IC|单点: test ICIR {champ_icir:+.2f} "
          f"vs {comp_res['ic     ']['test']['icir']:+.2f}（{'+' if icir_vs_ic>=0 else ''}{icir_vs_ic:.2f}）")
    beat_better = champ_roll["beat_rate"] >= roll["single"]["beat_rate"]
    print(f"  (c) 滚动跑赢比例: ICIR×多切分 {champ_roll['beat_rate']:.0%} "
          f"vs 最优单因子 {roll['single']['beat_rate']:.0%}（{'更高/持平' if beat_better else '更低'}）")
    if ic_more_stable and beat_better:
        print("  → 打磨奏效：ICIR降噪 + 多切分平均权重把单因子的 regime 依赖分散掉了，")
        print("     OOS IC 更稳且滚动跑赢比例不降。可走向组合资金管理/模拟盘。")
    elif ic_more_stable:
        print("  → ICIR×多切分提升了 OOS IC 稳定性，但滚动跑赢比例未超单因子——")
        print("     收益端 regime 风险仍在，根因更可能是正交信息不足而非加权方法。")
    else:
        print("  → 打磨后合成仍未超事后最优单因子。诚实判断：本池/本段的瓶颈是")
        print("     正交因子太少（看相关矩阵），加权技巧救不了信息量，需扩因子/扩池/拉长历史。")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
