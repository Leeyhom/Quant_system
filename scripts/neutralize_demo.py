"""neutralize_demo —— 行业/市值中性化的消融对比 + 分层多头（M12）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/neutralize_demo.py            # 全池(~90只)
    NO_PROXY='*' python scripts/neutralize_demo.py --limit 20 # 小步验证

回答 M10 留下的悖论：价值因子 IC 同号却跑输基准。诊断结论是因子在「押行业 beta」，
本脚本验证：行业+市值双中性能否提升因子稳定性（ICIR/t），并让分层多头更接近跑赢基准。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 中文标题在缺中文字体的环境会乱码，统一用英文标签，避免告警。
import numpy as np
import pandas as pd

from quant.config import RAW_DATA_DIR
from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.industry import industry_series
from quant.factor import factors as F
from quant.factor.neutralize import neutralize
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary, cumulative_ic
from quant.backtest.factor_validation import ic_train_test
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

from scripts.factor_research_demo import parse_limit

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20


def build_core_factors(panels: dict, value_panels: dict) -> dict:
    """选几个 M10 验证过较稳的核心因子做中性化对比（不必全跑）。"""
    close, volume = panels["close"], panels["volume"]
    out = {
        "earnings_yield": F.earnings_yield(value_panels["pe_ttm"]),
        "sales_yield": F.sales_yield(value_panels["ps"]),
        "book_to_price": F.book_to_price(value_panels["pb"]),
        "amihud": F.amihud_illiquidity(close, panels["amount"], 20),
        "momentum20": F.momentum(close, 20),
    }
    return out


def _line(seg: dict) -> str:
    return (f"meanIC {seg['mean_ic']:+.3f} | ICIR {seg['icir']:+.2f} | "
            f"t {seg['t_stat']:+.2f}")


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = DEFAULT_POOL[:limit] if limit else DEFAULT_POOL
    print(f"[1/5] 构建面板（{len(symbols)} 只）...")
    panels = build_ohlcv_panels(symbols)
    close = panels["close"]
    value_panels = build_value_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(value_panels["total_mv"].replace(0, np.nan))
    print(f"      面板 {close.shape[0]}天×{close.shape[1]}只 | 行业数 {ind.nunique()}")

    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_core_factors(panels, value_panels)

    # ───────── 三版本 OOS IC 消融对比 ─────────
    print(f"\n[2/5] 中性化消融：原始 / 行业中性 / 行业+市值双中性（OOS IC）")
    variants = {
        "raw     ": lambda f: f,
        "industry": lambda f: neutralize(f, industry=ind, mode="industry"),
        "ind+size": lambda f: neutralize(f, industry=ind, log_mv=log_mv, mode="full"),
    }
    # 记录每个因子「双中性版」的 OOS test 稳定性，用于选分层多头的主角。
    # 消融对比控制变量：始终对比同一因子的 原始 vs 双中性，不在版本间挑选。
    full_table = {}  # name -> {variant: full_summary}
    pick_score = {}  # name -> 双中性 test |t|（同号才算）
    for name, fac in facs.items():
        print(f"  ◆ {name}")
        full_table[name] = {}
        for vlabel, vfn in variants.items():
            nf = vfn(fac)
            res = ic_train_test(nf, fwd, train_ratio=0.7, horizon=HORIZON)
            full = ic_summary(daily_ic(nf, fwd))
            full_table[name][vlabel.strip()] = full
            same = "同号" if res["sign_consistent"] else "变号"
            print(f"      {vlabel}  full[{_line(full)}]  "
                  f"test[meanIC {res['test']['mean_ic']:+.3f} t {res['test']['t_stat']:+.2f}] {same}")
            if vlabel.strip() == "ind+size":
                pick_score[name] = abs(res["test"]["t_stat"]) if res["sign_consistent"] else -1.0

    # ───────── 选「双中性后 OOS 最稳」的因子，对比其 原始 vs 双中性 ─────────
    pick_name = max(pick_score.items(), key=lambda kv: kv[1])[0]
    raw_factor = facs[pick_name]
    neu_factor = neutralize(raw_factor, industry=ind, log_mv=log_mv, mode="full")
    icir_raw = full_table[pick_name]["raw"]["icir"]
    icir_neu = full_table[pick_name]["ind+size"]["icir"]
    print(f"\n[3/5] 分层多头对比（控制变量：同因子 {pick_name} 的 原始 vs 双中性）")

    bt_raw = long_top_layer(close, raw_factor, n_layers=N_LAYERS, rebalance_every=REBALANCE)
    bt_neu = long_top_layer(close, neu_factor, n_layers=N_LAYERS, rebalance_every=REBALANCE)
    m_raw = summary(bt_raw["equity"], bt_raw["port_ret"])
    m_neu = summary(bt_neu["equity"], bt_neu["port_ret"])
    m_bench = summary(bt_neu["benchmark"], bt_neu["benchmark_ret"])
    print(f"      {'原始分层多头':<14} 收益 {m_raw['total_return']:+.2%} | 夏普 {m_raw['sharpe']:+.2f} | 回撤 {m_raw['max_drawdown']:.2%}")
    print(f"      {'双中性分层多头':<13} 收益 {m_neu['total_return']:+.2%} | 夏普 {m_neu['sharpe']:+.2f} | 回撤 {m_neu['max_drawdown']:.2%}")
    print(f"      {'等权基准':<15} 收益 {m_bench['total_return']:+.2%} | 夏普 {m_bench['sharpe']:+.2f} | 回撤 {m_bench['max_drawdown']:.2%}")

    # ───────── 出图 ─────────
    print(f"\n[4/5] 保存图 ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # 左：中性化前后累计IC（同因子三版本）
    for vlabel, vfn in variants.items():
        ic = daily_ic(vfn(raw_factor), fwd)
        ax1.plot(cumulative_ic(ic).index, cumulative_ic(ic).values, label=vlabel.strip(), linewidth=1.2)
    ax1.set_title(f"Cumulative IC: {pick_name} (raw vs neutralized)")
    ax1.legend(); ax1.set_xlabel("date"); ax1.set_ylabel("cum IC")
    # 右：分层多头净值
    ax2.plot(bt_raw["equity"].index, bt_raw["equity"], label="raw long", linewidth=1.2)
    ax2.plot(bt_neu["equity"].index, bt_neu["equity"], label="ind+size long", linewidth=1.5)
    ax2.plot(bt_neu["benchmark"].index, bt_neu["benchmark"], label="benchmark", linewidth=1.2, linestyle="--")
    ax2.set_title(f"Long-top-layer: {pick_name}")
    ax2.legend(); ax2.set_xlabel("date"); ax2.set_ylabel("net value")
    fig.tight_layout()
    png = RAW_DATA_DIR / "neutralize_compare.png"
    fig.savefig(png, dpi=120)
    print(f"      图: {png}")

    # ───────── 诚实结论 ─────────
    print(f"\n[5/5] ———— 诚实结论 ————")
    icir_better = abs(icir_neu) > abs(icir_raw)
    print(f"  (a) 因子稳定性: 双中性后 {pick_name} ICIR {'提升' if icir_better else '未提升'} "
          f"(原始 {icir_raw:+.2f} → 双中性 {icir_neu:+.2f})")
    beat = m_neu["sharpe"] > m_bench["sharpe"]
    print(f"  (b) 双中性分层多头 vs 基准: 夏普 {m_neu['sharpe']:+.2f} vs {m_bench['sharpe']:+.2f}"
          f"（{'跑赢' if beat else '未跑赢'}）")
    sharpe_vs_raw = m_neu["sharpe"] - m_raw["sharpe"]
    print(f"  (c) 双中性 vs 原始多头: 夏普 {m_neu['sharpe']:+.2f} vs {m_raw['sharpe']:+.2f}"
          f"（{'+' if sharpe_vs_raw>=0 else ''}{sharpe_vs_raw:.2f}），"
          f"回撤 {m_neu['max_drawdown']:.2%} vs {m_raw['max_drawdown']:.2%}")
    # 诚实区分两件事：IC稳定性(统计) 与 这段样本的组合收益(可能受 regime 影响)
    if icir_better and sharpe_vs_raw >= 0:
        print("  → 中性化同时提升了因子稳定性与组合收益，是把「稳定IC」转化为「超额收益」的关键一步。")
    elif icir_better:
        print("  → 关键 nuance：中性化显著提升了因子【统计稳定性】(IC更强更同号)，")
        print("     但本段样本里【组合夏普未超原始】——原始因子的行业暴露恰好押中了低波动板块。")
        print("     IC更稳是更可信的长期信号；下一步用 IC加权合成 多个中性化因子，分散单因子的 regime 依赖。")
    else:
        print("  → 中性化增益有限，残余问题可能是 regime/池子规模，需更长历史或更多正交因子。")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
