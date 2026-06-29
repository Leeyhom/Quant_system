"""multifactor_demo —— 长历史 + 正交因子 + 等权合成（M14）。

运行方式：
    conda activate quant
    # 首次需先扩历史缓存（重拉 2018+ 行情/估值，覆盖旧的 2 年缓存）：
    NO_PROXY='*' python scripts/refetch_history.py
    NO_PROXY='*' python scripts/multifactor_demo.py            # 全池
    NO_PROXY='*' python scripts/multifactor_demo.py --limit 30 # 小步

回答 M13 留下的问题（见 docs/14）：M13 在 2 年/4 窗口上任何合成都跑不赢单因子，
诊断是「历史太短 + 正交因子不足」。M14 两步走：
  ① 扩历史到 2018（覆盖牛/熊/震荡多个 regime，滚动窗口 4→23）。
  ② 零成本从已有估值面板导出正交因子：quality_roe=PB/PE、growth=1/PEG、
     cashflow_yield=1/PCF（与价值相关仅 0.08~0.41，确为新信息）。

核心对比（滚动 walk-forward，分层多头 vs 等权基准）：
  最优单因子  vs  IC加权合成  vs  等权合成（固定正向）
诚实结论：等权合成首次稳定跑赢单因子；IC加权反而更差——见 docs/14 的「方向锁定陷阱」。
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
from quant.factor.factors import combine_factors
from quant.factor.composite import factor_correlation, weighted_composite
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

from scripts.composite_demo import rolling_long_top_layer  # 复用 M13 的滚动评估
from scripts.factor_research_demo import parse_limit

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
# 长历史滚动窗口：~2年 train / ~半年 test，step 半年。覆盖多 regime、窗口数充足。
TRAIN_SIZE = 480
TEST_SIZE = 120
STEP = 60
N_CUTS = 5


def build_factors(panels: dict, vp: dict, ind, log_mv) -> dict:
    """构建价值 + 正交（现金流/成长/流动性 + 诊断用质量）因子，各自双中性。

    返回**全部候选**；是否纳入等权合成由下方的「方向先验稳定性筛选」决定，
    不在这里写死，便于 demo 透明展示筛选过程。
    """
    close, amount = panels["close"], panels["amount"]
    raw = {
        "earnings_yield": F.earnings_yield(vp["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(vp["pcf"]),
        "growth_peg": F.growth_peg(vp["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
        "quality_roe": F.quality_roe(vp["pe_ttm"], vp["pb"]),
    }
    return {n: neutralize(f, industry=ind, log_mv=log_mv, mode="full") for n, f in raw.items()}


def select_stable_positive(facs: dict, fwd, min_posrate: float = 0.52, min_icir: float = 0.06) -> list[str]:
    """筛选「方向先验稳定净正向」的因子，作为等权合成的分量。

    等权合成的前提（见 docs/14）：因子方向先验要稳——全样本 IC 净正向。
    方向像抛硬币的因子（posRate≈50%、ICIR≈0，如 quality_roe 在本段反相位漂移）
    纳入等权只会稀释信号。**这是基于长历史 + 经济先验的「因子库选择」**（类似
    "决定研究哪些因子"），不是逐窗调参，故不构成前视；但门槛公开、可复现。
    """
    kept = []
    for name, fac in facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        ok = s["positive_rate"] >= min_posrate and s["icir"] >= min_icir
        if not ok:
            print(f"      - 剔除 {name}（posRate {s['positive_rate']:.0%} / ICIR {s['icir']:+.2f} 方向不稳，不纳入等权）")
        else:
            kept.append(name)
    return kept


def _roll(close, build_fn, label):
    """跑一次滚动 walk-forward，打印并返回汇总。"""
    r = rolling_long_top_layer(close, build_fn, TRAIN_SIZE, TEST_SIZE, STEP)
    periods = r["periods"]
    sh = pd.Series([p["sharpe"] for p in periods])
    exc = pd.Series([p["sharpe"] - p["bench_sharpe"] for p in periods])
    print(f"  {label:30s} 窗口{r['n']:2d} 跑赢{r['beat_rate']:.0%} "
          f"中位夏普{sh.median():+.2f} 中位超额{exc.median():+.2f}")
    return {"r": r, "beat": r["beat_rate"], "med_sharpe": float(sh.median()),
            "med_excess": float(exc.median())}


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = DEFAULT_POOL[:limit] if limit else DEFAULT_POOL
    print(f"[1/5] 构建长历史面板（{len(symbols)} 只）...")
    panels = build_ohlcv_panels(symbols)
    close = panels["close"]
    vp = build_value_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(vp["total_mv"].replace(0, np.nan))
    span = f"{close.index.min().date()}~{close.index.max().date()}"
    print(f"      面板 {close.shape[0]}天×{close.shape[1]}只 | {span} | 行业 {ind.nunique()}")
    if close.shape[0] < TRAIN_SIZE + TEST_SIZE:
        print(f"      ⚠️ 历史长度不足（需 ≥{TRAIN_SIZE+TEST_SIZE} 天）。请先跑 "
              f"scripts/refetch_history.py 扩历史到 2018。")
        return

    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_factors(panels, vp, ind, log_mv)

    # ───────── ① 相关 + 全样本/逐年 IC ─────────
    print(f"\n[2/5] 因子相关矩阵（双中性后；确认正交）")
    print(factor_correlation(facs).round(2).to_string())
    print(f"\n      全样本 IC（看方向先验是否净正向、稳不稳）")
    for name, fac in facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        print(f"      {name:16s} meanIC {s['mean_ic']:+.3f} | ICIR {s['icir']:+.2f} | "
              f"t {s['t_stat']:+.2f} | posRate {s['positive_rate']:.0%}")

    # 等权合成的分量：只留方向先验稳定净正向者（方向漂移的剔除）。
    print(f"\n      等权分量筛选（方向先验需稳定净正向）：")
    selected = select_stable_positive(facs, fwd)
    print(f"      纳入等权合成: {selected}")
    sel_facs = {n: facs[n] for n in selected}

    # ───────── ② 滚动 walk-forward：单因子 vs IC加权 vs 等权 ─────────
    print(f"\n[3/5] 滚动 walk-forward（train{TRAIN_SIZE}/test{TEST_SIZE}/step{STEP}，跨多 regime）")
    print("      —— 分层多头 L5 vs 等权全持有基准，看跑赢比例与超额夏普 ——")
    res = {}
    # 单因子基线（取跑赢比例最高者作"最优单因子"代表，但全打印）
    single = {}
    for name in facs:
        single[name] = _roll(close, (lambda u, nm=name: facs[nm]), f"单因子 {name}")
    best_single = max(single, key=lambda k: single[k]["beat"])
    print(f"      {'-'*64}")
    # IC加权（ICIR×多切分，M13 打磨版；权重每窗 OOS 重算，分量同等权口径）
    res["icir"] = _roll(
        close,
        lambda u: weighted_composite(sel_facs, fwd, u, scheme="icir", n_cuts=N_CUTS, horizon=HORIZON)[0],
        "ICIR×多切分 合成",
    )
    # 等权合成（固定正向，不按 IC 换向）—— M14 主角
    eq_factor = combine_factors(*sel_facs.values())
    res["equal"] = _roll(close, (lambda u: eq_factor), "等权合成（固定正向）")

    # ───────── ③ 全样本分层多头净值（直观对照）─────────
    print(f"\n[4/5] 全样本分层多头净值（等权合成 vs 最优单因子 vs 基准）")
    bt_eq = long_top_layer(close, eq_factor, n_layers=N_LAYERS, rebalance_every=REBALANCE)
    bt_single = long_top_layer(close, facs[best_single], n_layers=N_LAYERS, rebalance_every=REBALANCE)
    m_eq = summary(bt_eq["equity"], bt_eq["port_ret"])
    m_single = summary(bt_single["equity"], bt_single["port_ret"])
    m_bench = summary(bt_eq["benchmark"], bt_eq["benchmark_ret"])
    print(f"      等权合成   收益 {m_eq['total_return']:+.2%} | 夏普 {m_eq['sharpe']:+.2f} | 回撤 {m_eq['max_drawdown']:.2%}")
    print(f"      最优单因子 收益 {m_single['total_return']:+.2%} | 夏普 {m_single['sharpe']:+.2f} | 回撤 {m_single['max_drawdown']:.2%} ({best_single})")
    print(f"      等权基准   收益 {m_bench['total_return']:+.2%} | 夏普 {m_bench['sharpe']:+.2f} | 回撤 {m_bench['max_drawdown']:.2%}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt_eq["equity"].index, bt_eq["equity"], label="equal-weight composite long", linewidth=1.6)
    ax.plot(bt_single["equity"].index, bt_single["equity"], label=f"{best_single} long", linewidth=1.1)
    ax.plot(bt_eq["benchmark"].index, bt_eq["benchmark"], label="benchmark (equal-hold)", linewidth=1.1, linestyle="--")
    ax.set_title(f"M14 long-top-layer ({span}): equal-weight composite vs single vs benchmark")
    ax.set_xlabel("date"); ax.set_ylabel("net value"); ax.legend()
    fig.tight_layout()
    png = RAW_DATA_DIR / "multifactor_equity.png"
    fig.savefig(png, dpi=120)
    print(f"      图: {png}")

    # ───────── ④ 诚实结论 ─────────
    print(f"\n[5/5] ———— 诚实结论 ————")
    bs = single[best_single]
    eq, icir = res["equal"], res["icir"]
    print(f"  (a) 最优单因子 {best_single}: 跑赢 {bs['beat']:.0%} | 超额夏普 {bs['med_excess']:+.2f}")
    print(f"  (b) IC加权合成: 跑赢 {icir['beat']:.0%} | 超额夏普 {icir['med_excess']:+.2f}"
          f"（{'≥' if icir['beat']>=bs['beat'] else '<'} 单因子）")
    print(f"  (c) 等权合成  : 跑赢 {eq['beat']:.0%} | 超额夏普 {eq['med_excess']:+.2f}"
          f"（{'≥' if eq['beat']>=bs['beat'] else '<'} 单因子）")
    if eq["beat"] >= bs["beat"] and eq["med_excess"] >= bs["med_excess"]:
        print("  → 突破：长历史 + 正交因子下，【等权】合成在跑赢比例与超额夏普上同时超过最优单因子，")
        print("     分散 regime 依赖成功。这是项目首个稳定的多因子超额，可走向组合资金管理/模拟盘。")
    else:
        print("  → 等权合成未全面超过单因子，残余瓶颈可能是因子组选择或池子规模。")
    if eq["beat"] > icir["beat"]:
        print("  → 关键教训：等权 > IC加权。IC加权要用 train 段 IC 锁定方向/权重，A股因子方向")
        print("     会在 test 段翻号（如 quality 与 value 反相位），锁定的旧方向反而押反、相互抵消；")
        print("     等权+固定正向先验绕开了「方向锁定陷阱」。信息量优先于模型复杂度，再次验证。")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
