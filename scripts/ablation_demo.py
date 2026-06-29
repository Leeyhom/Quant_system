"""ablation_demo —— 等权合成稳健性消融（M15 验证）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/ablation_demo.py            # 全池
    NO_PROXY='*' python scripts/ablation_demo.py --limit 40 # 小步

为什么要这一步（用户要求「先做2的快速消融，确认 83% 不是窗口设置的运气」）：
M14 的「等权合成跑赢 83%」是在 **一组** 窗口设置（train480/test120/step60、
rebalance20、5层）上得到的。单组设置永远有「恰好挑到好参数」的嫌疑。稳健性消融
的逻辑：如果等权合成跑赢基准、且跑赢最优单因子的结论，在**一大片**窗口/再平衡/
分层设置上都成立，那它就是结构性的优势，而不是某组超参的运气。

做法（关键：因子本身不依赖回测窗口超参，只算一次；只在回测层扫超参）：
  ① 固定因子构建口径（与 multifactor_demo 完全一致，含方向先验筛选）。
  ② 网格扫描 (train,test,step) × rebalance × n_layers。
  ③ 每个组合都跑滚动 walk-forward，记录等权 / 最优单因子 / IC加权 三者的
     跑赢比例与中位超额夏普。
  ④ 汇总：等权在多少比例的设置下「跑赢基准过半」且「跑赢最优单因子」。

诚实地报告失败：若某些设置下等权不再占优，明确列出，不掩盖。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.industry import industry_series
from quant.factor.factors import combine_factors
from quant.factor.composite import weighted_composite
from quant.backtest.ic_analysis import forward_returns
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

from scripts.multifactor_demo import build_factors, select_stable_positive
from scripts.factor_research_demo import parse_limit

HORIZON = 20

# 消融策略：**单轴扫描**而非全笛卡尔积。固定 M14 基线，每次只动一个维度，
# 看等权的优势是否对该维度敏感。这比 36 组全组合更 honest（全组合里很多组只是
# 噪声叠加），也快得多。基线 = (480/120/60, reb20, 5层)。
BASELINE = {"train": 480, "test": 120, "step": 60, "reb": 20, "lyr": 5}

# 每个配置 = 基线改一个键。理由见各注释。
CONFIGS = [
    {"label": "基线", "over": {}},
    {"label": "短窗(360/90/45)", "over": {"train": 360, "test": 90, "step": 45}},  # 窗口更多、每窗信息更少
    {"label": "长train(600)", "over": {"train": 600}},                              # 更多历史定方向
    {"label": "不重叠step(90)", "over": {"step": 90}},                              # step=test，窗口独立
    {"label": "再平衡10(月内)", "over": {"reb": 10}},                               # 更频繁、换手成本更高
    {"label": "再平衡30(季频)", "over": {"reb": 30}},                              # 更稀疏
    {"label": "分层3(每层多)", "over": {"lyr": 3}},                                # 每层股票多、分散强
    {"label": "分层10(每层少)", "over": {"lyr": 10}},                             # 每层股票少、噪声大
]
N_CUTS = 3  # 消融里 IC加权只作陪衬，降到 3 切分点省算力（M14 已证它整体输等权）


def rolling_eval(close, build_fn, train, test, step, n_layers, rebalance) -> dict:
    """参数化的滚动 walk-forward（不依赖 composite_demo 的模块常量，便于扫超参）。

    与 composite_demo.rolling_long_top_layer 同口径：train 末日预热建仓、仅统计
    test 段、每窗用 build_fn(upto_date) 现算因子（合成因子的权重只用过去 IC）。
    """
    periods = []
    start, n = 0, len(close)
    while start + train + test <= n:
        train_end = start + train
        test_end = train_end + test
        upto = close.index[train_end]
        factor = build_fn(upto)
        seg = close.iloc[train_end - 1:test_end]
        bt = long_top_layer(
            seg, factor.reindex_like(seg),
            n_layers=n_layers, rebalance_every=rebalance, first_rebalance=True,
        ).iloc[1:].copy()
        bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
        bt["benchmark"] = (1.0 + bt["benchmark_ret"]).cumprod()
        m = summary(bt["equity"], bt["port_ret"])
        bm = summary(bt["benchmark"], bt["benchmark_ret"])
        periods.append({"sharpe": m["sharpe"], "bench": bm["sharpe"]})
        start += step
    if not periods:
        return {"n": 0, "beat": float("nan"), "med_excess": float("nan")}
    sh = pd.Series([p["sharpe"] for p in periods])
    bn = pd.Series([p["bench"] for p in periods])
    return {
        "n": len(periods),
        "beat": float((sh > bn).mean()),
        "med_excess": float((sh - bn).median()),
    }


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = DEFAULT_POOL[:limit] if limit else DEFAULT_POOL
    print(f"[1/3] 构建面板与因子（{len(symbols)} 只，因子只算一次）...")
    panels = build_ohlcv_panels(symbols)
    close = panels["close"]
    vp = build_value_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(vp["total_mv"].replace(0, np.nan))
    span = f"{close.index.min().date()}~{close.index.max().date()}"
    print(f"      面板 {close.shape[0]}天×{close.shape[1]}只 | {span}")

    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_factors(panels, vp, ind, log_mv)
    selected = select_stable_positive(facs, fwd)
    print(f"      纳入等权合成: {selected}")
    sel_facs = {n: facs[n] for n in selected}
    eq_factor = combine_factors(*sel_facs.values())

    # 最优单因子：用 M14 基线窗口先定出代表（一次），消融时固定用它做对照基准。
    # 这样比较公平：消融检验的是「等权 vs 同一个最优单因子」在各设置下谁更稳。
    def build_eq(_upto):
        return eq_factor

    def build_icir(upto):
        return weighted_composite(sel_facs, fwd, upto, scheme="icir", n_cuts=N_CUTS, horizon=HORIZON)[0]

    print(f"\n[2/3] 单轴消融：基线 + {len(CONFIGS)-1} 个单维度变体 = {len(CONFIGS)} 组")
    print(f"      每组报告 等权/单因子/IC加权 的 (跑赢比例, 中位超额夏普)")

    best_single_name = max(
        selected,
        key=lambda nm: rolling_eval(close, (lambda u, x=nm: facs[x]),
                                    BASELINE["train"], BASELINE["test"], BASELINE["step"],
                                    BASELINE["lyr"], BASELINE["reb"])["beat"],
    )
    print(f"      (最优单因子对照 = {best_single_name})\n", flush=True)

    def build_single(_upto):
        return facs[best_single_name]

    print(f"      {'配置':<18} | {'eq beat/exc':>14} {'single':>14} {'icir':>14} | win?", flush=True)
    rows = []
    for cfg in CONFIGS:
        p = {**BASELINE, **cfg["over"]}
        tr, te, st, reb, lyr = p["train"], p["test"], p["step"], p["reb"], p["lyr"]
        eq = rolling_eval(close, build_eq, tr, te, st, lyr, reb)
        sg = rolling_eval(close, build_single, tr, te, st, lyr, reb)
        ic = rolling_eval(close, build_icir, tr, te, st, lyr, reb)
        eq_wins = (eq["beat"] >= 0.5) and (eq["beat"] >= sg["beat"]) and (eq["beat"] >= ic["beat"])
        rows.append({
            "label": cfg["label"], "nwin": eq["n"],
            "eq_beat": eq["beat"], "eq_exc": eq["med_excess"],
            "sg_beat": sg["beat"], "sg_exc": sg["med_excess"],
            "ic_beat": ic["beat"], "ic_exc": ic["med_excess"],
            "eq_wins": eq_wins,
        })
        mark = "✓" if eq_wins else " "
        print(f"      {cfg['label']:<16} | "
              f"{eq['beat']:>5.0%}/{eq['med_excess']:>+5.2f}   "
              f"{sg['beat']:>5.0%}/{sg['med_excess']:>+5.2f}   "
              f"{ic['beat']:>5.0%}/{ic['med_excess']:>+5.2f} | {mark}  (n={eq['n']})", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n[3/3] ———— 消融汇总（{len(df)} 组设置）————")
    over50 = (df["eq_beat"] >= 0.5).mean()
    eq_ge_single = (df["eq_beat"] >= df["sg_beat"]).mean()
    eq_ge_ic = (df["eq_beat"] >= df["ic_beat"]).mean()
    eq_pos_exc = (df["eq_exc"] > 0).mean()
    print(f"  等权跑赢基准过半(≥50%)的设置占比 : {over50:.0%}  "
          f"(均值跑赢 {df['eq_beat'].mean():.0%}，最差 {df['eq_beat'].min():.0%})")
    print(f"  等权 ≥ 最优单因子 的设置占比      : {eq_ge_single:.0%}")
    print(f"  等权 ≥ IC加权 的设置占比          : {eq_ge_ic:.0%}")
    print(f"  等权中位超额夏普 > 0 的设置占比    : {eq_pos_exc:.0%}  "
          f"(中位 {df['eq_exc'].median():+.2f})")
    all_win = df["eq_wins"].mean()
    print(f"  三条都满足(过半 & ≥单因子 & ≥IC)   : {all_win:.0%}")

    # 哪些设置等权不占优？诚实列出。
    losers = df[~df["eq_wins"]]
    if len(losers):
        print(f"\n  等权未全面占优的 {len(losers)} 组（诚实列出，不掩盖）：")
        for _, r in losers.iterrows():
            print(f"    {r['label']}: eq {r['eq_beat']:.0%} vs single {r['sg_beat']:.0%} vs ic {r['ic_beat']:.0%}")

    print("\n  ———— 判定 ————")
    if over50 >= 0.8 and eq_ge_single >= 0.6:
        print("  → 稳健：等权合成在绝大多数窗口/再平衡/分层设置下都跑赢基准且不输单因子，")
        print("     M14 的 83% 不是单组超参的运气，是结构性优势。可进入下一步（港股/美股验证）。")
    elif over50 >= 0.6:
        print("  → 基本稳健：多数设置下等权占优，但对某些超参敏感（见上方列表），")
        print("     需注意这些设置的共性（如分层数/再平衡频率）。")
    else:
        print("  → 不稳健：等权优势高度依赖特定设置，M14 结论需打折，先排查再推进。")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
