"""us_expanded_demo —— 美股扩展池因子验证（Step 2/3，Stage 2c 前半）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/us_expanded_demo.py            # 全扩展池
    NO_PROXY='*' python scripts/us_expanded_demo.py --limit 100 # 小步测试

目的（对症 M17 的「方向先验漂移」诊断）：
    M17 基线：~72 只有基本面数据的大盘池，select_stable_positive 只筛出 1 个
    稳定因子( growth_rev )，等权合成 = 单因子，无法分散。

    本脚本用扩展池（S&P 500 + S&P 400，目标 300+ 只有效基本面）验证两个假设：

    假设 1（扩池效应）：更多股票 → 截面分化更大 → 价值/质量/成长信号更分明
              → select_stable_positive 筛出 ≥3 个稳定因子。
    假设 2（行业中性化效应）：剥离行业 beta 后因子 ICIR 提升（A股 M12 的
              美股复现：ICIR ↑、方向更稳）。

与 us_fundamental_demo.py（M17 基线）的差异：
    ① 池子从 US_POOL(81) → EXPANDED_US_POOL(500+)。
    ② 因子库新增 net_margin（盈利质量）、debt_ratio（杠杆风险）——
       这两个字段已在 us_fundamental_loader 加载但从未被使用。
    ③ 可选行业中性化（--neutralize），用 industry_us.py 的手工 GICS 映射。
    ④ 其余口径（HORIZON/分层/滚动窗口/等权vsIC加权/美股费用）完全一致。
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
from quant.factor.composite import factor_correlation, weighted_composite
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

from scripts.composite_demo import rolling_long_top_layer
from scripts.us_multifactor_demo import select_stable_positive

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
TRAIN_SIZE = 480
TEST_SIZE = 120
STEP = 60
N_CUTS = 5

US_COST_FN = make_layered_cost_fn()


def parse_limit(argv: list[str]) -> int | None:
    """支持 --limit N 参数控制池子大小（先小步跑通）。"""
    for i, a in enumerate(argv):
        if a == "--limit" and i + 1 < len(argv):
            return int(argv[i + 1])
    return None


def _has_flag(argv, flag):
    return flag in argv


def build_factors(close: pd.DataFrame, fund: dict) -> dict:
    """构建扩展基本面因子库（M17 5 个 + 新增 2 个 = 7 个正交因子）。

    新增因子的经济含义：
      - net_margin（净利率）：盈利质量维度——同样营收，扣完所有成本后剩多少。
        与 ROE（权益视角）互补：ROE = 净利/权益，net_margin = 净利/营收。
      - debt_ratio（资产负债率）：杠杆/风险维度——高杠杆公司在市场压力下
        更脆弱。取负使「低杠杆=高分」，方向是先验，真伪由 IC 验证。
    """
    out = {
        # M17 原有 5 因子
        "value_ey": F.us_earnings_yield(fund["eps_ttm"], close),
        "quality_roe": F.us_quality_roe(fund["roe"]),
        "quality_gm": F.us_quality_roe(fund["gross_margin"]),
        "growth_rev": F.us_growth(fund["rev_yoy"]),
        "growth_profit": F.us_growth(fund["profit_yoy"]),
    }
    # 新增因子（从已加载数据中零成本导出）
    if "net_margin" in fund:
        out["quality_nm"] = F.us_quality_roe(fund["net_margin"])  # 净利率（越高越好）
    if "debt_ratio" in fund:
        # 负债率越低越好 → 取负统一方向
        out["safety_debt"] = -fund["debt_ratio"]
    return out


def _roll(close, build_fn, label, cost_fn=US_COST_FN):
    """单条滚动 walk-forward 并打印汇总。"""
    r = rolling_long_top_layer(close, build_fn, TRAIN_SIZE, TEST_SIZE, STEP, cost_fn=cost_fn)
    periods = r["periods"]
    if not periods:
        print(f"  {label:32s} 窗口 0（历史不足）", flush=True)
        return {"r": r, "beat": float("nan"), "med_sharpe": float("nan"), "med_excess": float("nan")}
    sh = pd.Series([p["sharpe"] for p in periods])
    exc = pd.Series([p["sharpe"] - p["bench_sharpe"] for p in periods])
    print(f"  {label:32s} 窗口{r['n']:2d} 跑赢{r['beat_rate']:.0%} "
          f"中位夏普{sh.median():+.2f} 中位超额{exc.median():+.2f}", flush=True)
    return {"r": r, "beat": r["beat_rate"], "med_sharpe": float(sh.median()),
            "med_excess": float(exc.median())}


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    do_neutralize = _has_flag(argv, "--neutralize")

    symbols = EXPANDED_US_POOL[:limit] if limit else EXPANDED_US_POOL
    limit_str = f"{len(symbols)} 只" if limit else "全池"
    neut_str = "含行业中性化" if do_neutralize else "无中性化（原始因子）"
    print(f"[1/5] 构建美股长历史行情面板（{limit_str}）... {neut_str}", flush=True)
    panels = build_ohlcv_panels(symbols, loader=us_loader)
    close = panels["close"]
    span = f"{close.index.min().date()}~{close.index.max().date()}"
    n_stocks_with_data = close.notna().any().sum()
    print(f"      行情面板 {close.shape[0]}天×{close.shape[1]}只（有数据 {n_stocks_with_data} 只）| {span}", flush=True)
    if close.shape[0] < TRAIN_SIZE + TEST_SIZE:
        print(f"      ⚠️ 历史不足，需 ≥{TRAIN_SIZE+TEST_SIZE} 天。")
        return

    print(f"\n[2/5] 构建季报基本面面板（按公告日防前视）...", flush=True)
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    cover = {k: int(v.notna().any().sum()) for k, v in fund.items()}
    print(f"      基本面字段覆盖（有数据的股票数）：{cover}", flush=True)

    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_factors(close, fund)

    # —— 可选行业中性化 ——
    if do_neutralize:
        from quant.data.industry_us import industry_series
        from quant.factor.neutralize import neutralize as neut_fn
        ind = industry_series(list(close.columns))
        print(f"      行业分布: {ind.value_counts().to_dict()}", flush=True)
        facs = {n: neut_fn(f, industry=ind, mode="industry") for n, f in facs.items()}

    print(f"\n[3/5] 因子相关矩阵（确认正交）", flush=True)
    corr_mat = factor_correlation(facs)
    print(corr_mat.round(2).to_string())
    print(f"\n      全样本 IC（看方向先验是否净正向。池子大小改变后，这是关键诊断）", flush=True)
    for name, fac in facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        print(f"      {name:14s} meanIC {s['mean_ic']:+.3f} | ICIR {s['icir']:+.2f} | "
              f"t {s['t_stat']:+.2f} | posRate {s['positive_rate']:.0%}", flush=True)

    print(f"\n      等权分量筛选（方向先验需稳定净正向）：", flush=True)
    selected = select_stable_positive(facs, fwd)
    print(f"      纳入等权合成: {selected}（共 {len(selected)} 个）", flush=True)

    # —— 单因子滚动验证 ——
    print(f"\n[4/5] 滚动 walk-forward（train{TRAIN_SIZE}/test{TEST_SIZE}/step{STEP}，含美股费用）", flush=True)
    print("      —— 分层多头 L5 vs 等权全持有基准 ——", flush=True)
    single = {}
    for name in facs:
        single[name] = _roll(close, (lambda u, nm=name: facs[nm]), f"单因子 {name}")
    valid_single = {k: v for k, v in single.items() if v["beat"] == v["beat"]}
    best_single = max(valid_single, key=lambda k: valid_single[k]["beat"]) if valid_single else None

    if not selected:
        print(f"\n[5/5] ———— 诚实结论 ————", flush=True)
        print(f"  → 扩展池（{n_stocks_with_data} 只有行情数据）基本面因子仍无方向稳定净正向者。")
        if best_single:
            b = single[best_single]
            print(f"  参考：最优单因子 {best_single} 跑赢 {b['beat']:.0%} | 超额夏普 {b['med_excess']:+.2f}")
        print("完成 ✅")
        return

    sel_facs = {n: facs[n] for n in selected}
    print(f"      {'-'*60}", flush=True)

    # —— 等权合成 ——
    eq_factor = combine_factors(*sel_facs.values())
    res = {}
    res["equal"] = _roll(close, (lambda u: eq_factor), "等权合成（固定正向）")

    # —— IC加权合成 ——
    if len(sel_facs) >= 2:
        res["icir"] = _roll(
            close,
            lambda u: weighted_composite(sel_facs, fwd, u, scheme="icir", n_cuts=N_CUTS, horizon=HORIZON)[0],
            "ICIR×多切分 合成",
        )

    # —— 全样本分层多头净值 ——
    print(f"\n[5/5] 全样本分层多头净值（等权 vs 最优单因子 vs 基准，含美股费用）", flush=True)
    bt_eq = long_top_layer(close, eq_factor, n_layers=N_LAYERS, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt_single = long_top_layer(close, facs[best_single], n_layers=N_LAYERS, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    m_eq = summary(bt_eq["equity"], bt_eq["port_ret"])
    m_single = summary(bt_single["equity"], bt_single["port_ret"])
    m_bench = summary(bt_eq["benchmark"], bt_eq["benchmark_ret"])
    print(f"      等权合成 ({len(selected)}因子) 收益 {m_eq['total_return']:+.2%} | 夏普 {m_eq['sharpe']:+.2f} | 回撤 {m_eq['max_drawdown']:.2%}")
    print(f"      最优单因子 {best_single:14s} 收益 {m_single['total_return']:+.2%} | 夏普 {m_single['sharpe']:+.2f} | 回撤 {m_single['max_drawdown']:.2%}")
    print(f"      等权基准          收益 {m_bench['total_return']:+.2%} | 夏普 {m_bench['sharpe']:+.2f} | 回撤 {m_bench['max_drawdown']:.2%}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt_eq["equity"].index, bt_eq["equity"], label=f"equal-weight ({len(selected)} factors)", linewidth=1.6)
    ax.plot(bt_single["equity"].index, bt_single["equity"], label=f"best single ({best_single})", linewidth=1.1)
    ax.plot(bt_eq["benchmark"].index, bt_eq["benchmark"], label="benchmark (equal-hold)", linewidth=1.1, linestyle="--")
    ax.set_title(f"US expanded pool ({n_stocks_with_data} stocks, {span}): equal vs single vs benchmark")
    ax.set_xlabel("date"); ax.set_ylabel("net value"); ax.legend()
    fig.tight_layout()
    png = RAW_DATA_DIR / "us_expanded_equity.png"
    fig.savefig(png, dpi=120)
    print(f"      图: {png}")

    # —— 总结 ——
    print(f"\n———— 诚实结论 ————", flush=True)
    if best_single:
        bs = single[best_single]
        print(f"  (a) 最优单因子 {best_single}: 跑赢 {bs['beat']:.0%} | 超额夏普 {bs['med_excess']:+.2f}")
    eq = res["equal"]
    print(f"  (c) 等权合成 ({len(selected)}因子): 跑赢 {eq['beat']:.0%} | 超额夏普 {eq['med_excess']:+.2f}")
    if "icir" in res:
        icir = res["icir"]
        print(f"  (b) IC加权合成 : 跑赢 {icir['beat']:.0%} | 超额夏普 {icir['med_excess']:+.2f}")

    # 与 M17 基线对比
    print(f"\n  —— 与 M17 基线对比 ——")
    print(f"  M17 基线(72只): 等权=50% | 最优单因子(growth_profit)=71% | 稳定因子数=1")
    print(f"  本次扩展池({n_stocks_with_data}只): 等权={eq['beat']:.0%} | "
          f"最优单因子({best_single})={single[best_single]['beat']:.0%} | "
          f"稳定因子数={len(selected)}")
    if len(selected) >= 3:
        print(f"  ✅ 假设1成立：扩池后稳定因子从 1 → {len(selected)}，等权分散有望。")
    elif len(selected) >= 2:
        print(f"  ⚠️ 部分改善：稳定因子从 1 → 2。有进步但分散仍不足。")
    else:
        print(f"  → 扩池未增加稳定因子数。方向漂移可能不是池子规模问题，需上多空对冲(Step 4)。")
    print(f"完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
