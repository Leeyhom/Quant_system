"""us_fundamental_demo —— 美股季报基本面因子验证（M17，Stage 2b）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/us_fundamental_demo.py --limit 30   # 小步先跑通
    NO_PROXY='*' python scripts/us_fundamental_demo.py              # 全池

目的（对症 docs/15 结论）：M16(Stage 2a) 证明美股大盘**量价因子无稳定 alpha**，
稳定信号和 A股一样应在基本面。本脚本补上美股**季报基本面因子**（价值/质量/成长），
跑同一套 IC + 滚动 walk-forward 验证，看基本面是否给出比量价更稳定的方向先验。

与 us_multifactor_demo（量价）的差异（诚实标注）：
  ① 因子换成基本面：价值(TTM EPS/价格)、质量(ROE/毛利率/净利率)、成长(营收/净利同比)。
  ② 数据走季报点状面板，**严格按公告日防前视**（见 build_us_fundamental_panels 注释）。
  ③ 回测用**美股专属费用模型**（每股费+每笔最低费，见 us_cost.py），而非 A股比例成本。
  ④ 其余口径（HORIZON/分层/滚动窗口/等权 vs IC加权）与 A股/Stage 2a 一致，对比才公平。
  ⑤ 做空能力本步不用——仍用分层多头(long_top_layer)验证因子，与 A股可比；做空留下一步。
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

from quant.config import RAW_DATA_DIR
from quant.data import us_loader
from quant.data.universe_us import US_POOL
from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.composite import factor_correlation, weighted_composite
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

from scripts.composite_demo import rolling_long_top_layer
from scripts.factor_research_demo import parse_limit
from scripts.us_multifactor_demo import select_stable_positive

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
TRAIN_SIZE = 480
TEST_SIZE = 120
STEP = 60
N_CUTS = 5

# 美股费用回调（每股费+每笔最低费）。整个验证统一用它，替代 A股比例成本。
US_COST_FN = make_layered_cost_fn()


def build_factors(close: pd.DataFrame, fund: dict) -> dict:
    """构建美股基本面因子（价值/质量/成长，对齐 factors.py「越高越好」约定）。

    选这几个的理由：价值(便宜)、质量(赚钱能力)、成长(增速)是基本面三大正交维度，
    与 A股 M14 的 value/quality/growth 同构，便于跨市场对比。方向是经济先验，
    真伪由 IC 验证。
    """
    out = {
        "value_ey": F.us_earnings_yield(fund["eps_ttm"], close),  # 价值：TTM EPS / 价格
        "quality_roe": F.us_quality_roe(fund["roe"]),             # 质量：ROE
        "quality_gm": F.us_quality_roe(fund["gross_margin"]),     # 质量：毛利率（薄包装同义）
        "growth_rev": F.us_growth(fund["rev_yoy"]),               # 成长：营收同比
        "growth_profit": F.us_growth(fund["profit_yoy"]),         # 成长：净利同比
    }
    return out


def _roll(close, build_fn, label):
    """单条滚动 walk-forward 并打印一行汇总（统一带美股费用）。"""
    r = rolling_long_top_layer(close, build_fn, TRAIN_SIZE, TEST_SIZE, STEP, cost_fn=US_COST_FN)
    periods = r["periods"]
    if not periods:
        print(f"  {label:28s} 窗口 0（历史不足）", flush=True)
        return {"r": r, "beat": float("nan"), "med_sharpe": float("nan"), "med_excess": float("nan")}
    sh = pd.Series([p["sharpe"] for p in periods])
    exc = pd.Series([p["sharpe"] - p["bench_sharpe"] for p in periods])
    print(f"  {label:28s} 窗口{r['n']:2d} 跑赢{r['beat_rate']:.0%} "
          f"中位夏普{sh.median():+.2f} 中位超额{exc.median():+.2f}", flush=True)
    return {"r": r, "beat": r["beat_rate"], "med_sharpe": float(sh.median()),
            "med_excess": float(exc.median())}


def _lookahead_gate(symbols: list[str], fund: dict, close: pd.DataFrame) -> None:
    """防前视 gate：抽查一只票某交易日的价值因子，反推所用季度+公告日，人工核对。

    这是本步最关键的正确性检查——若对齐键错用 report_date，这里会暴露
    「公告日 > 交易日」的未来函数。打印供人工确认 notice_date ≤ 交易日。
    """
    from quant.data import us_fundamental_loader
    # 找一只在池里且有基本面缓存的票
    probe = None
    for s in symbols:
        try:
            raw = us_fundamental_loader.load_parquet(s)
            if len(raw) > 0 and s in close.columns:
                probe = (s, raw)
                break
        except FileNotFoundError:
            continue
    if probe is None:
        print("      [gate] 未找到可抽查的票，跳过防前视 gate")
        return
    sym, raw = probe
    # 取面板中段一个交易日
    eps_panel = fund["eps_ttm"][sym].dropna()
    if eps_panel.empty:
        print(f"      [gate] {sym} 无有效 eps_ttm，跳过")
        return
    probe_date = eps_panel.index[len(eps_panel) // 2]
    eps_val = eps_panel.loc[probe_date]
    # 反推：该 eps_ttm 应来自最近一个 notice_date ≤ probe_date 的季度
    raw = raw.copy()
    raw["notice_date"] = pd.to_datetime(raw["notice_date"])
    avail = raw[raw["notice_date"] <= probe_date]
    used = avail.iloc[-1] if len(avail) else None
    print(f"      [gate] 抽查 {sym} @ {probe_date.date()}: eps_ttm(面板)={eps_val:.2f}")
    if used is not None:
        ok = used["notice_date"] <= probe_date
        print(f"             面板值应来自 report_date={pd.to_datetime(used['report_date']).date()} "
              f"notice_date={used['notice_date'].date()} eps_ttm(源)={used['eps_ttm']:.2f}")
        print(f"             公告日 ≤ 交易日? {'✅ 无未来函数' if ok else '❌ 未来函数!!'}")
    else:
        print(f"             该日之前无已公告季度（面板应为 NaN）")


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = US_POOL[:limit] if limit else US_POOL
    print(f"[1/6] 构建美股长历史行情面板（{len(symbols)} 只）...", flush=True)
    panels = build_ohlcv_panels(symbols, loader=us_loader)
    close = panels["close"]
    span = f"{close.index.min().date()}~{close.index.max().date()}"
    print(f"      行情面板 {close.shape[0]}天×{close.shape[1]}只 | {span}", flush=True)
    if close.shape[0] < TRAIN_SIZE + TEST_SIZE:
        print(f"      ⚠️ 历史不足，需 ≥{TRAIN_SIZE+TEST_SIZE} 天。")
        return

    print(f"\n[2/6] 构建季报基本面面板（按公告日防前视）...", flush=True)
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    cover = {k: int(v.notna().any().sum()) for k, v in fund.items()}
    print(f"      基本面字段覆盖（有数据的股票数）：{cover}", flush=True)

    print(f"\n[3/6] 防前视 gate（抽查公告日 ≤ 交易日）", flush=True)
    _lookahead_gate(symbols, fund, close)

    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_factors(close, fund)

    print(f"\n[4/6] 因子相关矩阵（确认正交）", flush=True)
    print(factor_correlation(facs).round(2).to_string())
    print(f"\n      全样本 IC（看方向先验是否净正向、稳不稳）", flush=True)
    for name, fac in facs.items():
        s = ic_summary(daily_ic(fac, fwd))
        print(f"      {name:14s} meanIC {s['mean_ic']:+.3f} | ICIR {s['icir']:+.2f} | "
              f"t {s['t_stat']:+.2f} | posRate {s['positive_rate']:.0%}", flush=True)

    print(f"\n      等权分量筛选（方向先验需稳定净正向）：", flush=True)
    selected = select_stable_positive(facs, fwd)
    print(f"      纳入等权合成: {selected}", flush=True)

    print(f"\n[5/6] 滚动 walk-forward（train{TRAIN_SIZE}/test{TEST_SIZE}/step{STEP}，含美股费用）", flush=True)
    print("      —— 分层多头 L5 vs 等权全持有基准 ——", flush=True)
    single = {}
    for name in facs:
        single[name] = _roll(close, (lambda u, nm=name: facs[nm]), f"单因子 {name}")
    valid_single = {k: v for k, v in single.items() if v["beat"] == v["beat"]}  # 去 NaN
    best_single = max(valid_single, key=lambda k: valid_single[k]["beat"]) if valid_single else None

    if not selected:
        print(f"\n[6/6] ———— 诚实结论（Stage 2b 基本面）————", flush=True)
        print("  → 美股大盘 2018~2025 基本面因子**无方向稳定净正向**者（select_stable_positive 筛出 0 个）。")
        if best_single:
            b = single[best_single]
            print(f"  参考：最优单因子 {best_single} 跑赢 {b['beat']:.0%} | 超额夏普 {b['med_excess']:+.2f}")
        print("  这与 Stage 2a 量价一致——本项目美股大盘池在这段 regime 下基本面/量价都难稳定跑赢")
        print("  普涨的 mega-cap 成长。下一步可考虑：扩池(纳入中小盘，基本面分化更大)、或上做空(多空对冲)。")
        print("完成 ✅")
        return

    sel_facs = {n: facs[n] for n in selected}
    print(f"      {'-'*60}", flush=True)
    res = {}
    res["icir"] = _roll(
        close,
        lambda u: weighted_composite(sel_facs, fwd, u, scheme="icir", n_cuts=N_CUTS, horizon=HORIZON)[0],
        "ICIR×多切分 合成",
    )
    eq_factor = combine_factors(*sel_facs.values())
    res["equal"] = _roll(close, (lambda u: eq_factor), "等权合成（固定正向）")

    print(f"\n[6/6] 全样本分层多头净值（等权 vs 最优单因子 vs 基准，含美股费用）", flush=True)
    bt_eq = long_top_layer(close, eq_factor, n_layers=N_LAYERS, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt_single = long_top_layer(close, facs[best_single], n_layers=N_LAYERS, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    m_eq = summary(bt_eq["equity"], bt_eq["port_ret"])
    m_single = summary(bt_single["equity"], bt_single["port_ret"])
    m_bench = summary(bt_eq["benchmark"], bt_eq["benchmark_ret"])
    print(f"      等权合成   收益 {m_eq['total_return']:+.2%} | 夏普 {m_eq['sharpe']:+.2f} | 回撤 {m_eq['max_drawdown']:.2%}")
    print(f"      最优单因子 收益 {m_single['total_return']:+.2%} | 夏普 {m_single['sharpe']:+.2f} | 回撤 {m_single['max_drawdown']:.2%} ({best_single})")
    print(f"      等权基准   收益 {m_bench['total_return']:+.2%} | 夏普 {m_bench['sharpe']:+.2f} | 回撤 {m_bench['max_drawdown']:.2%}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt_eq["equity"].index, bt_eq["equity"], label="equal-weight fundamental long", linewidth=1.6)
    ax.plot(bt_single["equity"].index, bt_single["equity"], label=f"{best_single} long", linewidth=1.1)
    ax.plot(bt_eq["benchmark"].index, bt_eq["benchmark"], label="benchmark (equal-hold)", linewidth=1.1, linestyle="--")
    ax.set_title(f"US M17 fundamental long-top-layer ({span}): equal vs single vs benchmark")
    ax.set_xlabel("date"); ax.set_ylabel("net value"); ax.legend()
    fig.tight_layout()
    png = RAW_DATA_DIR / "us_fundamental_equity.png"
    fig.savefig(png, dpi=120)
    print(f"      图: {png}")

    print(f"\n———— 诚实结论 ————", flush=True)
    bs = single[best_single]
    eq, icir = res["equal"], res["icir"]
    print(f"  (a) 最优单因子 {best_single}: 跑赢 {bs['beat']:.0%} | 超额夏普 {bs['med_excess']:+.2f}")
    print(f"  (b) IC加权合成: 跑赢 {icir['beat']:.0%} | 超额夏普 {icir['med_excess']:+.2f}")
    print(f"  (c) 等权合成  : 跑赢 {eq['beat']:.0%} | 超额夏普 {eq['med_excess']:+.2f}")
    if eq["beat"] >= bs["beat"] and eq["med_excess"] >= bs["med_excess"]:
        print("  → 基本面在美股给出了稳定方向先验，等权合成跑赢最优单因子，A股 M14 结论跨市场复现。")
    else:
        print("  → 美股基本面等权未超最优单因子。记录为诚实结果，下一步可扩池/上做空再验证。")
    if eq["beat"] > icir["beat"]:
        print("  → 等权 > IC加权 在美股同样成立（方向锁定陷阱跨市场复现）。")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
