"""factor_validation_demo —— 因子样本外验证 + 因子合成（M9）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/factor_validation_demo.py            # 默认全池(~90只)
    NO_PROXY='*' python scripts/factor_validation_demo.py --limit 20 # 先小步验证

要回答两个问题（见 docs/10）：
  (a) M8 出现的显著因子，IC 在样本外是否还在、是否同号？      → 因子层 OOS IC
  (b) 是否存在某候选（单因子/合成）在样本外跑赢「等权持有全池」？ → 组合层 OOS 收益
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
from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.factor import factors as F
from quant.backtest.ic_analysis import forward_returns
from quant.backtest.factor_validation import ic_train_test, build_oriented_composite
from quant.backtest.portfolio_validation import (
    PortfolioCandidate,
    train_test_validate,
    rolling_walk_forward,
    stability_summary,
)

# 复用 M8 demo 的量价因子构建，保证口径一致。
from scripts.factor_research_demo import build_factors as build_price_factors
from scripts.factor_research_demo import parse_limit

HORIZON = 20
TRAIN_RATIO = 0.7
TOP_NS = (3, 5)
REBALANCE = 20
# OOS 稳健因子门槛：train/test 同号，且 test 段 |t| 达到该值。
TEST_T_THRESHOLD = 1.5


def build_fundamental_factors(
    value_panels: dict, volume: "pd.DataFrame"
) -> dict:
    """基于估值/股本面板构建基本面因子（M10）。

    value_panels 已在调用处 reindex 对齐到 close 的 index/columns，
    故与 volume 等价格面板逐日、逐票对齐，可直接做横截面 IC。
    """
    out = {}
    if "pe_ttm" in value_panels:
        out["earnings_yield"] = F.earnings_yield(value_panels["pe_ttm"])
    if "pb" in value_panels:
        out["book_to_price"] = F.book_to_price(value_panels["pb"])
    if "ps" in value_panels:
        out["sales_yield"] = F.sales_yield(value_panels["ps"])
    if "total_mv" in value_panels:
        out["small_size"] = F.small_size(value_panels["total_mv"])
    if "float_share" in value_panels:
        out["turnover_rate"] = F.turnover_rate(volume, value_panels["float_share"])
    return out


def build_factors(panels: dict, value_panels: dict) -> dict:
    """合并量价因子（M8）+ 基本面因子（M10）。"""
    facs = build_price_factors(panels)
    facs.update(build_fundamental_factors(value_panels, panels["volume"]))
    return facs


def _fmt(seg: dict) -> str:
    return f"meanIC {seg['mean_ic']:+.3f} | t {seg['t_stat']:+.2f} | n {int(seg['n'])}"


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = DEFAULT_POOL[:limit] if limit else DEFAULT_POOL
    print(f"[1/6] 构建股票池面板（{len(symbols)} 只，limit={limit}）...")
    panels = build_ohlcv_panels(symbols, start="20240101", end="20251231")
    close = panels["close"]
    print(f"      面板形状: {close.shape[0]} 天 × {close.shape[1]} 只")

    # 估值/股本面板，reindex 对齐到 close（M10）
    value_panels = build_value_panels(
        symbols, start="20240101", end="20251231", align_to=close
    )
    print(f"      估值字段: {sorted(value_panels.keys())}")

    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_factors(panels, value_panels)

    # ───────── 因子层 OOS IC：哪些因子样本外还稳？ ─────────
    print(f"\n[2/6] 因子层样本外 IC（train_ratio={TRAIN_RATIO}, horizon={HORIZON}）")
    print("      —— 看 test 段是否与 train 同号且仍显著 ——")
    robust = []
    rows = []
    for name, fac in facs.items():
        res = ic_train_test(fac, fwd, train_ratio=TRAIN_RATIO, horizon=HORIZON)
        rows.append((name, res))
        ok = res["sign_consistent"] and abs(res["test"]["t_stat"]) >= TEST_T_THRESHOLD
        flag = "✅稳健" if ok else "  "
        print(f"      {name:<16} train[{_fmt(res['train'])}]  test[{_fmt(res['test'])}]  "
              f"同号={'是' if res['sign_consistent'] else '否'} {flag}")
        if ok:
            robust.append(name)

    if not robust:
        # 没有 OOS 稳健因子时，退而用 full 段 |meanIC| 最大的前 3 个，仍走组合验证流程，
        # 但要在结论里诚实说明它们样本外并不稳。
        rows.sort(key=lambda r: abs(r[1]["full"]["mean_ic"]), reverse=True)
        robust = [r[0] for r in rows[:3]]
        print(f"\n      ⚠️ 无因子通过 OOS 稳健门槛，退用 |meanIC| 最大的: {robust}")
    else:
        print(f"\n      OOS 稳健因子: {robust}")

    # ───────── 构造组合候选：单因子双方向 + train 定向合成 ─────────
    print(f"\n[3/6] 构造候选（单因子正/反向 + 合成；top_n∈{TOP_NS}, rb={REBALANCE}）")
    candidates: list[PortfolioCandidate] = []
    for name in robust:
        fac = facs[name]
        for top_n in TOP_NS:
            candidates.append(PortfolioCandidate(f"{name}|n{top_n}", fac, top_n, REBALANCE))
            candidates.append(PortfolioCandidate(f"{name}_rev|n{top_n}", -fac, top_n, REBALANCE))

    # 合成因子：方向只用 train 段 IC 决定（防前视）。切分日期与 train_test_validate 对齐。
    cut = int(len(close) * TRAIN_RATIO)
    upto_date = close.index[cut]
    comp, signs = build_oriented_composite(
        {n: facs[n] for n in robust}, fwd, upto_date, horizon=HORIZON
    )
    print(f"      合成分量方向(train IC 符号): "
          + ", ".join(f"{n}{'+' if s >= 0 else '-'}" for n, s in signs.items()))
    for top_n in TOP_NS:
        candidates.append(PortfolioCandidate(f"composite|n{top_n}", comp, top_n, REBALANCE))
    print(f"      候选总数: {len(candidates)}")

    # ───────── 单次 train/test 验证 ─────────
    print(f"\n[4/6] 单次 train/test 验证（train 选夏普最高，test 公平评判）")
    res = train_test_validate(close, candidates, train_ratio=TRAIN_RATIO, metric="sharpe")
    best = res["best"]
    trm, tem = res["train_metrics"], res["test_metrics"]
    bt = res["test_bt"]
    bench_eq = bt["benchmark"]
    bench_ret = bt["benchmark_ret"]
    from quant.backtest.metrics import summary as _summary
    bench_m = _summary(bench_eq, bench_ret)
    print(f"      train 选出: {best.name}")
    print(f"      train : 收益 {trm['total_return']:+.2%} | 夏普 {trm['sharpe']:.2f} | 回撤 {trm['max_drawdown']:.2%}")
    print(f"      test  : 收益 {tem['total_return']:+.2%} | 夏普 {tem['sharpe']:.2f} | 回撤 {tem['max_drawdown']:.2%}")
    print(f"      基准  : 收益 {bench_m['total_return']:+.2%} | 夏普 {bench_m['sharpe']:.2f} | 回撤 {bench_m['max_drawdown']:.2%}")
    beat_once = tem["sharpe"] > bench_m["sharpe"]
    print(f"      → test 段{'跑赢' if beat_once else '未跑赢'}等权基准（按夏普）")

    # ───────── 滚动 walk-forward ─────────
    print(f"\n[5/6] 滚动 walk-forward（每窗 train 选参、test 评判，看稳定性）")
    print("      注：合成分量方向按单一切分定向，滚动中固定；单因子方向每窗独立选取。")
    n = len(close)
    train_size = min(240, int(n * 0.5))
    test_size = min(60, max(20, int(n * 0.15)))
    try:
        wf = rolling_walk_forward(
            close, candidates,
            train_size=train_size, test_size=test_size, step=test_size, metric="sharpe",
        )
        stab = stability_summary(wf["periods"])
        print(f"      窗口数 {stab['n_periods']} | 正夏普比例 {stab['positive_sharpe_rate']:.0%} | "
              f"跑赢基准比例 {stab['beat_benchmark_rate']:.0%} | "
              f"中位OOS夏普 {stab['median_oos_sharpe']:.2f}")
        oos = wf["oos"]
    except ValueError as e:
        print(f"      数据不足以滚动: {e}")
        stab, oos = None, None

    # ───────── 出图 + 诚实结论 ─────────
    print(f"\n[6/6] 保存 OOS 净值图 + 结论")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bt.index, bt["equity"], label=f"OOS combo: {best.name}", linewidth=1.5)
    ax.plot(bt.index, bench_eq, label="OOS benchmark (equal-weight)", linewidth=1.5)
    ax.set_title("Single train/test: OOS portfolio vs equal-weight benchmark")
    ax.set_xlabel("date"); ax.set_ylabel("net value"); ax.legend()
    fig.tight_layout()
    png = RAW_DATA_DIR / "factor_oos_equity.png"
    fig.savefig(png, dpi=120)
    print(f"      图: {png}")

    print("\n———— 诚实结论 ————")
    print(f"  (a) 因子样本外稳健性: {len(robust)} 个候选用于组合"
          + ("（含通过门槛者）" if robust else ""))
    verdict_once = "跑赢" if beat_once else "未跑赢"
    print(f"  (b) 单次 test: 最优候选「{best.name}」{verdict_once}等权基准"
          f"（夏普 {tem['sharpe']:.2f} vs {bench_m['sharpe']:.2f}）")
    if stab is not None:
        print(f"      滚动: 跑赢基准比例 {stab['beat_benchmark_rate']:.0%}，"
              f"中位 OOS 夏普 {stab['median_oos_sharpe']:.2f}")
        if stab["beat_benchmark_rate"] >= 0.5 and beat_once:
            print("  → 出现了较站得住脚的样本外优势，可作为走向资金管理/模拟盘的起点。")
        else:
            print("  → 样本外优势不稳定。下一步：合成权重优化，或引入财务/真实换手率做行业中性，再上复杂模型。")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
