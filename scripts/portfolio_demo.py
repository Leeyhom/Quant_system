"""portfolio_demo —— E+F：横截面因子选股 + 组合资金管理。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/portfolio_demo.py

研究流程：
  1) 构建10只股票的收盘价面板。
  2) 比较多个候选因子/参数组合（动量、反转、低波、多因子）。
  3) 选择夏普最高的组合，和「等权持有全部股票」基准画净值曲线。

注意：这仍是样本内研究，下一步应做样本外/滚动验证，不能直接实盘。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant.config import RAW_DATA_DIR
from quant.data.panel import build_close_panel
from quant.factor.factors import momentum, reversal, low_volatility, combine_factors
from quant.backtest.portfolio import run_factor_portfolio
from quant.backtest.metrics import summary

SYMBOLS = [
    "600519", "000001", "600036", "601318", "000651",
    "600276", "002415", "600900", "601012", "000333",
]


def _fmt(name: str, m: dict) -> str:
    return (
        f"{name:<16} 总收益 {m['total_return']:+.2%} | 年化 {m['annualized_return']:+.2%} | "
        f"最大回撤 {m['max_drawdown']:.2%} | 夏普 {m['sharpe']:.2f}"
    )


def main() -> None:
    print(f"[1/4] 构建 {len(SYMBOLS)} 只股票的收盘价面板 ...")
    close = build_close_panel(SYMBOLS, start="20240101", end="20251231")
    print(f"      面板形状: {close.shape[0]} 天 × {close.shape[1]} 只")

    print("[2/4] 计算候选因子 ...")
    factor_defs = {
        "momentum20": momentum(close, window=20),
        "momentum60": momentum(close, window=60),
        "reversal20": reversal(close, window=20),
        "lowvol20": low_volatility(close, window=20),
        "multi_rank": combine_factors(
            momentum(close, window=60),
            reversal(close, window=20),
            low_volatility(close, window=20),
        ),
    }

    # 候选组合：因子 × 持股数量 × 再平衡周期。
    # 不追求穷举，只演示真实研究中「横向比较候选」的流程。
    candidates = []
    for factor_name, factor in factor_defs.items():
        for top_n in (3, 5):
            for rebalance_every in (10, 20, 40):
                bt = run_factor_portfolio(
                    close,
                    factor,
                    top_n=top_n,
                    rebalance_every=rebalance_every,
                    cost_rate=0.001,
                )
                m = summary(bt["equity"], bt["port_ret"])
                candidates.append({
                    "factor": factor_name,
                    "top_n": top_n,
                    "rebalance_every": rebalance_every,
                    "metrics": m,
                    "bt": bt,
                })

    best = max(candidates, key=lambda x: x["metrics"]["sharpe"])
    bench_bt = best["bt"]
    bench = summary(bench_bt["benchmark"], bench_bt["benchmark"].pct_change().fillna(0.0))

    print("[3/4] 候选组合 Top 8（按夏普排序）...")
    for c in sorted(candidates, key=lambda x: x["metrics"]["sharpe"], reverse=True)[:8]:
        m = c["metrics"]
        print(
            f"      {c['factor']:<10} top{c['top_n']} rb{c['rebalance_every']:<2} -> "
            f"总收益 {m['total_return']:+.2%} | 夏普 {m['sharpe']:.2f} | 回撤 {m['max_drawdown']:.2%}"
        )
    print("      " + _fmt("等权全持有基准", bench))
    print(
        f"\n      最佳候选: {best['factor']} / top{best['top_n']} / "
        f"每{best['rebalance_every']}日再平衡"
    )
    print("      " + _fmt("最佳因子组合", best["metrics"]))

    print("[4/4] 画最佳候选 vs 基准净值曲线 ...")
    bt = best["bt"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt.index, bt["equity"], label="best factor portfolio")
    ax.plot(bt.index, bt["benchmark"], label="equal-weight all", alpha=0.7)
    ax.set_title(
        f"best: {best['factor']} top{best['top_n']} rb{best['rebalance_every']}"
    )
    ax.set_xlabel("date")
    ax.set_ylabel("net value (start=1)")
    ax.legend()
    fig.tight_layout()
    out_png = RAW_DATA_DIR / "portfolio_equity.png"
    fig.savefig(out_png, dpi=120)
    print(f"      图已保存：{out_png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
