"""portfolio_walkforward_demo —— M6：因子组合样本外 / 滚动 walk-forward 验证。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/portfolio_walkforward_demo.py

严谨性：
  - 每个窗口只用 train 段选择候选组合；
  - test 段只验证，不再调参；
  - test 首日用 train 末日因子建仓，避免空仓等待造成的人为偏差；
  - 输出样本外拼接净值和每期稳定性。
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
from quant.backtest.metrics import summary
from quant.backtest.portfolio_validation import (
    PortfolioCandidate,
    rolling_walk_forward,
    stability_summary,
    train_test_validate,
)

SYMBOLS = [
    "600519", "000001", "600036", "601318", "000651",
    "600276", "002415", "600900", "601012", "000333",
]


def build_candidates(close):
    """构建候选组合池。候选池必须事先定义，不能看结果后再临时加。"""
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
    candidates = []
    for factor_name, factor in factor_defs.items():
        for top_n in (3, 5):
            for rebalance_every in (10, 20, 40):
                candidates.append(PortfolioCandidate(
                    name=f"{factor_name}|top{top_n}|rb{rebalance_every}",
                    factor=factor,
                    top_n=top_n,
                    rebalance_every=rebalance_every,
                ))
    return candidates


def _fmt(m: dict) -> str:
    return (
        f"总收益 {m['total_return']:+.2%} | 年化 {m['annualized_return']:+.2%} | "
        f"最大回撤 {m['max_drawdown']:.2%} | 夏普 {m['sharpe']:.2f}"
    )


def main() -> None:
    print(f"[1/5] 构建 {len(SYMBOLS)} 只股票的收盘价面板 ...")
    close = build_close_panel(SYMBOLS, start="20240101", end="20251231")
    print(f"      面板形状: {close.shape[0]} 天 × {close.shape[1]} 只")

    print("[2/5] 预先定义候选池（因子 × topN × 再平衡周期）...")
    candidates = build_candidates(close)
    print(f"      候选数: {len(candidates)}")

    print("[3/5] 单次 train/test 验证（前70%选参，后30%验证）...")
    holdout = train_test_validate(close, candidates, train_ratio=0.7)
    print(f"      train选中: {holdout['best'].name}")
    print(f"      train: {_fmt(holdout['train_metrics'])}")
    print(f"      test : {_fmt(holdout['test_metrics'])}")

    print("[4/5] 滚动 walk-forward：240天训练，60天验证，每60天滚动 ...")
    wf = rolling_walk_forward(
        close,
        candidates,
        train_size=240,
        test_size=60,
        step=60,
    )
    stats = stability_summary(wf["periods"])
    for i, p in enumerate(wf["periods"], start=1):
        print(
            f"      窗口{i}: {p['test_start'].date()}~{p['test_end'].date()} | "
            f"选 {p['candidate'].name:<20} | "
            f"OOS夏普 {p['test_metrics']['sharpe']:.2f} | "
            f"基准夏普 {p['benchmark_metrics']['sharpe']:.2f} | "
            f"{'赢' if p['beat_benchmark'] else '输'}"
        )
    print(
        f"      稳定性: 正夏普窗口 {stats['positive_sharpe_rate']:.0%} | "
        f"跑赢基准窗口 {stats['beat_benchmark_rate']:.0%} | "
        f"OOS夏普中位数 {stats['median_oos_sharpe']:.2f}"
    )

    oos = wf["oos"]
    oos_m = summary(oos["equity"], oos["port_ret"])
    bench_m = summary(oos["benchmark"], oos["benchmark_ret"])
    print("      拼接样本外组合 : " + _fmt(oos_m))
    print("      拼接样本外基准 : " + _fmt(bench_m))

    print("[5/5] 画样本外拼接净值曲线 ...")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(oos.index, oos["equity"], label="walk-forward factor portfolio")
    ax.plot(oos.index, oos["benchmark"], label="equal-weight benchmark", alpha=0.7)
    ax.set_title("out-of-sample walk-forward portfolio")
    ax.set_xlabel("date")
    ax.set_ylabel("net value (start=1)")
    ax.legend()
    fig.tight_layout()
    out_png = RAW_DATA_DIR / "portfolio_walkforward_oos.png"
    fig.savefig(out_png, dpi=120)
    print(f"      图已保存：{out_png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
