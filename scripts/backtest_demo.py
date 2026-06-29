"""backtest_demo —— M2 端到端：行情 → 双均线信号 → 回测 → 绩效报告 → 净值曲线图。

运行方式：
    conda activate quant
    python scripts/backtest_demo.py            # 默认用本地已有的 600519
    python scripts/backtest_demo.py 000001     # 指定股票代码（无本地数据会自动拉取）

产出：
    1) 终端打印「策略 vs 买入持有」的四项绩效对比
    2) data/raw/<symbol>_equity.png 净值曲线图（策略 vs 基准）
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
from quant.data.akshare_loader import fetch_daily, save_parquet, load_parquet
from quant.strategy.dual_ma import dual_ma_signal
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import summary


def _load_or_fetch(symbol: str):
    """优先读本地 Parquet；没有就拉取并落地（省去重复联网）。"""
    path = RAW_DATA_DIR / f"{symbol}.parquet"
    if path.exists():
        return load_parquet(symbol)
    df = fetch_daily(symbol, start="20240101", end="20251231")
    save_parquet(df, symbol)
    return df


def _fmt(metrics: dict) -> str:
    return (
        f"总收益 {metrics['total_return']:+.2%} | "
        f"年化 {metrics['annualized_return']:+.2%} | "
        f"最大回撤 {metrics['max_drawdown']:.2%} | "
        f"夏普 {metrics['sharpe']:.2f}"
    )


def main(symbol: str = "600519") -> None:
    print(f"[1/4] 读取/拉取 {symbol} 行情 ...")
    df = _load_or_fetch(symbol)
    print(f"      {len(df)} 个交易日")

    print(f"[2/4] 生成双均线信号(MA5/MA20) ...")
    signal = dual_ma_signal(df, short_window=5, long_window=20)

    print(f"[3/4] 运行回测 ...")
    bt = run_backtest(df, signal)
    strat = summary(bt["equity"], bt["strat_ret"])
    bench = summary(bt["benchmark"], bt["ret"])
    print(f"      策略     : {_fmt(strat)}")
    print(f"      买入持有 : {_fmt(bench)}")

    print(f"[4/4] 画净值曲线 ...")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = bt["date"] if "date" in bt.columns else bt.index
    ax.plot(x, bt["equity"], label="strategy (dual MA)")
    ax.plot(x, bt["benchmark"], label="buy & hold", alpha=0.7)
    ax.set_title(f"{symbol} equity curve")
    ax.set_xlabel("date")
    ax.set_ylabel("net value (start=1)")
    ax.legend()
    fig.tight_layout()
    out_png = RAW_DATA_DIR / f"{symbol}_equity.png"
    fig.savefig(out_png, dpi=120)
    print(f"      图已保存：{out_png}")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "600519")
