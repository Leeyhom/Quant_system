"""compare_demo —— 三策略横向对比：双均线 vs 均值回归 vs 买入持有。

运行方式：
    conda activate quant
    python scripts/compare_demo.py            # 默认 600519
    python scripts/compare_demo.py 000001

产出：
    1) 终端打印三者的四项绩效对比
    2) data/raw/<symbol>_compare.png 三条净值曲线叠加图
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
from quant.data.akshare_loader import fetch_daily, save_parquet, load_parquet
from quant.strategy.dual_ma import dual_ma_signal
from quant.strategy.mean_reversion import mean_reversion_signal
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import summary


def _load_or_fetch(symbol: str):
    path = RAW_DATA_DIR / f"{symbol}.parquet"
    if path.exists():
        return load_parquet(symbol)
    df = fetch_daily(symbol, start="20240101", end="20251231")
    save_parquet(df, symbol)
    return df


def _fmt(name: str, m: dict) -> str:
    return (
        f"{name:<10} 总收益 {m['total_return']:+.2%} | 年化 {m['annualized_return']:+.2%} | "
        f"最大回撤 {m['max_drawdown']:.2%} | 夏普 {m['sharpe']:.2f}"
    )


def main(symbol: str = "600519") -> None:
    print(f"[1/3] 读取/拉取 {symbol} 行情 ...")
    df = _load_or_fetch(symbol)
    print(f"      {len(df)} 个交易日")

    print(f"[2/3] 跑三种策略 ...")
    # 双均线
    bt_ma = run_backtest(df, dual_ma_signal(df, 5, 20))
    # 均值回归
    bt_mr = run_backtest(df, mean_reversion_signal(df, window=20, entry_z=1.0, exit_z=0.0))
    # 买入持有（用全 1 信号取 benchmark 列）
    bt_bh = run_backtest(df, pd.Series(1, index=df.index))

    m_ma = summary(bt_ma["equity"], bt_ma["strat_ret"])
    m_mr = summary(bt_mr["equity"], bt_mr["strat_ret"])
    m_bh = summary(bt_bh["benchmark"], bt_bh["ret"])
    print("      " + _fmt("双均线", m_ma))
    print("      " + _fmt("均值回归", m_mr))
    print("      " + _fmt("买入持有", m_bh))

    print(f"[3/3] 画净值对比曲线 ...")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = bt_ma["date"] if "date" in bt_ma.columns else bt_ma.index
    ax.plot(x, bt_ma["equity"], label="dual MA (trend)")
    ax.plot(x, bt_mr["equity"], label="mean reversion")
    ax.plot(x, bt_bh["benchmark"], label="buy & hold", alpha=0.7)
    ax.set_title(f"{symbol} strategy comparison")
    ax.set_xlabel("date")
    ax.set_ylabel("net value (start=1)")
    ax.legend()
    fig.tight_layout()
    out_png = RAW_DATA_DIR / f"{symbol}_compare.png"
    fig.savefig(out_png, dpi=120)
    print(f"      图已保存：{out_png}")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "600519")
