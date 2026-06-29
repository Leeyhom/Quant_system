"""param_scan_demo —— M3 起步：双均线参数网格扫描 + 夏普热力图。

运行方式：
    conda activate quant
    python scripts/param_scan_demo.py            # 默认 600519
    python scripts/param_scan_demo.py 000001

产出：
    1) 终端打印：表现最好的几组参数 + 买入持有基准对照
    2) data/raw/<symbol>_sharpe_heatmap.png 夏普热力图（看平原 vs 尖峰）
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
from quant.backtest.param_scan import scan, pivot
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import summary

# 待扫描的参数网格
SHORT_WINDOWS = [3, 5, 8, 10, 15, 20]
LONG_WINDOWS = [20, 30, 40, 50, 60, 90, 120]


def _load_or_fetch(symbol: str):
    path = RAW_DATA_DIR / f"{symbol}.parquet"
    if path.exists():
        return load_parquet(symbol)
    df = fetch_daily(symbol, start="20240101", end="20251231")
    save_parquet(df, symbol)
    return df


def main(symbol: str = "600519") -> None:
    print(f"[1/4] 读取/拉取 {symbol} 行情 ...")
    df = _load_or_fetch(symbol)
    print(f"      {len(df)} 个交易日")

    print(f"[2/4] 扫描 {len(SHORT_WINDOWS)}×{len(LONG_WINDOWS)} 参数网格 ...")
    result = scan(df, SHORT_WINDOWS, LONG_WINDOWS)

    # 基准：买入持有。run_backtest 的 benchmark 列与信号无关，传全 1 信号即可。
    full_position = pd.Series(1, index=df.index)
    bench_bt = run_backtest(df, full_position)
    bench = summary(bench_bt["benchmark"], bench_bt["ret"])

    print(f"[3/4] 表现最好的 5 组参数（按夏普）：")
    top = result.sort_values("sharpe", ascending=False).head(5)
    for _, r in top.iterrows():
        print(
            f"      MA{int(r['short']):>2}/{int(r['long']):>3} -> "
            f"总收益 {r['total_return']:+.2%} | 夏普 {r['sharpe']:.2f} | 回撤 {r['max_drawdown']:.2%}"
        )
    print(
        f"      [基准] 买入持有 -> 总收益 {bench['total_return']:+.2%} | "
        f"夏普 {bench['sharpe']:.2f} | 回撤 {bench['max_drawdown']:.2%}"
    )
    n_beat = (result["sharpe"] > bench["sharpe"]).sum()
    print(f"      {n_beat}/{len(result)} 组参数的夏普跑赢了买入持有")

    print(f"[4/4] 画夏普热力图 ...")
    mat = pivot(result, metric="sharpe")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("long window")
    ax.set_ylabel("short window")
    ax.set_title(f"{symbol} dual-MA Sharpe heatmap")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    out_png = RAW_DATA_DIR / f"{symbol}_sharpe_heatmap.png"
    fig.savefig(out_png, dpi=120)
    print(f"      图已保存：{out_png}")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "600519")
