"""hk_strategy_v2 —— 港股策略深度调整（M19 v2）。

v1失败原因：池太小(136只)/因子方向搬美股/基本面因子噪音。
v2策略：
    ① 扩池到900只→取前200只高成交额（流动性最好）
    ② 纯量价因子（港股基本面因子IC诊断全是噪音）
    ③ 独立IC方向筛选，HK-specific
    ④ 尝试5/20/60日多周期
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
from quant.data import hk_loader
from quant.data.panel import build_ohlcv_panels
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

HORIZON = 20
REBALANCE = 20


def main():
    # 读取过滤后的HK全量ticker，取前200只
    with open(PROJECT_ROOT / "data" / "raw" / "hk_all_tickers.txt") as f:
        tickers = [l.strip() for l in f if l.strip()]
    tickers = tickers[:200]
    print(f"HK v2 pool: {len(tickers)} stocks (top 200 by liquidity)")

    # 行情面板（200只需要几分钟）
    panels = build_ohlcv_panels(tickers, loader=hk_loader)
    close = panels["close"]
    # 质量过滤
    ok = (close.iloc[-1] >= 1.0) & (panels["volume"].rolling(60).mean().iloc[-1] >= 50000)
    keep = ok[ok].index.tolist()
    close = close[keep]
    for k in panels: panels[k] = panels[k][keep].replace([np.inf,-np.inf],np.nan).ffill().fillna(0.0)
    print(f"Filtered: {len(keep)} stocks")

    fwd = forward_returns(close, horizon=HORIZON)
    amount = panels["amount"]; high, low = panels["high"], panels["low"]

    # 纯量价因子（不依赖基本面）
    raw = {}
    for w in [5, 20, 60]:
        raw[f"reversal{w}"] = F.reversal(close, w)
        raw[f"lowvol{w}"] = F.low_volatility(close, w)
        raw[f"amihud{w}"] = F.amihud_illiquidity(close, amount, w)
        raw[f"momentum{w}"] = F.momentum(close, w)
    raw["parkinson20"] = F.parkinson_volatility(high, low, 20)
    raw["maslope60"] = F.ma_slope(close, 60)

    # IC方向独立筛选+定向
    print("\nHK Independent IC:")
    oriented = {}
    for name, fac in raw.items():
        s = ic_summary(daily_ic(fac, fwd))
        if abs(s["t_stat"]) < 2.0:
            print(f"  {name:18s} IC{s['mean_ic']:+.4f} t{s['t_stat']:+.2f} SKIP")
            continue
        oriented[name] = -fac if s["mean_ic"] < 0 else fac
        d = "FLIP" if s["mean_ic"] < 0 else "OK"
        print(f"  {name:18s} IC{s['mean_ic']:+.4f} t{s['t_stat']:+.2f} {d}")

    if not oriented:
        print("\nNo stable factors found in HK. Market resists factor approach.")
        return

    print(f"\nSelected: {len(oriented)} factors")
    eq = combine_factors(*oriented.values())

    bt_ew = long_top_layer(close, eq, rebalance_every=REBALANCE)
    bt_rp = long_top_layer(close, eq, rebalance_every=REBALANCE, weight_mode="risk_parity")
    s_ew = summary(bt_ew["equity"], bt_ew["port_ret"])
    s_rp = summary(bt_rp["equity"], bt_rp["port_ret"])
    s_b = summary(bt_ew["benchmark"], bt_ew["benchmark_ret"])
    print(f"\n  EW L5:  {s_ew['total_return']:>+8.1%}  Sharpe {s_ew['sharpe']:>+6.2f}  DD {s_ew['max_drawdown']:>+7.1%}")
    print(f"  RP L5:  {s_rp['total_return']:>+8.1%}  Sharpe {s_rp['sharpe']:>+6.2f}  DD {s_rp['max_drawdown']:>+7.1%}")
    print(f"  Bench:  {s_b['total_return']:>+8.1%}  Sharpe {s_b['sharpe']:>+6.2f}  DD {s_b['max_drawdown']:>+7.1%}")

    excess = s_rp['total_return'] - s_b['total_return']
    print(f"\n  >>> Excess: {excess:+.1%}  {'✅ POSITIVE' if excess > 0 else '❌ NEGATIVE'}")
    if excess > 0:
        print("  HK strategy PASSES!")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bt_ew["equity"].index, bt_ew["equity"], label="EW L5")
    ax.plot(bt_rp["equity"].index, bt_rp["equity"], label="RP L5", color="darkorange")
    ax.plot(bt_ew["benchmark"].index, bt_ew["benchmark"], label="Bench", ls="--", color="gray")
    ax.set_title(f"HK Strategy V2 ({len(oriented)} PV factors, 200 stocks)")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend()
    fig.tight_layout()
    png = RAW_DATA_DIR / "hk_strategy_v2.png"
    fig.savefig(png, dpi=150)
    print(f"  Chart: {png}")


if __name__ == "__main__":
    main()
