"""hk_factor_demo —— 港股因子策略全流程验证（M19）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/hk_factor_demo.py [--fetch]

首次运行需 --fetch 拉取数据（~150只×2接口≈2分钟），之后读缓存（秒级）。

对标美股 Phase 5 的最优方法：
    15因子（3基本面+5量价+5多周期+2变动）
    → 等权合成 → 分层多头L5 + 风险平价对比
    费用默认零（港股费用模型待建）
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

from quant.config import RAW_DATA_DIR, HISTORY_START, HISTORY_END
from quant.data import hk_loader
from quant.data.hk_pool import HK_POOL
from quant.data.panel import build_ohlcv_panels, build_hk_fundamental_panels
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

REBALANCE = 20


def build_15_factors(close, fund, panels):
    """15因子（方向参考美股Phase 3a/3b的IC方向，HK需独立验证）。"""
    amount = panels["amount"]; high, low = panels["high"], panels["low"]
    raw = {}
    raw["quality_roe"] = F.us_quality_roe(fund["roe"])
    raw["quality_gm"] = F.us_quality_roe(fund["gross_margin"])
    raw["growth_rev"] = F.us_growth(fund["rev_yoy"])
    raw["pv_momentum60"] = -F.momentum(close, 60)
    raw["pv_reversal20"] = F.reversal(close, 20)
    raw["pv_lowvol20"] = -F.low_volatility(close, 20)
    raw["pv_amihud"] = -F.amihud_illiquidity(close, amount, 20)
    raw["pv_parkinson"] = -F.parkinson_volatility(high, low, 20)
    raw["pv_reversal5"] = F.reversal(close, 5)
    raw["pv_lowvol5"] = -F.low_volatility(close, 5)
    raw["pv_amihud5"] = -F.amihud_illiquidity(close, amount, 5)
    raw["pv_maslope60"] = -F.ma_slope(close, 60)
    raw["pv_lowvol60"] = -F.low_volatility(close, 60)
    raw["gm_yoy_change"] = fund["gross_margin"] - fund["gross_margin"].shift(252)
    raw["rev_growth_accel"] = fund["rev_yoy"] - fund["rev_yoy"].shift(252)
    return raw


def _stats(bt):
    s = summary(bt["equity"], bt["port_ret"])
    return {"return": s["total_return"], "sharpe": s["sharpe"],
            "dd": s["max_drawdown"], "ann": s["annualized_return"]}


def main():
    do_fetch = "--fetch" in sys.argv
    symbols = HK_POOL
    print(f"HK Pool: {len(symbols)} stocks")

    # ── 数据 ──
    if do_fetch:
        print("Fetching price data...")
        for i, sym in enumerate(symbols):
            try:
                df = hk_loader.fetch_daily(sym, start=HISTORY_START, end=HISTORY_END)
                hk_loader.save_parquet(df, sym)
            except Exception as e:
                print(f"  [{i+1}/{len(symbols)}] {sym} SKIP: {type(e).__name__}")
        print("Fetching fundamental data...")
        from quant.data import hk_fundamental_loader as hfl
        for i, sym in enumerate(symbols):
            try:
                df = hfl.fetch_fundamental(sym)
                hfl.save_parquet(df, sym)
            except Exception:
                pass

    panels = build_ohlcv_panels(symbols, loader=hk_loader)
    close = panels["close"]
    n_stocks = close.notna().any().sum()
    span = f"{close.index.min().date()}~{close.index.max().date()}"
    print(f"Price: {close.shape[0]}d x {n_stocks} stocks | {span}")

    fund = build_hk_fundamental_panels(close.columns.tolist(), align_to=close)
    n_fund = fund.get("roe", pd.DataFrame()).notna().any().sum()
    print(f"Fundamental: {n_fund} stocks")

    # 质量过滤
    close_ok = (close.iloc[-1] >= 1.0) & (panels["volume"].rolling(60).mean().iloc[-1] >= 50000)
    keep = close_ok[close_ok].index.tolist()
    close = close[keep]
    for k in ["close", "amount", "high", "low", "volume"]:
        panels[k] = panels[k][keep]
    for k in panels:
        panels[k] = panels[k].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    print(f"Filtered: {len(keep)} stocks")

    # ── 因子 ──
    facs = build_15_factors(close, fund, panels)
    eq_linear = combine_factors(*facs.values())
    print(f"Factors: {len(facs)}")

    # ── 回测 ──
    print("\nBacktest...")
    bt_ew = long_top_layer(close, eq_linear, rebalance_every=REBALANCE)
    bt_rp = long_top_layer(close, eq_linear, rebalance_every=REBALANCE, weight_mode="risk_parity")

    s_ew = _stats(bt_ew)
    s_rp = _stats(bt_rp)
    s_bench = summary(bt_ew["benchmark"], bt_ew["benchmark_ret"])
    print(f"  EW L5:     {s_ew['return']:>+8.1%}  Sharpe {s_ew['sharpe']:>+6.2f}  DD {s_ew['dd']:>+7.1%}")
    print(f"  RP L5:     {s_rp['return']:>+8.1%}  Sharpe {s_rp['sharpe']:>+6.2f}  DD {s_rp['dd']:>+7.1%}")
    print(f"  Benchmark: {s_bench['total_return']:>+8.1%}  Sharpe {s_bench['sharpe']:>+6.2f}  DD {s_bench['max_drawdown']:>+7.1%}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bt_ew["equity"].index, bt_ew["equity"], label="EW L5", lw=1.8)
    ax.plot(bt_rp["equity"].index, bt_rp["equity"], label="RP L5", lw=1.8)
    ax.plot(bt_ew["benchmark"].index, bt_ew["benchmark"], label="Bench", lw=1.0, ls="--")
    ax.set_title(f"HK Factor Strategy ({n_stocks} stocks, 15 factors)")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend()
    fig.tight_layout()
    png = RAW_DATA_DIR / "hk_factor_equity.png"
    fig.savefig(png, dpi=150)
    print(f"  Chart: {png}")
    print("Done")


if __name__ == "__main__":
    main()
