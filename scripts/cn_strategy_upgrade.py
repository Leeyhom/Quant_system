"""cn_strategy_upgrade —— A股策略升级：美股全套方法论迁移验证（M19）。

运行方式:
    conda activate quant
    NO_PROXY='*' python scripts/cn_strategy_upgrade.py

对标美股 Phase 5 方法：
    ① 15因子（基本面5+量价8+变动2，A股有日频估值所以更多基本因子可用）
    ② IC方向独立筛选（不搬美股方向）
    ③ 行业+市值双中性化（A股独有优势！行业映射已有+日频市值可得）
    ④ 分层多头L5 + 风险平价对比
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
from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.industry import industry_series
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize as neut_fn
from quant.factor.composite import factor_correlation
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

HORIZON = 20
REBALANCE = 20
N_LAYERS = 5


def build_cn_factors(close, value_panels, panels):
    """A股15+因子：基本面(日频估值→更多因子) + 量价 + 变动。"""
    amount = panels["amount"]; high, low = panels["high"], panels["low"]
    raw = {}
    # 基本面6个（A股日频估值独有优势）
    raw["earnings_yield"] = F.earnings_yield(value_panels["pe_ttm"])
    raw["book_to_price"] = F.book_to_price(value_panels["pb"])
    raw["sales_yield"] = F.sales_yield(value_panels["ps"])
    raw["cashflow_yield"] = F.cashflow_yield(value_panels["pcf"])
    raw["quality_roe"] = F.quality_roe(value_panels["pe_ttm"], value_panels["pb"])
    raw["growth_peg"] = F.growth_peg(value_panels["peg"])
    # 小市值
    raw["small_size"] = F.small_size(value_panels["total_mv"])
    # 量价8个（多周期）
    raw["pv_momentum60"] = F.momentum(close, 60)
    raw["pv_reversal20"] = F.reversal(close, 20)
    raw["pv_lowvol20"] = F.low_volatility(close, 20)
    raw["pv_amihud"] = F.amihud_illiquidity(close, amount, 20)
    raw["pv_parkinson"] = F.parkinson_volatility(high, low, 20)
    raw["pv_reversal5"] = F.reversal(close, 5)
    raw["pv_maslope60"] = F.ma_slope(close, 60)
    raw["pv_lowvol60"] = F.low_volatility(close, 60)
    return raw


def _stats(bt):
    s = summary(bt["equity"], bt["port_ret"])
    return {"return": s["total_return"], "sharpe": s["sharpe"], "dd": s["max_drawdown"]}


def main():
    do_ic_screen = "--ic-screen" in sys.argv
    symbols = DEFAULT_POOL
    print(f"A-share pool: {len(symbols)} stocks")

    # 面板
    panels = build_ohlcv_panels(symbols)
    close = panels["close"]
    value = build_value_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    print(f"Price: {close.shape[0]}d x {close.shape[1]} | Value: {len(value)} fields")

    fwd = forward_returns(close, horizon=HORIZON)
    facs_raw = build_cn_factors(close, value, panels)

    # IC诊断+方向筛选
    print("\nA-share Factor IC:")
    facs_oriented = {}
    for name, fac in facs_raw.items():
        s = ic_summary(daily_ic(fac, fwd))
        if do_ic_screen:
            if abs(s["t_stat"]) < 1.5:
                print(f"  {name:20s} IC{s['mean_ic']:+.4f} t{s['t_stat']:+.2f} → SKIP (noise)")
                continue
        facs_oriented[name] = -fac if s["mean_ic"] < 0 else fac
        direction = "FLIP" if s["mean_ic"] < 0 else "OK"
        print(f"  {name:20s} IC{s['mean_ic']:+.4f} t{s['t_stat']:+.2f} → {direction}")

    # 双中性化
    facs_neut = {n: neut_fn(f, industry=ind, log_mv=log_mv, mode="full")
                 for n, f in facs_oriented.items()}
    print(f"\nFactors: {len(facs_neut)} selected, dual-neutralized")

    eq = combine_factors(*facs_neut.values())

    # 回测
    bt_ew = long_top_layer(close, eq, rebalance_every=REBALANCE)
    bt_rp = long_top_layer(close, eq, rebalance_every=REBALANCE, weight_mode="risk_parity")
    s_ew = _stats(bt_ew); s_rp = _stats(bt_rp)
    s_b = summary(bt_ew["benchmark"], bt_ew["benchmark_ret"])
    print(f"\n  EW L5:  {s_ew['return']:>+8.1%}  Sharpe {s_ew['sharpe']:>+6.2f}  DD {s_ew['dd']:>+7.1%}")
    print(f"  RP L5:  {s_rp['return']:>+8.1%}  Sharpe {s_rp['sharpe']:>+6.2f}  DD {s_rp['dd']:>+7.1%}")
    print(f"  Bench:  {s_b['total_return']:>+8.1%}  Sharpe {s_b['sharpe']:>+6.2f}  DD {s_b['max_drawdown']:>+7.1%}")

    # 图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bt_ew["equity"].index, bt_ew["equity"], label="EW L5", lw=1.8)
    ax.plot(bt_rp["equity"].index, bt_rp["equity"], label="RP L5", lw=1.8, color="darkorange")
    ax.plot(bt_ew["benchmark"].index, bt_ew["benchmark"], label="Bench", lw=1.0, ls="--", color="gray")
    ax.set_title(f"A-Share Factor Upgrade ({len(facs_neut)} factors, dual-neutralized)")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend()
    fig.tight_layout()
    png = RAW_DATA_DIR / "cn_strategy_upgrade.png"
    fig.savefig(png, dpi=150)
    print(f"  Chart: {png}")
    print("Done")


if __name__ == "__main__":
    main()
