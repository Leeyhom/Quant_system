"""three_market_compare —— 三市场真实小资金口径对比报告。

回答用户三个问题：
  1. 美股 1w / A股 6w / 港股 6w 下，回测收益到底是多少；
  2. 相比"之前"（旧比例成本 0.1%换手）有什么变化；
  3. 哪些市场的配置真能跑实盘（费用 + 整手约束体检）。

做法：每市场用 sizing_sweep 选出的最优 (top_n, rebalance)，
跑两套口径的全样本 fixed_topn_portfolio：
  - 旧口径：比例成本 cost_rate=0.001（复现"之前"的数字）；
  - 新口径：各市场真实小资金费用回调（us_cost/cn_cost/hk_cost）。
并对最后一期持仓做整手/碎股可行性体检。

运行：
    NO_PROXY='*' python scripts/three_market_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from quant.factor.factors import combine_factors
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import fixed_topn_portfolio, vol_targeted, drawdown_brake
from quant.backtest.metrics import summary
from quant.backtest.lot_sizing import market_lot_config, affordable_lots
from quant.backtest import us_cost, cn_cost, hk_cost
from scripts.quant_engine import QuantMarket

# 各市场 sizing_sweep 选出的最优配置（已跑扫描填入）。
# HK 取稳健解 6只/60日（净夏普+0.63, 费拖累2.38%, 0%买不进），
# 而非原始最高 8只/20日（+0.67 但费拖累6.89%、12%买不进——20日再平衡费用脆弱）。
# HK 本金提到 9w（HK$90k）：6只→每只预算 HK$1.5w，覆盖 HK$150 以下股一手，消除整手约束。
# vol-target 目标按市场调优（全样本目标波动扫描）。
# 移动回撤刹车（drawdown_brake）：实测美股上夏普↑且回撤↓，比 vol-target 更优；按市场设触发阈值。
VOL_TARGET = {"US": 0.20, "CN": 0.25, "HK": 0.30}
BRAKE = {"US": (0.15, 0.3), "CN": (0.15, 0.3), "HK": (0.15, 0.3)}  # (触发回撤, 触发后仓位)
OPTIMAL = {
    "US": {"top_n": 8, "rebalance": 60, "capital": 10_000.0, "ccy": "$",
           "cost": us_cost.make_layered_cost_fn},
    "CN": {"top_n": 6, "rebalance": 60, "capital": 60_000.0, "ccy": "¥",
           "cost": cn_cost.make_layered_cost_fn},
    "HK": {"top_n": 6, "rebalance": 60, "capital": 90_000.0, "ccy": "HK$",
           "cost": hk_cost.make_layered_cost_fn},
}
LOADERS = {"US": QuantMarket.load_us, "CN": QuantMarket.load_cn, "HK": QuantMarket.load_hk}


def build_composite(market: str):
    """取 (close, 等权合成因子)，全样本 IC 符号定向后等权。"""
    close, factors = LOADERS[market]()
    fwd = forward_returns(close, horizon=20)
    oriented = []
    for name, fac in factors.items():
        ic = ic_summary(daily_ic(fac, fwd))["mean_ic"]
        oriented.append(fac if ic >= 0 else -fac)
    return close, combine_factors(*oriented)


def run_one(close, factor, top_n, rebalance, cost_fn, cost_rate, vol_target=None, brake=None):
    """跑一套口径的全样本 fixed_topn，返回绩效 dict。
    vol_target 非None叠加波动率目标；brake=(trigger,exposure) 非None叠加移动回撤刹车。"""
    bt = fixed_topn_portfolio(close, factor, top_n=top_n, rebalance_every=rebalance,
                              cost_fn=cost_fn, cost_rate=cost_rate)
    if vol_target is not None:
        # 全样本够长，lookback=60；只降仓不加杠杆（max_leverage=1.0）
        bt = vol_targeted(bt, target_vol=vol_target, lookback=60, max_leverage=1.0)
        bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
    if brake is not None:
        bt = drawdown_brake(bt, dd_trigger=brake[0], reduced_exposure=brake[1])
        bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
    m = summary(bt["equity"], bt["port_ret"])
    bm = summary(bt["benchmark"], bt["benchmark_ret"])
    return {
        "total": m["total_return"], "ann": m["annualized_return"],
        "sharpe": m["sharpe"], "mdd": m["max_drawdown"],
        "bench_total": bm["total_return"], "bench_sharpe": bm["sharpe"],
    }


def lot_check(close, factor, top_n, market, capital):
    scores = factor.iloc[-1].dropna().sort_values(ascending=False)
    sel = scores.head(top_n).index
    w = {s: 1.0 / len(sel) for s in sel}
    px = {s: float(close.iloc[-1].get(s, float("nan"))) for s in sel}
    cfg = market_lot_config(market)
    feas = affordable_lots(w, px, capital, lot_size=cfg["lot_size"],
                           allow_fractional=cfg["allow_fractional"])
    return len(feas["unaffordable"]), len(sel)


def main():
    print("=" * 96)
    print("  三市场真实小资金口径对比（旧比例成本 vs 新真实费用 vs 新真实费用+波动率目标）")
    print("=" * 96)

    results = {}
    for mkt in ["US", "CN", "HK"]:
        cfg = OPTIMAL[mkt]
        print(f"\n[{mkt}] 加载数据 + 因子...", flush=True)
        close, factor = build_composite(mkt)
        factor = factor.reindex_like(close)
        span = f"{close.index.min().date()}~{close.index.max().date()}"
        n_stocks = int(close.notna().any().sum())

        old = run_one(close, factor, cfg["top_n"], cfg["rebalance"],
                      cost_fn=None, cost_rate=0.001)               # 旧比例成本
        new = run_one(close, factor, cfg["top_n"], cfg["rebalance"],
                      cost_fn=cfg["cost"](), cost_rate=0.001)       # 新真实费用
        vt = run_one(close, factor, cfg["top_n"], cfg["rebalance"],
                     cost_fn=cfg["cost"](), cost_rate=0.001,
                     vol_target=VOL_TARGET[mkt])                   # 新真实费用 + vol-target(按市场调优)
        n_bad, n_sel = lot_check(close, factor, cfg["top_n"], mkt, cfg["capital"])

        results[mkt] = {"cfg": cfg, "span": span, "n_stocks": n_stocks,
                        "old": old, "new": new, "vt": vt, "n_bad": n_bad, "n_sel": n_sel}
        print(f"      {mkt} {span} | {n_stocks}只 | 配置 {cfg['top_n']}只/{cfg['rebalance']}日")

    # ─── 对比表 ───
    print("\n" + "=" * 96)
    print("  全样本绩效对比（各市场最优配置；vol-target 按市场调优 US20%/CN25%/HK30%、只降仓不加杠杆）")
    print("=" * 96)
    hdr = (f"  {'市场':5s} {'本金':>11s} {'配置':>10s} "
           f"{'口径':>14s} {'累计收益':>10s} {'年化':>8s} {'夏普':>7s} {'回撤':>8s}")
    print(hdr)
    print("  " + "-" * 92)
    for mkt in ["US", "CN", "HK"]:
        r = results[mkt]; cfg = r["cfg"]
        cap = f"{cfg['ccy']}{cfg['capital']:,.0f}"
        conf = f"{cfg['top_n']}只/{cfg['rebalance']}d"
        for tag, key in [("旧比例成本", "old"), ("新真实费用", "new"), ("新+vol目标", "vt")]:
            d = r[key]
            print(f"  {mkt:5s} {cap:>11s} {conf:>10s} {tag:>14s} "
                  f"{d['total']:>+9.1%} {d['ann']:>+7.1%} {d['sharpe']:>+7.2f} {d['mdd']:>7.1%}")
        print("  " + "-" * 92)

    # ─── 变化量 + 实盘结论 ───
    print("\n" + "=" * 96)
    print("  优化影响（费用口径 + 波动率目标）+ 实盘可行性结论")
    print("=" * 96)
    for mkt in ["US", "CN", "HK"]:
        r = results[mkt]; cfg = r["cfg"]
        new, vt = r["new"], r["vt"]
        d_mdd = vt["mdd"] - new["mdd"]          # 负=回撤改善
        d_shp = vt["sharpe"] - new["sharpe"]
        d_ret = vt["total"] - new["total"]
        beat = vt["sharpe"] > vt["bench_sharpe"]
        bad_pct = r["n_bad"] / max(1, r["n_sel"])
        print(f"\n  【{mkt}】最优 {cfg['top_n']}只/{cfg['rebalance']}日, 本金 {cfg['ccy']}{cfg['capital']:,.0f}, vol目标{VOL_TARGET[mkt]:.0%}")
        print(f"    vol-target 效果: 回撤 {new['mdd']:.1%} → {vt['mdd']:.1%} ({d_mdd:+.1%}，负=改善) | "
              f"夏普 {new['sharpe']:+.2f} → {vt['sharpe']:+.2f} ({d_shp:+.2f}) | "
              f"收益 {new['total']:+.0%} → {vt['total']:+.0%} ({d_ret:+.0%})")
        print(f"    最终(新+vol) vs 基准夏普 {vt['bench_sharpe']:+.2f}: "
              f"{'✅跑赢' if beat else '❌未跑赢'}")
        print(f"    整手可行性: {r['n_bad']}/{r['n_sel']} 只买不进 ({bad_pct:.0%})"
              + ("  ⚠️需提高本金或剔除高价股" if bad_pct > 0 else "  ✅全部可买"))
    print("\n" + "=" * 96)
    print("  注：旧口径=比例成本0.1%换手（忽略最低佣金/平台费/整手）；新口径=各市场真实小资金费用；")
    print("      vol-target=波动率目标(US20%/CN25%/HK30%)、只降仓不加杠杆压回撤。结论以样本外滚动为准（见 sizing_sweep）。")
    print("=" * 96)


if __name__ == "__main__":
    main()
