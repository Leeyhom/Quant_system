"""sizing_sweep —— 持仓数 × 再平衡频率网格扫描（三市场真实小资金费用口径）。

为什么需要它：
    小资金实盘下固定费/最低费不随本金缩放。持仓越多、每只越小额，固定费占比越高。
    本脚本在【各市场真实小资金费用口径】下，用样本外滚动 walk-forward 扫描
    (持仓数 top_n × 再平衡频率)，让数据告诉我们每市场的最优配置：
      - 美股 1w：一世免佣 + $1/笔平台费（us_cost）。
      - A股 6w：佣金万2.5/最低5元 + 印花税 + 过户费（cn_cost）。
      - 港股 6w：低佣 + 平台费15/笔 + 印花税双边 + 各项杂费（hk_cost）。

口径（遵守 CLAUDE.md）：
    - 防未来函数：fixed_topn_portfolio 用昨日因子选股（i-1）。
    - 结论只看样本外 test 段净夏普，不看样本内最优。
    - 合成因子方向用全样本 IC 符号定向（M19 已验证的稳定先验）——本扫描比较的是
      【持仓数/频率】，对固定合成因子的相对排序稳健；alpha 本身由 M19 滚动验证背书。

运行：
    conda activate quant
    NO_PROXY='*' python scripts/sizing_sweep.py --market US   # 或 CN / HK
"""

from __future__ import annotations

import argparse
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

# 三市场费用回调（各自默认本金：US=1w, CN/HK=6w）
from quant.backtest import us_cost, cn_cost, hk_cost

# 数据/因子复用 quant_engine 现成 loader，避免重写管线
from scripts.quant_engine import QuantMarket

# ─── 扫描网格 ───
TOP_N_GRID = [5, 6, 8, 10, 15]
REBALANCE_GRID = [20, 40, 60]   # 月 / 双月 / 季

# 各市场 walk-forward 窗口（与 quant_engine 默认 train 对齐）+ 费用回调 + 本金
MARKET_CFG = {
    "US": {"train": 480, "test": 120, "step": 60, "cost": us_cost.make_layered_cost_fn,
           "capital": 10_000.0, "ccy": "$"},
    "CN": {"train": 240, "test": 120, "step": 60, "cost": cn_cost.make_layered_cost_fn,
           "capital": 60_000.0, "ccy": "¥"},
    "HK": {"train": 120, "test": 120, "step": 60, "cost": hk_cost.make_layered_cost_fn,
           "capital": 60_000.0, "ccy": "HK$"},
}


def build_composite(market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用 QuantMarket loader 取 (close, factors)，按全样本 IC 符号定向后等权合成。"""
    print(f"[1/3] 构建 {market} 面板 + 因子（loader 复用 quant_engine）...", flush=True)
    loader = {"US": QuantMarket.load_us, "CN": QuantMarket.load_cn,
              "HK": QuantMarket.load_hk}[market]
    close, factors = loader()
    n_stocks = close.notna().any().sum()
    span = f"{close.index.min().date()} ~ {close.index.max().date()}"
    print(f"      行情: {close.shape[0]}天 × {n_stocks}只 | {span} | 因子{len(factors)}个")

    # 全样本 IC 符号定向（M19 稳定先验），等权合成
    fwd = forward_returns(close, horizon=20)
    oriented = []
    for name, fac in factors.items():
        ic = ic_summary(daily_ic(fac, fwd))["mean_ic"]
        oriented.append(fac if ic >= 0 else -fac)
    eq_factor = combine_factors(*oriented)
    return close, eq_factor


def rolling_fixed_topn(close: pd.DataFrame, factor: pd.DataFrame,
                       top_n: int, rebalance: int) -> dict:
    """对固定 top_n 组合做样本外滚动 walk-forward，每窗仅统计 test 段净收益。

    与 composite_demo.rolling_long_top_layer 同口径：train 末日预热建仓、仅统计 test。
    返回跨窗汇总：跑赢率、净夏普中位、累计净收益、年化换手费拖累。
    """
    periods = []
    start, n = 0, len(close)
    while start + TRAIN_SIZE + TEST_SIZE <= n:
        train_end = start + TRAIN_SIZE
        test_end = train_end + TEST_SIZE
        test = close.iloc[train_end - 1:test_end]  # 多带一行预热
        fac = factor.reindex_like(test)
def rolling_fixed_topn(close: pd.DataFrame, factor: pd.DataFrame,
                       top_n: int, rebalance: int, cost_fn,
                       train_size: int, test_size: int, step: int,
                       vol_target: float | None = None,
                       vol_lookback: int = 20, max_leverage: float = 1.0,
                       brake_trigger: float | None = None,
                       brake_exposure: float = 0.3) -> dict:
    """对固定 top_n 组合做样本外滚动 walk-forward，每窗仅统计 test 段净收益。

    与 composite_demo.rolling_long_top_layer 同口径：train 末日预热建仓、仅统计 test。
    vol_target / brake_trigger 非 None 时，对每窗 test 段【逐窗】应用风控缩放（防跨窗口径不一致）。
    返回跨窗汇总：跑赢率、净夏普中位、累计净收益、回撤中位。
    """
    periods = []
    start, n = 0, len(close)
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size
        test = close.iloc[train_end - 1:test_end]  # 多带一行预热
        fac = factor.reindex_like(test)
        bt = fixed_topn_portfolio(
            test, fac, top_n=top_n, rebalance_every=rebalance,
            first_rebalance=True, cost_fn=cost_fn,
        ).iloc[1:].copy()  # 去掉预热行
        if vol_target is not None:
            # 逐窗应用波动率目标（只用本窗 test 段历史，scale.shift(1) 防未来函数）
            bt = vol_targeted(bt, target_vol=vol_target, lookback=vol_lookback,
                              max_leverage=max_leverage)
        if brake_trigger is not None:
            # 逐窗应用移动回撤刹车（peak 不跨窗，scale.shift(1) 防未来函数）
            bt = drawdown_brake(bt, dd_trigger=brake_trigger, reduced_exposure=brake_exposure)
        # 重建净值（去掉预热行后重新累乘）
        bt["equity"] = (1.0 + bt["port_ret"]).cumprod()
        bt["benchmark"] = (1.0 + bt["benchmark_ret"]).cumprod()
        m = summary(bt["equity"], bt["port_ret"])
        bm = summary(bt["benchmark"], bt["benchmark_ret"])

        periods.append({
            "test_start": bt.index[0], "test_end": bt.index[-1],
            "net_sharpe": m["sharpe"], "bench_sharpe": bm["sharpe"],
            "net_ret": m["total_return"], "mdd": m["max_drawdown"],
            "beat": m["sharpe"] > bm["sharpe"],
        })
        start += step

    if not periods:
        return {"n": 0}
    net_sharpes = pd.Series([p["net_sharpe"] for p in periods])
    net_rets = pd.Series([p["net_ret"] for p in periods])
    mdds = pd.Series([p["mdd"] for p in periods])
    return {
        "n": len(periods),
        "beat_rate": float(pd.Series([p["beat"] for p in periods]).mean()),
        "median_net_sharpe": float(net_sharpes.median()),
        "median_bench_sharpe": float(pd.Series([p["bench_sharpe"] for p in periods]).median()),
        "mean_net_ret": float(net_rets.mean()),
        "median_mdd": float(mdds.median()),
    }


def _fee_series(close: pd.DataFrame, factor: pd.DataFrame,
                top_n: int, rebalance: int, cost_fn) -> pd.Series:
    """估算 fixed_topn 在该段的逐日费用拖累（用无费 vs 有费两次回测之差）。"""
    no_fee = fixed_topn_portfolio(close, factor, top_n=top_n,
                                  rebalance_every=rebalance, first_rebalance=True,
                                  cost_fn=None, cost_rate=0.0).iloc[1:]
    with_fee = fixed_topn_portfolio(close, factor, top_n=top_n,
                                    rebalance_every=rebalance, first_rebalance=True,
                                    cost_fn=cost_fn).iloc[1:]
    return (no_fee["port_ret"] - with_fee["port_ret"]).reindex(close.index[1:]).fillna(0.0)


def _lot_infeasible_pct(close: pd.DataFrame, factor: pd.DataFrame,
                        top_n: int, market: str, capital: float) -> float:
    """该 top_n 下，最后一期等权持仓里一手/一股买不起的票占比（整手约束体检）。"""
    scores = factor.iloc[-1].dropna().sort_values(ascending=False)
    sel = scores.head(top_n).index
    w = {s: 1.0 / len(sel) for s in sel}
    px = {s: float(close.iloc[-1].get(s, float("nan"))) for s in sel}
    cfg = market_lot_config(market)
    feas = affordable_lots(w, px, capital, lot_size=cfg["lot_size"],
                           allow_fractional=cfg["allow_fractional"])
    return len(feas["unaffordable"]) / max(1, len(sel))


def main() -> None:
    ap = argparse.ArgumentParser(description="三市场持仓数×再平衡频率扫描（真实小资金费用）")
    ap.add_argument("--market", default="US", choices=["US", "CN", "HK"])
    ap.add_argument("--vol-target", dest="vol_target", type=float, default=None,
                    help="波动率目标年化（如0.15）；默认None=关闭。开启后对每窗逐窗缩放敞口压回撤")
    ap.add_argument("--vol-lookback", dest="vol_lookback", type=int, default=20,
                    help="波动率估算回溯窗口（默认20，适配120日test段）")
    ap.add_argument("--max-leverage", dest="max_leverage", type=float, default=1.0,
                    help="vol-target 缩放上界（默认1.0=只降仓不加杠杆，适合小资金无融资）")
    ap.add_argument("--brake-trigger", dest="brake_trigger", type=float, default=None,
                    help="移动回撤刹车触发阈值（如0.15=回撤超15%降仓）；默认None=关闭")
    ap.add_argument("--brake-exposure", dest="brake_exposure", type=float, default=0.3,
                    help="回撤刹车触发后保留仓位（默认0.3=降到三成仓）")
    args = ap.parse_args()
    cfg = MARKET_CFG[args.market]
    cost_fn = cfg["cost"]()       # 该市场默认本金的费用回调
    capital, ccy = cfg["capital"], cfg["ccy"]
    train, test, step = cfg["train"], cfg["test"], cfg["step"]
    vt, vl, ml = args.vol_target, args.vol_lookback, args.max_leverage
    bt_trig, bt_exp = args.brake_trigger, args.brake_exposure

    print("=" * 80)
    vt_tag = f"，vol-target {vt:.0%}/lev≤{ml:g}" if vt else ""
    bk_tag = f"，回撤刹车 {bt_trig:.0%}→{bt_exp:.0%}仓" if bt_trig else ""
    risk_tag = (vt_tag + bk_tag) or "，无风控"
    print(f"  {args.market} 持仓数 × 再平衡频率扫描"
          f"（真实小资金费用，本金 {ccy}{capital:,.0f}{risk_tag}）")
    print("=" * 80)
    close, eq_factor = build_composite(args.market)
    eq_full = eq_factor.reindex_like(close)

    print(f"\n[2/3] 网格扫描 top_n={TOP_N_GRID} × rebalance={REBALANCE_GRID} "
          f"(TRAIN/TEST/STEP={train}/{test}/{step})...", flush=True)
    rows = []
    for top_n in TOP_N_GRID:
        for reb in REBALANCE_GRID:
            r = rolling_fixed_topn(close, eq_factor, top_n, reb, cost_fn, train, test, step,
                                   vol_target=vt, vol_lookback=vl, max_leverage=ml,
                                   brake_trigger=bt_trig, brake_exposure=bt_exp)
            if r.get("n", 0) == 0:
                continue
            fee = _fee_series(close, eq_full, top_n, reb, cost_fn)
            ann_fee = float((fee.sum() / len(fee)) * 252) if len(fee) else 0.0
            infeasible = _lot_infeasible_pct(close, eq_full, top_n, args.market, capital)
            rows.append({
                "top_n": top_n, "rebalance": reb, "windows": r["n"],
                "beat_rate": r["beat_rate"],
                "net_sharpe": r["median_net_sharpe"],
                "bench_sharpe": r["median_bench_sharpe"],
                "mean_net_ret": r["mean_net_ret"],
                "median_mdd": r["median_mdd"],
                "ann_fee_drag": ann_fee,
                "lot_infeasible": infeasible,
            })
            print(f"      top_n={top_n:>2d} reb={reb:>2d}d | "
                  f"跑赢率 {r['beat_rate']:>5.0%} | 净夏普 {r['median_net_sharpe']:>+5.2f} "
                  f"| 窗均净收益 {r['mean_net_ret']:>+6.1%} | 回撤 {r['median_mdd']:>5.1%} "
                  f"| 费拖累 {ann_fee:>5.2%} | 买不进 {infeasible:>4.0%}", flush=True)

    df = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    out_dir = PROJECT_ROOT / "data" / "raw" / "sizing"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_parts = [args.market.lower()]
    if vt is not None:
        tag_parts.append(f"vt{int(vt * 100)}")
    if bt_trig is not None:
        tag_parts.append(f"brake{int(bt_trig * 100)}")
    out_prefix = "_".join(tag_parts)
    df.to_csv(out_dir / f"{out_prefix}_sizing_sweep.csv", index=False)

    print(f"\n[3/3] 结果排序（按样本外净夏普降序）")
    print("=" * 84)
    print(f"  {'排名':>3s} {'top_n':>6s} {'再平衡':>6s} {'跑赢率':>7s} {'净夏普':>7s} "
          f"{'基准夏普':>8s} {'窗均净收益':>10s} {'回撤':>7s} {'费拖累':>8s} {'买不进':>7s}")
    print(f"  {'-'*80}")
    for i, row in df.iterrows():
        flag = " ★" if i == 0 else ""
        print(f"  {i+1:>3d} {int(row['top_n']):>6d} {int(row['rebalance']):>5d}d "
              f"{row['beat_rate']:>7.0%} {row['net_sharpe']:>+7.2f} "
              f"{row['bench_sharpe']:>+8.2f} {row['mean_net_ret']:>+10.1%} "
              f"{row['median_mdd']:>7.1%} {row['ann_fee_drag']:>8.2%} {row['lot_infeasible']:>7.0%}{flag}")

    best = df.iloc[0]
    bn, br = int(best['top_n']), int(best['rebalance'])
    sizing_summary = {
        "market": args.market,
        "capital": capital,
        "currency": ccy,
        "top_n": bn,
        "rebalance": br,
        "windows": int(best["windows"]),
        "beat_rate": float(best["beat_rate"]),
        "net_sharpe": float(best["net_sharpe"]),
        "bench_sharpe": float(best["bench_sharpe"]),
        "mean_net_ret": float(best["mean_net_ret"]),
        "median_mdd": float(best["median_mdd"]),
        "ann_fee_drag": float(best["ann_fee_drag"]),
        "lot_infeasible": float(best["lot_infeasible"]),
        "vol_target": vt,
        "brake_trigger": bt_trig,
    }
    print("\n" + "=" * 84)
    print(f"  最优配置：持有 {bn} 只，每 {br} 日再平衡")
    print(f"    样本外净夏普 {best['net_sharpe']:+.2f}（基准 {best['bench_sharpe']:+.2f}），"
          f"跑赢率 {best['beat_rate']:.0%}，回撤 {best['median_mdd']:.1%}，"
          f"费拖累 {best['ann_fee_drag']:.2%}，买不进 {best['lot_infeasible']:.0%}")

    # ─── 最优配置上：风控开/关对比（未开任何风控时补跑 vol-target 与 brake 两个版本） ───
    if vt is None and bt_trig is None:
        on_vt = rolling_fixed_topn(close, eq_factor, bn, br, cost_fn, train, test, step,
                                   vol_target=0.20, vol_lookback=vl, max_leverage=1.0)
        on_bk = rolling_fixed_topn(close, eq_factor, bn, br, cost_fn, train, test, step,
                                   brake_trigger=0.15, brake_exposure=0.3)
        sizing_summary["risk_compare"] = {
            "none": {
                "net_sharpe": float(best["net_sharpe"]),
                "median_mdd": float(best["median_mdd"]),
                "mean_net_ret": float(best["mean_net_ret"]),
            },
            "vol_target_20": {
                "net_sharpe": float(on_vt["median_net_sharpe"]),
                "median_mdd": float(on_vt["median_mdd"]),
                "mean_net_ret": float(on_vt["mean_net_ret"]),
            },
            "drawdown_brake_15": {
                "net_sharpe": float(on_bk["median_net_sharpe"]),
                "median_mdd": float(on_bk["median_mdd"]),
                "mean_net_ret": float(on_bk["mean_net_ret"]),
            },
        }
        print(f"\n  ▸ 风控开/关对比（最优 {bn}只/{br}日）:")
        print(f"      无风控        : 净夏普 {best['net_sharpe']:+.2f} | 回撤 {best['median_mdd']:>5.1%} "
              f"| 窗均净收益 {best['mean_net_ret']:>+6.1%}")
        print(f"      vol-target20% : 净夏普 {on_vt['median_net_sharpe']:+.2f} | 回撤 {on_vt['median_mdd']:>5.1%} "
              f"| 窗均净收益 {on_vt['mean_net_ret']:>+6.1%}  (Δ夏普{on_vt['median_net_sharpe']-best['net_sharpe']:+.2f}/回撤{on_vt['median_mdd']-best['median_mdd']:+.1%})")
        print(f"      回撤刹车15%   : 净夏普 {on_bk['median_net_sharpe']:+.2f} | 回撤 {on_bk['median_mdd']:>5.1%} "
              f"| 窗均净收益 {on_bk['mean_net_ret']:>+6.1%}  (Δ夏普{on_bk['median_net_sharpe']-best['net_sharpe']:+.2f}/回撤{on_bk['median_mdd']-best['median_mdd']:+.1%})")
        print(f"      （回撤为单窗120日口径，偏小；全样本大回撤对比见 three_market_compare.py）")

    print(f"\n  → 跑实盘：python scripts/quant_engine.py --market {args.market} --live "
          f"--top-n {bn} --capital {int(capital)} --brake-trigger 0.15")
    pd.Series(sizing_summary).to_json(
        out_dir / f"{out_prefix}_sizing_summary.json",
        force_ascii=False,
        indent=2,
    )
    print(f"  数据已保存：{out_dir / f'{out_prefix}_sizing_sweep.csv'}")
    print(f"  摘要已保存：{out_dir / f'{out_prefix}_sizing_summary.json'}")
    print("=" * 84)


if __name__ == "__main__":
    main()
