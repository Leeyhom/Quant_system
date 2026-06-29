"""wf_runner —— 无泄漏 Walk-Forward Runner（审计修正版）

核心改进：彻底消除全样本因子方向/筛选泄漏
1. 每个 walk-forward 窗口内，只用 train 段 IC 定因子方向
2. 每个 walk-forward 窗口内，只用 train 段 IC 做因子筛选
3. test 段完全盲测，不使用任何未来信息
4. train 段尾部裁掉 horizon 天，避免标签重叠

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/wf_runner.py --market CN --ic-screen
    NO_PROXY='*' python scripts/wf_runner.py --market US --train 480 --test 120 --step 120
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from quant.backtest.ic_analysis import daily_ic, ic_summary
from quant.backtest.metrics import summary
from quant.backtest.layered import long_top_layer
from quant.factor.neutralize import neutralize


@dataclass
class WFConfig:
    train_days: int = 480
    test_days: int = 120
    step_days: int = 120
    horizon: int = 20
    rebalance: int = 20
    ic_screen_threshold: float = 1.5
    weight_mode: str = "equal"
    n_layers: int = 5


def build_factors_cn(close, value_panels, panels):
    """A股因子库"""
    from quant.factor import factors as F
    raw = {}
    raw["earnings_yield"] = F.earnings_yield(value_panels["pe_ttm"])
    raw["book_to_price"] = F.book_to_price(value_panels["pb"])
    raw["sales_yield"] = F.sales_yield(value_panels["ps"])
    raw["cashflow_yield"] = F.cashflow_yield(value_panels["pcf"])
    raw["quality_roe"] = F.quality_roe(value_panels["pe_ttm"], value_panels["pb"])
    raw["growth_peg"] = F.growth_peg(value_panels["peg"])
    raw["small_size"] = F.small_size(value_panels["total_mv"])
    raw["pv_momentum60"] = F.momentum(close, 60)
    raw["pv_reversal20"] = F.reversal(close, 20)
    raw["pv_lowvol20"] = F.low_volatility(close, 20)
    raw["pv_amihud"] = F.amihud_illiquidity(close, panels["amount"], 20)
    raw["pv_parkinson"] = F.parkinson_volatility(panels["high"], panels["low"], 20)
    raw["pv_reversal5"] = F.reversal(close, 5)
    raw["pv_maslope60"] = F.ma_slope(close, 60)
    raw["pv_lowvol60"] = F.low_volatility(close, 60)
    return raw


def build_factors_us(close, fund, panels):
    """美股因子库"""
    from quant.factor import factors as F
    raw = {}
    if fund.get("roe") is not None:
        raw["quality_roe"] = F.us_quality_roe(fund["roe"])
    if fund.get("gross_margin") is not None:
        raw["quality_gm"] = F.us_quality_roe(fund["gross_margin"])
    if fund.get("rev_yoy") is not None:
        raw["growth_rev"] = F.us_growth(fund["rev_yoy"])
    raw["pv_momentum60"] = F.momentum(close, 60)
    raw["pv_reversal20"] = F.reversal(close, 20)
    raw["pv_lowvol20"] = F.low_volatility(close, 20)
    raw["pv_amihud"] = F.amihud_illiquidity(close, panels["amount"], 20)
    raw["pv_parkinson"] = F.parkinson_volatility(panels["high"], panels["low"], 20)
    raw["pv_reversal5"] = F.reversal(close, 5)
    raw["pv_lowvol5"] = F.low_volatility(close, 5)
    raw["pv_amihud5"] = F.amihud_illiquidity(close, panels["amount"], 5)
    raw["pv_maslope60"] = F.ma_slope(close, 60)
    raw["pv_lowvol60"] = F.low_volatility(close, 60)
    return raw


def get_factor_ic_stats(
    raw_factors: Dict[str, pd.DataFrame],
    fwd_ret_train: pd.DataFrame,
) -> Dict[str, Dict]:
    """只用 train 段 IC 计算每个因子的统计量（不做定向，避免修改输入因子的 index 范围）"""
    stats = {}
    for name, fac in raw_factors.items():
        fac_train = fac.reindex(index=fwd_ret_train.index)
        ic = daily_ic(fac_train, fwd_ret_train, min_count=5)
        s = ic_summary(ic)
        mean_ic = float(s["mean_ic"]) if pd.notna(s["mean_ic"]) else 0.0
        t_stat = float(s["t_stat"]) if pd.notna(s["t_stat"]) else 0.0
        stats[name] = {"mean_ic": mean_ic, "t_stat": t_stat}
    return stats


def combine_factors_equal_weight(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """等权合成：每个因子先横截面 rank，再平均"""
    from quant.factor.factors import combine_factors
    return combine_factors(*factors.values())


def walk_forward_backtest(
    close_full: pd.DataFrame,
    raw_factors_full: Dict[str, pd.DataFrame],
    neutralize_fn: Callable | None = None,
    cfg: WFConfig | None = None,
) -> Dict:
    """核心：无泄漏 Walk-Forward 回测主函数"""
    cfg = cfg or WFConfig()
    dates = close_full.index.sort_values()
    n_dates = len(dates)

    all_test_rets = []
    all_test_bench = []
    window_info = []

    start = 0
    window_idx = 0
    while start + cfg.train_days + cfg.test_days <= n_dates:
        train_start = dates[start]
        train_end = dates[start + cfg.train_days - 1]
        test_start = dates[start + cfg.train_days]
        test_end = dates[min(start + cfg.train_days + cfg.test_days - 1, n_dates - 1)]

        fwd_ret_full = close_full.shift(-cfg.horizon) / close_full - 1.0
        fwd_train = fwd_ret_full.loc[train_start:train_end].copy()
        if len(fwd_train) > cfg.horizon:
            fwd_train = fwd_train.iloc[:-cfg.horizon]

        # 先用 train 段 IC 定向，然后把方向应用到全期因子
        # 关键：orient_and_select_factors 需要 train 段 IC 来定方向，但定向后需要用在全期
        facs_for_ic = {
            name: fac.loc[train_start:train_end].copy()
            for name, fac in raw_factors_full.items()
        }
        train_stats = get_factor_ic_stats(facs_for_ic, fwd_train)

        # 根据 train 段 IC 符号，对全期因子做定向
        oriented_full = {}
        for name, fac in raw_factors_full.items():
            mean_ic = train_stats[name]["mean_ic"]
            t_stat = train_stats[name]["t_stat"]
            # 筛选：只用 train 段 t-stat 超过阈值的因子
            if abs(t_stat) >= cfg.ic_screen_threshold:
                oriented_full[name] = -fac if mean_ic < 0 else fac

        if not oriented_full:
            oriented_full = {name: fac for name, fac in raw_factors_full.items()}

        if neutralize_fn is not None:
            oriented_full = {
                name: neutralize_fn(fac) for name, fac in oriented_full.items()
            }

        composite = combine_factors_equal_weight(oriented_full)
        composite_test = composite.loc[test_start:test_end].copy()
        close_test = close_full.loc[test_start:test_end].copy()

        bt = long_top_layer(
            close_test, composite_test,
            rebalance_every=cfg.rebalance,
            weight_mode=cfg.weight_mode,
        )

        all_test_rets.append(bt["port_ret"])
        all_test_bench.append(bt["benchmark_ret"])

        s_test = summary(bt["equity"], bt["port_ret"])
        s_bench = summary((1 + bt["benchmark_ret"]).cumprod(), bt["benchmark_ret"])

        window_info.append({
            "window": window_idx,
            "train_start": str(train_start)[:10],
            "train_end": str(train_end)[:10],
            "test_start": str(test_start)[:10],
            "test_end": str(test_end)[:10],
            "n_factors_selected": len(oriented_full),
            "factors": list(oriented_full.keys()),
            "test_sharpe": s_test["sharpe"],
            "bench_sharpe": s_bench["sharpe"],
        })

        start += cfg.step_days
        window_idx += 1

    full_test_ret = pd.concat(all_test_rets).sort_index()
    full_test_bench = pd.concat(all_test_bench).sort_index()
    full_test_ret = full_test_ret[~full_test_ret.index.duplicated(keep="first")]
    full_test_bench = full_test_bench[~full_test_bench.index.duplicated(keep="first")]

    equity = (1 + full_test_ret).cumprod()
    bench_eq = (1 + full_test_bench).cumprod()
    s = summary(equity, full_test_ret)
    s_bench = summary(bench_eq, full_test_bench)

    return {
        "summary": s,
        "bench_summary": s_bench,
        "excess_sharpe": s["sharpe"] - s_bench["sharpe"],
        "n_windows": len(window_info),
        "windows": window_info,
        "full_test_ret": full_test_ret,
        "full_test_equity": equity,
        "bench_equity": bench_eq,
    }


def run_cn(cfg: WFConfig, do_ic_screen: bool = False):
    """运行 A股 无泄漏 Walk-Forward"""
    from quant.data.universe import DEFAULT_POOL
    from quant.data.panel import build_ohlcv_panels, build_value_panels
    from quant.data.industry import industry_series

    print("=" * 72)
    print(f"  A股 Walk-Forward 验证（IC筛选={do_ic_screen}，"
          f"train={cfg.train_days} test={cfg.test_days} step={cfg.step_days}）")
    print("=" * 72)

    print("\n[1] 加载数据...")
    symbols = DEFAULT_POOL
    panels = build_ohlcv_panels(symbols)
    close = panels["close"]
    value = build_value_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    print(f"    股票数: {len(close.columns)} 日期数: {len(close)}")

    print("\n[2] 构建因子库...")
    raw = build_factors_cn(close, value, panels)
    print(f"    原始因子数: {len(raw)}")

    def neut_fn(fac):
        return neutralize(fac, industry=ind, log_mv=log_mv, mode="full")

    print(f"\n[3] Walk-Forward 回测...")
    result = walk_forward_backtest(close, raw, neut_fn, cfg)

    s = result["summary"]
    sb = result["bench_summary"]
    print(f"\n{'=' * 72}")
    print(f"  结果汇总（{result['n_windows']} 个窗口拼接）")
    print(f"{'=' * 72}")
    print(f"  策略: 累计收益 {s['total_return']:>+8.1%}  年化 {s['annualized_return']:>+7.1%}  ")
    print(f"         夏普 {s['sharpe']:>+6.2f}  最大回撤 {s['max_drawdown']:>+7.1%}")
    print(f"  基准: 累计收益 {sb['total_return']:>+8.1%}  年化 {sb['annualized_return']:>+7.1%}  ")
    print(f"         夏普 {sb['sharpe']:>+6.2f}  最大回撤 {sb['max_drawdown']:>+7.1%}")
    print(f"  超额夏普: {result['excess_sharpe']:+.2f}")
    print(f"\n  各窗口明细:")
    for w in result["windows"]:
        print(f"    [{w['window']:2d}] {w['train_start']}~{w['test_end']}  "
              f"因子数={w['n_factors_selected']:2d}  "
              f"策略夏普={w['test_sharpe']:+.2f}  "
              f"基准夏普={w['bench_sharpe']:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result["full_test_equity"].index, result["full_test_equity"].values,
            label="策略（无泄漏WF）", lw=2, color="crimson")
    ax.plot(result["bench_equity"].index, result["bench_equity"].values,
            label="等权基准", lw=1.5, ls="--", color="gray")
    ax.set_title(f"A股 无泄漏 Walk-Forward 验证（{result['n_windows']}窗口拼接）")
    ax.set_ylabel("净值")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RAW_DATA_DIR / "cn_wf_corrected.png"
    fig.savefig(out, dpi=150)
    print(f"\n图: {out}")
    return result


def run_us(cfg: WFConfig, do_ic_screen: bool = False):
    """运行美股 无泄漏 Walk-Forward"""
    from quant.data import us_loader
    from quant.data.universe_us_expanded import EXPANDED_US_POOL
    from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels

    print("=" * 72)
    print(f"  美股 Walk-Forward 验证（IC筛选={do_ic_screen}，"
          f"train={cfg.train_days} test={cfg.test_days} step={cfg.step_days}）")
    print("=" * 72)

    print("\n[1] 加载数据...")
    symbols = EXPANDED_US_POOL[:300]
    panels = build_ohlcv_panels(symbols, loader=us_loader)
    close = panels["close"]
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    print(f"    股票数: {len(close.columns)} 日期数: {len(close)}")

    print("\n[2] 构建因子库...")
    raw = build_factors_us(close, fund, panels)
    print(f"    原始因子数: {len(raw)}")

    print(f"\n[3] Walk-Forward 回测...")
    result = walk_forward_backtest(close, raw, None, cfg)

    s = result["summary"]
    sb = result["bench_summary"]
    print(f"\n{'=' * 72}")
    print(f"  结果汇总（{result['n_windows']} 个窗口拼接）")
    print(f"{'=' * 72}")
    print(f"  策略: 累计收益 {s['total_return']:>+8.1%}  年化 {s['annualized_return']:>+7.1%}  ")
    print(f"         夏普 {s['sharpe']:>+6.2f}  最大回撤 {s['max_drawdown']:>+7.1%}")
    print(f"  基准: 累计收益 {sb['total_return']:>+8.1%}  年化 {sb['annualized_return']:>+7.1%}  ")
    print(f"         夏普 {sb['sharpe']:>+6.2f}  最大回撤 {sb['max_drawdown']:>+7.1%}")
    print(f"  超额夏普: {result['excess_sharpe']:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result["full_test_equity"].index, result["full_test_equity"].values,
            label="策略（无泄漏WF）", lw=2, color="teal")
    ax.plot(result["bench_equity"].index, result["bench_equity"].values,
            label="等权基准", lw=1.5, ls="--", color="gray")
    ax.set_title(f"美股 无泄漏 Walk-Forward 验证（{result['n_windows']}窗口拼接）")
    ax.set_ylabel("净值")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RAW_DATA_DIR / "us_wf_corrected.png"
    fig.savefig(out, dpi=150)
    print(f"\n图: {out}")
    return result


def main():
    parser = argparse.ArgumentParser(description="无泄漏 Walk-Forward Runner")
    parser.add_argument("--market", type=str, default="CN", choices=["CN", "US", "HK"])
    parser.add_argument("--train", type=int, default=480, help="train 窗口天数")
    parser.add_argument("--test", type=int, default=120, help="test 窗口天数")
    parser.add_argument("--step", type=int, default=120, help="滚动步长")
    parser.add_argument("--ic-screen", action="store_true", help="启用 IC 筛选（只用 train 段）")
    parser.add_argument("--ic-threshold", type=float, default=1.5, help="IC 筛选 t-stat 阈值")
    args = parser.parse_args()

    cfg = WFConfig(
        train_days=args.train,
        test_days=args.test,
        step_days=args.step,
        ic_screen_threshold=args.ic_threshold,
    )

    if args.market == "CN":
        run_cn(cfg, args.ic_screen)
    elif args.market == "US":
        run_us(cfg, args.ic_screen)
    else:
        print(f"Market {args.market} 暂未实现")
        sys.exit(1)


if __name__ == "__main__":
    from quant.config import RAW_DATA_DIR
    main()
