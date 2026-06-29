"""factor_research_demo —— 因子 IC 与分层回测（M7 框架 + M8 扩池/量价因子）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/factor_research_demo.py            # 默认全池(~90只)
    NO_PROXY='*' python scripts/factor_research_demo.py --limit 20 # 先小步验证

目标：先判断因子本身是否有稳定预测力，再考虑组合/复杂模型。
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
from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels
from quant.factor import factors as F
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary, cumulative_ic
from quant.backtest.layered import layered_backtest, layer_summary, is_monotonic_by_return

HORIZON = 20
N_LAYERS = 5


def parse_limit(argv: list[str]) -> int | None:
    """解析 --limit N；无依赖的极简解析。"""
    if "--limit" in argv:
        i = argv.index("--limit")
        if i + 1 < len(argv):
            return int(argv[i + 1])
    return None


def build_factors(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = panels["close"]
    high, low = panels["high"], panels["low"]
    volume, amount = panels["volume"], panels["amount"]
    return {
        # 价格类（M7 已有）
        "momentum20": F.momentum(close, 20),
        "momentum60": F.momentum(close, 60),
        "reversal20": F.reversal(close, 20),
        "lowvol20": F.low_volatility(close, 20),
        # 量能
        "volume_trend": F.volume_trend(volume, 5, 20),
        "amount_liquidity": F.amount_liquidity(amount, 20),
        # 波动
        "parkinson_vol": F.parkinson_volatility(high, low, 20),
        "atr_vol": F.atr_volatility(high, low, close, 20),
        # 趋势
        "ma_slope20": F.ma_slope(close, 20),
        "price_to_ma20": F.price_to_ma(close, 20),
        # 流动性冲击
        "amihud": F.amihud_illiquidity(close, amount, 20),
    }


def _fmt_ic(row: pd.Series) -> str:
    return (
        f"meanIC {row['mean_ic']:+.3f} | ICIR {row['icir']:+.2f} | "
        f"正IC {row['positive_rate']:.0%} | t {row['t_stat']:+.2f} | n {int(row['n'])}"
    )


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = DEFAULT_POOL[:limit] if limit else DEFAULT_POOL
    print(f"[1/5] 构建股票池面板（{len(symbols)} 只，limit={limit}）...")
    panels = build_ohlcv_panels(symbols, start="20240101", end="20251231")
    close = panels["close"]
    print(f"      面板形状: {close.shape[0]} 天 × {close.shape[1]} 只")

    print(f"[2/5] 计算未来 {HORIZON} 日收益与候选因子 ...")
    fwd = forward_returns(close, horizon=HORIZON)
    facs = build_factors(panels)

    print("[3/5] 计算 Spearman Rank IC（按 |meanIC| 排序）...")
    ic_series = {}
    rows = []
    for name, factor in facs.items():
        ic = daily_ic(factor, fwd, method="spearman", min_count=5)
        ic_series[name] = ic
        rows.append({"factor": name, **ic_summary(ic)})
    ic_table = pd.DataFrame(rows)
    ic_table["abs_ic"] = ic_table["mean_ic"].abs()
    ic_table = ic_table.sort_values("abs_ic", ascending=False)
    for _, row in ic_table.iterrows():
        print(f"      {row['factor']:<16} {_fmt_ic(row)}")

    # 选 |meanIC| 最大的因子做分层；若 meanIC<0 则反向使用（分数取负）
    best = ic_table.iloc[0]
    best_name = best["factor"]
    best_factor = facs[best_name]
    reversed_flag = best["mean_ic"] < 0
    if reversed_flag:
        best_factor = -best_factor
    print(f"\n      分层演示因子: {best_name}"
          f"{'（IC为负，反向使用）' if reversed_flag else ''}")

    print(f"[4/5] 对该因子做 {N_LAYERS} 分层回测 ...")
    layered = layered_backtest(close, best_factor, n_layers=N_LAYERS, rebalance_every=HORIZON)
    layers = layer_summary(layered, n_layers=N_LAYERS)
    mono = is_monotonic_by_return(layers, n_layers=N_LAYERS)
    for _, r in layers.iterrows():
        print(
            f"      {r['layer']:<10} 总收益 {r['total_return']:+.2%} | "
            f"最大回撤 {r['max_drawdown']:.2%} | 夏普 {r['sharpe']:.2f}"
        )
    print(f"      分层收益是否 L1→L{N_LAYERS} 单调递增: {'是' if mono else '否'}")

    print("[5/5] 保存累计 IC 图与分层净值图 ...")
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, ic in ic_series.items():
        cum = cumulative_ic(ic)
        ax.plot(cum.index, cum.values, label=name, linewidth=1)
    ax.set_title(f"Cumulative Rank IC (horizon={HORIZON}, {len(symbols)} stocks)")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative IC")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    ic_png = RAW_DATA_DIR / "factor_ic_cum.png"
    fig.savefig(ic_png, dpi=120)

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in [f"L{i}" for i in range(1, N_LAYERS + 1)] + ["top_bottom"]:
        ax.plot(layered["equity"].index, layered["equity"][col], label=col)
    ax.set_title(f"Layered backtest: {best_name}{' (reversed)' if reversed_flag else ''}")
    ax.set_xlabel("date")
    ax.set_ylabel("net value")
    ax.legend()
    fig.tight_layout()
    layer_png = RAW_DATA_DIR / "factor_layers.png"
    fig.savefig(layer_png, dpi=120)

    print(f"      累计IC图: {ic_png}")
    print(f"      分层净值图: {layer_png}")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
