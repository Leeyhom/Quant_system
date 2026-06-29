"""batch_demo —— C：多标的批量回测 + D：止损开关 A/B 对比。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/batch_demo.py   # 首次会联网拉取一篮子股票

逻辑：
  1) 在一篮子股票上跑均值回归（无止损），看胜率/中位数 —— 验证是否普适。
  2) 同一篮子再跑「均值回归 + 8% 止损」，对比止损是否改善整体表现。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.strategy.mean_reversion import mean_reversion_signal
from quant.backtest.batch import batch_backtest, cross_section_stats

# 一篮子沪深主要股票（不同行业，避免单一行业偏差）
SYMBOLS = [
    "600519",  # 贵州茅台
    "000001",  # 平安银行
    "600036",  # 招商银行
    "601318",  # 中国平安
    "000651",  # 格力电器
    "600276",  # 恒瑞医药
    "002415",  # 海康威视
    "600900",  # 长江电力
    "601012",  # 隆基绿能
    "000333",  # 美的集团
]


def _print_stats(title: str, stats: dict) -> None:
    print(f"\n[{title}]")
    print(f"  标的数 {stats['n']} | 跑赢买入持有比例 {stats['win_rate_vs_bh']:.0%}")
    print(f"  策略收益中位数 {stats['median_strat_return']:+.2%} | "
          f"夏普中位数 {stats['median_strat_sharpe']:.2f}")
    print(f"  (对照) 买入持有收益中位数 {stats['median_bh_return']:+.2%}")


def main() -> None:
    print(f"批量回测 {len(SYMBOLS)} 只股票，均值回归策略 ...")

    # 1) 无止损
    res_no = batch_backtest(SYMBOLS, lambda d: mean_reversion_signal(d, window=20, entry_z=1.0))
    _print_stats("均值回归 · 无止损", cross_section_stats(res_no))

    # 2) 8% 止损
    res_sl = batch_backtest(
        SYMBOLS,
        lambda d: mean_reversion_signal(d, window=20, entry_z=1.0, stop_loss=0.08),
    )
    _print_stats("均值回归 · 8%止损", cross_section_stats(res_sl))

    # 明细对比（按无止损收益排序）
    print("\n逐股票明细（收益: 无止损 -> 有止损）：")
    merged = res_no[["symbol", "strat_return"]].rename(columns={"strat_return": "no_sl"})
    merged["with_sl"] = res_sl["strat_return"].values
    for _, r in merged.sort_values("no_sl", ascending=False).iterrows():
        print(f"  {r['symbol']}: {r['no_sl']:+.2%} -> {r['with_sl']:+.2%}")
    print("\n完成 ✅")


if __name__ == "__main__":
    main()
