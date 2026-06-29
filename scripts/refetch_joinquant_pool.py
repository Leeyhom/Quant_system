"""refetch_joinquant_pool —— 拉取聚宽策略股票池的本地行情和估值缓存。

v7 失败暴露出一个重要问题：本地验证使用 DEFAULT_POOL 的 89 只有效股票，
而聚宽策略实际股票池是 152 只。不同池子会改变横截面排名，也会让参数搜索
在本地看起来很好、到聚宽却失真。

用法：
    conda activate quant
    NO_PROXY='*' PYTHONPATH=. python scripts/refetch_joinquant_pool.py
    NO_PROXY='*' PYTHONPATH=. python scripts/refetch_joinquant_pool.py --limit 10

说明：
    - 行情接口仍使用 akshare 新浪 daily loader；
    - 估值接口仍使用项目约定的 stock_value_em；
    - 本地估值历史通常从 2018-01 起，这是当前 A股本地因子研究的硬边界。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.config import HISTORY_END, HISTORY_START
from quant.data.akshare_loader import fetch_daily, save_parquet
from quant.data.fundamental_loader import fetch_value, save_value_parquet


DEFAULT_STRATEGY = PROJECT_ROOT / "scripts" / "joinquant_cn_sim_strategy_v8.py"


def load_stock_pool(strategy_path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("jq_strategy", strategy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [s.split(".")[0] for s in module.STOCK_POOL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = load_stock_pool(args.strategy)
    if args.limit:
        symbols = symbols[: args.limit]

    print(f"重拉聚宽策略池 {len(symbols)} 只 {HISTORY_START}~{HISTORY_END} 行情+估值（覆盖旧缓存）")
    ok_price = ok_value = 0
    price_rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_daily(sym, start=HISTORY_START, end=HISTORY_END)
            save_parquet(df, sym)
            ok_price += 1
            price_rows.append((sym, df["date"].min(), df["date"].max(), len(df)))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 行情跳过 {sym}: {type(exc).__name__}: {exc}")

        try:
            v = fetch_value(sym, start=HISTORY_START, end=HISTORY_END)
            save_value_parquet(v, sym)
            ok_value += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 估值跳过 {sym}: {type(exc).__name__}: {exc}")

        if i % 10 == 0:
            print(f"  ...{i}/{len(symbols)}")

    print(f"\n完成：行情 {ok_price}/{len(symbols)} 估值 {ok_value}/{len(symbols)}")
    if price_rows:
        starts = pd.Series([r[1] for r in price_rows])
        ends = pd.Series([r[2] for r in price_rows])
        rows = pd.Series([r[3] for r in price_rows])
        print(f"行情起始日: 最早 {starts.min().date()} / 中位 {starts.median().date()} / 最晚 {starts.max().date()}")
        print(f"行情结束日: 最早 {ends.min().date()} / 最晚 {ends.max().date()}")
        print(f"每票行数: 中位 {int(rows.median())} (≈{rows.median()/250:.1f}年)")


if __name__ == "__main__":
    main()
