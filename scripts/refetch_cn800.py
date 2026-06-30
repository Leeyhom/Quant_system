#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""refetch_cn800 —— 拉取 CN800 扩展池全部历史行情+估值缓存。

800 只股票 × 2 个接口 ≈ 1600 次 API 调用，按 ~0.5s/次估算约 13 分钟。
实际受 akshare 接口限流和网络波动影响，预计 15-30 分钟。

用法:
    conda activate quant
    NO_PROXY='*' PYTHONPATH=. python scripts/refetch_cn800.py
    NO_PROXY='*' PYTHONPATH=. python scripts/refetch_cn800.py --limit 10   # 先小步验证
    NO_PROXY='*' PYTHONPATH=. python scripts/refetch_cn800.py --skip-existing  # 只拉缺失的

容错: 单只失败不中断，最后汇总成功率。停牌/退市/接口异常自动跳过。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from quant.config import HISTORY_START, HISTORY_END
from quant.data.universe_cn800 import CN800_POOL
from quant.data.akshare_loader import fetch_daily, save_parquet, _parquet_path as price_path
from quant.data.fundamental_loader import fetch_value, save_value_parquet, _value_parquet_path as value_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="拉取 CN800 扩展池历史数据")
    p.add_argument("--limit", type=int, default=0, help="只拉前 N 只（验证用）")
    p.add_argument("--skip-existing", action="store_true", help="跳过已有缓存的股票")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = CN800_POOL[:args.limit] if args.limit else CN800_POOL

    skipped_price = skipped_value = 0
    if args.skip_existing:
        _old = len(symbols)
        symbols = [s for s in symbols if not price_path(s).exists() or not value_path(s).exists()]
        skipped_price = _old - len([s for s in CN800_POOL if not price_path(s).exists()])
        skipped_value = _old - len([s for s in CN800_POOL if not value_path(s).exists()])

    n = len(symbols)
    est_sec = n * 2 * 0.7  # ~0.7s per API call average
    print(f"拉取 {n} 只股票 {HISTORY_START}~{HISTORY_END} 行情+估值")
    if args.skip_existing:
        print(f"  （跳过已有: 行情{skipped_price}只 / 估值{skipped_value}只）")
    print(f"  预计 {n*2} 次API调用，约 {est_sec/60:.0f} 分钟\n")

    ok_price = ok_value = 0
    fail_price: list[tuple[str, str]] = []
    fail_value: list[tuple[str, str]] = []
    t_start = time.time()

    for i, sym in enumerate(symbols, 1):
        # ── 行情 ──
        try:
            df = fetch_daily(sym, start=HISTORY_START, end=HISTORY_END)
            save_parquet(df, sym)
            ok_price += 1
        except Exception as e:
            fail_price.append((sym, f"{type(e).__name__}: {e}"))

        # ── 估值 ──
        try:
            v = fetch_value(sym, start=HISTORY_START, end=HISTORY_END)
            save_value_parquet(v, sym)
            ok_value += 1
        except Exception as e:
            fail_value.append((sym, f"{type(e).__name__}: {e}"))

        # ── 进度 ──
        if i % 20 == 0 or i == n:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n - i) / rate if rate > 0 else 0
            print(
                f"  [{i:3d}/{n}] "
                f"行情{ok_price}✅{len(fail_price)}❌ "
                f"估值{ok_value}✅{len(fail_value)}❌ "
                f"| {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining",
                flush=True,
            )

        # 温和限速：避免被 akshare/东财封 IP
        time.sleep(0.15)

    # ── 汇总 ──
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  完成！耗时 {elapsed/60:.1f} 分钟")
    print(f"  行情: {ok_price}✅ / {len(fail_price)}❌ ({ok_price/n*100:.1f}%)")
    print(f"  估值: {ok_value}✅ / {len(fail_value)}❌ ({ok_value/n*100:.1f}%)")
    print(f"{'='*60}")

    if fail_price:
        print(f"\n  行情失败 ({len(fail_price)} 只):")
        for sym, err in fail_price[:20]:
            print(f"    {sym}: {err}")
        if len(fail_price) > 20:
            print(f"    ... 还有 {len(fail_price)-20} 只")

    if fail_value:
        print(f"\n  估值失败 ({len(fail_value)} 只):")
        for sym, err in fail_value[:20]:
            print(f"    {sym}: {err}")
        if len(fail_value) > 20:
            print(f"    ... 还有 {len(fail_value)-20} 只")

    # ── 日期覆盖统计 ──
    ok_syms = [s for s in symbols if s not in {f[0] for f in fail_price}]
    if ok_syms:
        print(f"\n  行情覆盖统计（抽样10只）:")
        for sym in ok_syms[:10]:
            try:
                df = pd.read_parquet(price_path(sym))
                print(f"    {sym}: {len(df):4d} 天  {df['date'].min().date()} ~ {df['date'].max().date()}")
            except Exception:
                pass

    print("\n  ✅ CN800 数据拉取完成，可以开始本地验证")


if __name__ == "__main__":
    main()
