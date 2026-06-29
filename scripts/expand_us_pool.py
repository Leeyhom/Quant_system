"""expand_us_pool —— 批量拉取扩展美股池的行情+基本面数据并落地缓存。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/expand_us_pool.py

做什么：
    ① 对 EXPANDED_US_POOL 中所有新股票，检查本地缓存是否已存在。
    ② 缺失则联网拉取行情(us_loader.fetch_daily→保存Parquet)和基本面
       (us_fundamental_loader.fetch_fundamental→保存Parquet)。
    ③ 容错：单只失败不中断整批，最后汇总可用率。

约 430 只新股票 × 2 个接口 ≈ 860 次 API 调用，按 ~0.5s/次 估算 ~7 分钟。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from quant.data import us_loader
from quant.data import us_fundamental_loader
from quant.data.universe_us_expanded import EXPANDED_US_POOL, LEGACY_US_POOL
from quant.config import HISTORY_START, HISTORY_END

# 哪些是新增的（本地可能无缓存）
NEW_SYMBOLS = [s for s in EXPANDED_US_POOL if s not in set(LEGACY_US_POOL)]


def _has_price_cache(sym: str) -> bool:
    try:
        us_loader.load_parquet(sym)
        return True
    except FileNotFoundError:
        return False


def _has_fund_cache(sym: str) -> bool:
    try:
        us_fundamental_loader.load_parquet(sym)
        return True
    except FileNotFoundError:
        return False


def main(argv: list[str]) -> None:
    print(f"扩展池总计: {len(EXPANDED_US_POOL)} 只（原池 {len(LEGACY_US_POOL)} + 新增 {len(NEW_SYMBOLS)}）")

    # ── 行情拉取 ──
    need_price = [s for s in NEW_SYMBOLS if not _has_price_cache(s)]
    print(f"\n{'='*60}")
    print(f"行情拉取: {len(need_price)} 只需要联网（{len(NEW_SYMBOLS)-len(need_price)} 已缓存）")
    print(f"{'='*60}")

    price_ok, price_fail = [], []
    t0 = time.time()
    for i, sym in enumerate(need_price):
        try:
            df = us_loader.fetch_daily(sym, start=HISTORY_START, end=HISTORY_END)
            us_loader.save_parquet(df, sym)
            price_ok.append(sym)
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (len(need_price) - i - 1)
            print(f"  [{i+1:3d}/{len(need_price)}] {sym:6s} ✅ {len(df):4d}天 "
                  f"| 耗时{elapsed:.0f}s | 预计剩余{eta:.0f}s", flush=True)
        except Exception as e:
            price_fail.append((sym, f"{type(e).__name__}: {e}"))
            print(f"  [{i+1:3d}/{len(need_price)}] {sym:6s} ❌ {type(e).__name__}", flush=True)
            time.sleep(0.3)  # 遇错稍等再继续

    print(f"\n行情结果: 成功 {len(price_ok)}, 失败 {len(price_fail)}")
    if price_fail:
        print(f"  失败列表: {price_fail[:10]}")

    # ── 基本面拉取 ──
    # 对行情拉取成功的票 + 原池中缺基本面的票，尝试拉基本面
    all_with_price = LEGACY_US_POOL + price_ok
    need_fund = [s for s in all_with_price if not _has_fund_cache(s)]
    print(f"\n{'='*60}")
    print(f"基本面拉取: {len(need_fund)} 只需要联网")
    print(f"{'='*60}")

    fund_ok, fund_fail = [], []
    t0 = time.time()
    for i, sym in enumerate(need_fund):
        try:
            df = us_fundamental_loader.fetch_fundamental(sym)
            us_fundamental_loader.save_parquet(df, sym)
            fund_ok.append(sym)
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (len(need_fund) - i - 1)
            print(f"  [{i+1:3d}/{len(need_fund)}] {sym:6s} ✅ {len(df):2d}季 "
                  f"| 耗时{elapsed:.0f}s | 预计剩余{eta:.0f}s", flush=True)
        except Exception as e:
            fund_fail.append((sym, f"{type(e).__name__}: {e}"))
            print(f"  [{i+1:3d}/{len(need_fund)}] {sym:6s} ❌ {type(e).__name__}", flush=True)
            time.sleep(0.3)

    print(f"\n基本面结果: 成功 {len(fund_ok)}, 失败 {len(fund_fail)}")
    if fund_fail:
        print(f"  失败列表: {fund_fail[:20]}")

    # ── 最终统计 ──
    # 复检：哪些票两者缓存都有
    final_price = [s for s in EXPANDED_US_POOL if _has_price_cache(s)]
    final_fund = [s for s in EXPANDED_US_POOL if _has_fund_cache(s)]
    final_both = [s for s in final_price if s in final_fund]
    print(f"\n{'='*60}")
    print(f"扩展池最终覆盖:")
    print(f"  行情: {len(final_price)}/{len(EXPANDED_US_POOL)}")
    print(f"  基本面: {len(final_fund)}/{len(EXPANDED_US_POOL)}")
    print(f"  两者均有: {len(final_both)}/{len(EXPANDED_US_POOL)}")
    print(f"  有行情无基本面 (多为金融股): {[s for s in final_price if s not in final_fund]}")
    print(f"完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
