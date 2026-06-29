"""fetch_dynamic_pool —— 批量拉取全市场动态池(2073只)的行情+基本面。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/fetch_dynamic_pool.py

特性：
    - 自动跳过已有缓存的股票
    - 单只失败不中断，容错继续
    - 进度+ETA显示
    - 支持 --resume 从中断处继续
"""

from __future__ import annotations

import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.data import us_loader, us_fundamental_loader
from quant.config import HISTORY_START, HISTORY_END


def _has_cache(sym, kind="price"):
    try:
        if kind == "price":
            us_loader.load_parquet(sym)
        else:
            us_fundamental_loader.load_parquet(sym)
        return True
    except FileNotFoundError:
        return False


def main(argv):
    with open(PROJECT_ROOT / "data" / "raw" / "us_dynamic_tickers.txt") as f:
        all_tickers = [l.strip() for l in f if l.strip()]

    # 过滤需要拉取的
    need_price = [t for t in all_tickers if not _has_cache(t, "price")]
    need_fund = [t for t in all_tickers if not _has_cache(t, "fund")]

    # 支持 --resume：从中断处继续
    if "--resume" in argv:
        # 从最后一个成功缓存的股票之后继续
        pass  # 自动处理（上面的检查已做）

    print(f"动态池: {len(all_tickers)} 只")
    print(f"需拉行情: {len(need_price)} | 需拉基本面: {len(need_fund)}")
    total = len(need_price) + len(need_fund)
    print(f"总API调用: {total} | 预估 {total*0.4/60:.0f} 分钟\n")

    # ── 行情 ──
    if need_price:
        print(f"{'='*50}")
        print(f"行情拉取: {len(need_price)} 只")
        print(f"{'='*50}")
        ok, fail = 0, 0
        t0 = time.time()
        for i, sym in enumerate(need_price):
            try:
                df = us_loader.fetch_daily(sym, start=HISTORY_START, end=HISTORY_END)
                us_loader.save_parquet(df, sym)
                ok += 1
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (len(need_price) - i - 1)
                    print(f"  [{i+1}/{len(need_price)}] {ok}✅ {fail}❌ | {elapsed:.0f}s | ETA {eta:.0f}s")
            except Exception as e:
                fail += 1
                if fail <= 10:
                    print(f"  [{i+1}/{len(need_price)}] {sym} ❌ {type(e).__name__}")
        print(f"行情完成: {ok}✅ {fail}❌")

    # ── 基本面 ──
    if need_fund:
        print(f"\n{'='*50}")
        print(f"基本面拉取: {len(need_fund)} 只")
        print(f"{'='*50}")
        ok, fail = 0, 0
        t0 = time.time()
        for i, sym in enumerate(need_fund):
            try:
                df = us_fundamental_loader.fetch_fundamental(sym)
                us_fundamental_loader.save_parquet(df, sym)
                ok += 1
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (len(need_fund) - i - 1)
                    print(f"  [{i+1}/{len(need_fund)}] {ok}✅ {fail}❌ | {elapsed:.0f}s | ETA {eta:.0f}s")
            except Exception as e:
                fail += 1
                if fail <= 20:
                    print(f"  [{i+1}/{len(need_fund)}] {sym} ❌ {type(e).__name__}")
        print(f"基本面完成: {ok}✅ {fail}❌")

    # 最终统计
    final_price = sum(1 for t in all_tickers if _has_cache(t, "price"))
    final_fund = sum(1 for t in all_tickers if _has_cache(t, "fund"))
    both = sum(1 for t in all_tickers if _has_cache(t, "price") and _has_cache(t, "fund"))
    print(f"\n最终覆盖: 行情{final_price} | 基本面{final_fund} | 两者均有{both}")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1:])
