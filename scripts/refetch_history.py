"""refetch_history —— 把全池行情+估值缓存重拉到 HISTORY_START（M14）。

为什么需要单独一个脚本：`panel.py` 的加载是**缓存优先**的——只要本地存在
{symbol}.parquet 就直接读回，不管它覆盖的日期窗口。M13 时缓存只到 2024+，
要扩历史就必须**强制覆盖重拉**，否则面板仍是旧的 2 年。

用法：
    conda activate quant
    NO_PROXY='*' python scripts/refetch_history.py            # 全池
    NO_PROXY='*' python scripts/refetch_history.py --limit 10  # 先小步验证

幂等：重复跑会重新拉取并覆盖。单只失败（停牌/退市/接口异常）打印告警跳过，
不中断整池——这也顺带暴露幸存者偏差（退市股本就拉不到，池子是手工静态池）。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from quant.config import RAW_DATA_DIR, HISTORY_START, HISTORY_END
from quant.data.universe import DEFAULT_POOL
from quant.data.akshare_loader import fetch_daily, save_parquet
from quant.data.fundamental_loader import fetch_value, save_value_parquet

from scripts.factor_research_demo import parse_limit


def main(argv: list[str]) -> None:
    limit = parse_limit(argv)
    symbols = DEFAULT_POOL[:limit] if limit else DEFAULT_POOL
    print(f"重拉 {len(symbols)} 只 {HISTORY_START}~{HISTORY_END} 行情+估值（覆盖旧缓存）")

    ok_price = ok_value = 0
    price_rows = []
    for i, sym in enumerate(symbols, 1):
        # 行情
        try:
            df = fetch_daily(sym, start=HISTORY_START, end=HISTORY_END)
            save_parquet(df, sym)
            ok_price += 1
            price_rows.append((sym, df["date"].min(), df["date"].max(), len(df)))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 行情跳过 {sym}：{type(exc).__name__}: {exc}")
        # 估值
        try:
            v = fetch_value(sym, start=HISTORY_START, end=HISTORY_END)
            save_value_parquet(v, sym)
            ok_value += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 估值跳过 {sym}：{type(exc).__name__}: {exc}")
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
        # 上市晚于 2019 的票（历史不全，横截面早期会是 NaN，提示幸存/上市偏差）
        late = [(r[0], r[1].date()) for r in price_rows if r[1] > pd.Timestamp("2019-01-01")]
        if late:
            print(f"⚠️ {len(late)} 只起始晚于 2019（上市晚/数据缺），早期截面参与度低：")
            print("   " + ", ".join(f"{s}({d})" for s, d in late[:15]) + (" ..." if len(late) > 15 else ""))
    print("缓存已更新 ✅，下游 demo 直接读回即覆盖全历史。")


if __name__ == "__main__":
    main(sys.argv[1:])
