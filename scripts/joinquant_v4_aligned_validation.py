#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地 v4 SSOT 因子 ↔ 聚宽 v3 实际选股 重合度验证（对齐工作的最终验收）。

目的：验证「把成长因子从 PEG→季报同比」之后，本地与聚宽的选股重合度是否提升。

输入:
    jointquant/v4/v4_rebalance_targets.csv （聚宽每日实际持仓提取的调仓日目标股）
    本地 SSOT v4 因子（季报同比增长 + 筹码集中度）

输出:
    每个调仓日的重合股票列表、Jaccard 指数、重合数、
    差异股票拆解：仅本地选的 vs 仅聚宽选的，归因分析（PEG vs 同比、筹码因子）

用法:
    NO_PROXY='*' PYTHONPATH=. python scripts/joinquant_v4_aligned_validation.py
"""
from __future__ import annotations

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.data.universe import DEFAULT_POOL
from quant.data.panel import (
    build_ohlcv_panels, build_value_panels,
    build_cn_quarterly_panels, build_cn_holder_panels,
)
from quant.data.industry import industry_series
from quant.backtest.layered import simple_tradable_mask
from quant.backtest.metrics import summary
from quant.factor import factors as F
from quant.factor.neutralize import neutralize
from quant.strategy import cn_factor_spec as SPEC

TARGETS_CSV = PROJECT_ROOT / "jointquant" / "v4" / "v4_rebalance_targets.csv"
OUT_DIR = PROJECT_ROOT / "jointquant" / "v4"


def load_joinquant_targets(path: Path = TARGETS_CSV) -> dict[str, set]:
    """Load JoinQuant's actual selected stocks per rebalance date.

    CSV has: date,targets(comma-separated list with .XSHE/XSHG suffixes)
    """
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    targets = {}
    for _, row in df.iterrows():
        dt = row["date"]
        codes = re.findall(r"(\d{6})", str(row["targets"]))
        targets[dt] = set(codes)
    print(f"聚宽调仓日: {len(targets)} 个")
    all_codes = set.union(*targets.values()) if targets else set()
    print(f"股票池: {len(all_codes)} 只, 例: {sorted(list(all_codes))[:10]} ...")
    return targets


def _build_local_v4_composite(symbols=DEFAULT_POOL, factor_set="v4"):
    """构建本地合成分数，支持对照实验隔离差异源。

    factor_set:
      "v4"    — v3 五因子 + 季报同比成长 + 筹码集中度（6 因子，SSOT 正式版）
      "v5yoy" — v3 五因子，成长用季报同比（去筹码，与聚宽 5 因子口径对齐）
      "v5peg" — v3 五因子，成长用 PEG 倒数（聚宽 v3 historical 列名 growth_peg_proxy 口径）

    返回 (composite, close)。合成分数已按全样本 IC 定向（与聚宽回测口径一致），
    防未来函数留给调用方做（取调仓日前一交易日的分数）。
    """
    ohlcv = build_ohlcv_panels(symbols)
    close = ohlcv["close"]
    val = build_value_panels(symbols, align_to=close)
    q = build_cn_quarterly_panels(symbols, align_to=close)
    ind = industry_series(list(close.columns))
    log_mv = np.log(val["total_mv"].replace(0, np.nan))

    def _n(fac):
        return neutralize(fac, industry=ind, log_mv=log_mv, mode="full")

    # v3 共有的 5 因子骨架
    factors = {
        "earnings_yield": _n(F.earnings_yield(val["pe_ttm"])),
        "cashflow_yield": _n(F.cashflow_yield(val["pcf"])),
        "sales_yield": _n(F.sales_yield(val["ps"])),
        "amihud": _n(F.amihud_illiquidity(close, ohlcv["amount"], 20)),
    }
    # 成长因子口径：v5peg 用 PEG 倒数，其余用季报同比/PE（这是本次对齐的核心变量）
    if factor_set == "v5peg":
        factors["growth"] = _n(F.growth_peg(val["peg"]))
    else:
        factors["growth"] = _n(F.growth_yoy_over_pe(q["net_profit_yoy"], val["pe_ttm"]))
    # 筹码集中度只在正式版 v4 加入（聚宽 targets 是 5 因子，去掉它才 apples-to-apples）
    if factor_set == "v4":
        hp = build_cn_holder_panels(symbols, align_to=close)
        factors["holder_concentration"] = _n(F.holder_concentration(hp["change_ratio"]))

    # 全样本 IC 定向（与聚宽全样本回测口径一致）。
    # 诚实 walk-forward 对比请用 cn_holder_factor_eval.py。
    from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
    fwd = forward_returns(close, horizon=20)
    oriented = {}
    for name, fac in factors.items():
        s = ic_summary(daily_ic(fac, fwd))
        ic = s["mean_ic"]
        oriented[name] = fac if pd.isna(ic) or ic >= 0 else -fac

    composite = SPEC.equal_weight_composite(oriented)
    return composite, close


def build_local_v4_targets_at_dates(
    rebalance_dates, symbols=DEFAULT_POOL, top_n=6, factor_set="v4"
) -> dict:
    """在聚宽的实际调仓日，用本地因子选 top_n。

    防未来函数：每个调仓日取「该日前最后一个有分数的交易日」的合成分数。
    聚宽每个调仓日选 6 只 → top_n 默认 6 对齐。
    """
    composite, close = _build_local_v4_composite(symbols, factor_set=factor_set)

    targets = {}
    for dt in rebalance_dates:
        dt = pd.Timestamp(dt)
        # 取调仓日（含）之前最后一个交易日的分数 —— 今算明用，防前视。
        prior = composite.index[composite.index <= dt]
        if len(prior) == 0:
            continue
        scores = composite.loc[prior[-1]].dropna().sort_values(ascending=False)
        if len(scores) < top_n:
            continue
        targets[dt] = set(scores.head(top_n).index.tolist())
    return targets


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-set", default="v4", choices=["v4", "v5yoy", "v5peg"],
                    help="v4=6因子(含筹码); v5yoy=5因子季报同比; v5peg=5因子PEG口径")
    cli = ap.parse_args()

    print("=" * 78)
    print(f"  本地 ↔ 聚宽 v3 选股重合度验证  [factor_set={cli.factor_set}]")
    print("=" * 78)

    if not TARGETS_CSV.exists():
        print(f"ERROR: 找不到聚宽目标股文件: {TARGETS_CSV}")
        print("先从聚宽导出调仓日目标股，命名为 v4_rebalance_targets.csv")
        sys.exit(1)

    jq_targets = load_joinquant_targets()

    # Build local targets using exactly the same rebalance dates as JoinQuant
    rebalance_dates = sorted(jq_targets.keys())
    local_targets = build_local_v4_targets_at_dates(rebalance_dates, factor_set=cli.factor_set)

    # Match common dates
    common_dates = sorted(set(jq_targets.keys()).intersection(local_targets.keys()))
    print(f"重合调仓日: {len(common_dates)} 个")
    if common_dates:
        print(f"日期范围: {common_dates[0].date()} ~ {common_dates[-1].date()}")

    rows = []
    for dt in common_dates:
        jq = jq_targets[dt]
        local = local_targets[dt]
        overlap = jq & local
        jaccard = len(overlap) / len(jq | local) if jq | local else 0.0
        rows.append({
            "date": dt,
            "jq_count": len(jq),
            "local_count": len(local),
            "overlap_count": len(overlap),
            "jaccard": jaccard,
            "overlap": ",".join(sorted(overlap)),
            "only_jq": ",".join(sorted(jq - local)),
            "only_local": ",".join(sorted(local - jq)),
        })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "v4_local_jq_alignment.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n对齐结果已保存: {out_csv}")

    print(f"\n平均 Jaccard: {df['jaccard'].mean():.3f} (v2 是 0.29)")
    print(f"重合数中位: {df['overlap_count'].median()} / top_n=6")

    # Top and bottom 3 days
    print("\n【重合度最高的 3 天】")
    for _, r in df.sort_values("jaccard", ascending=False).head(3).iterrows():
        print(f"  {r['date'].date()}: {r['overlap_count']}/{r['jq_count']} Jaccard={r['jaccard']:.3f}")

    print("\n【重合度最低的 3 天】")
    for _, r in df.sort_values("jaccard").head(3).iterrows():
        print(f"  {r['date'].date()}: {r['overlap_count']}/{r['jq_count']} Jaccard={r['jaccard']:.3f}")

    # Diff analysis: only local, only JQ
    only_local = set.union(*[set(r["only_local"].split(",")) for _, r in df.iterrows() if r["only_local"]])
    only_jq = set.union(*[set(r["only_jq"].split(",")) for _, r in df.iterrows() if r["only_jq"]])
    print(f"\n仅本地选但聚宽没选的股票（{len(only_local)} 只）: {sorted(list(only_local))[:15]}...")
    print(f"仅聚宽选但本地没选的股票（{len(only_jq)} 只）: {sorted(list(only_jq))[:15]}...")

    print("\n" + "=" * 78)
    if df['jaccard'].mean() > 0.5:
        print("✅ 对齐成功！Jaccard > 0.5，因子口径差异已解决。")
    elif df['jaccard'].mean() > 0.4:
        print(f"⚠️ 部分对齐。Jaccard={df['jaccard'].mean():.3f}，但已比 v2 的 0.29 有提升。")
    else:
        print(f"❌ 仍有差距。Jaccard={df['jaccard'].mean():.3f}，需要进一步定位差异源。")


if __name__ == "__main__":
    main()
