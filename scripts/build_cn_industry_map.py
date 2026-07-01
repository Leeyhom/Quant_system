#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_cn_industry_map —— 基于申万一级行业分类，自动构建行业映射。

替换 quant/data/industry.py 中的手工18分类为申万一级行业（31类），覆盖全市场。
申万行业分类是中国量化研究的标准分类体系，粒度适中（31类），适合行业中性化。

双路径策略:
  ① 主力: akshare.index_component_sw 获取各行业成分股（25/31可用）
  ② 补充: 申万Excel全量分类文件，通过前缀映射补充API失败的6个行业
          (汽车/机械设备/煤炭/石油石化/环保/美容护理)

用法:
    conda activate quant
    NO_PROXY='*' PYTHONPATH=. python scripts/build_cn_industry_map.py

输出:
    quant/data/industry.py  （覆盖旧的手工分类）
"""
from __future__ import annotations

import sys
import time
import tempfile
import os as _os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_os.environ["NO_PROXY"] = "*"

import requests
import warnings
warnings.filterwarnings("ignore")

import akshare as ak
import pandas as pd


# ───────────────────────── 配置 ─────────────────────────

OUT_PATH = PROJECT_ROOT / "quant" / "data" / "industry.py"

# 申万Excel分类文件（2021版，全量个股→行业映射）
SW_EXCEL_URL = (
    "https://www.swsresearch.com/swindex/pdf/SwClass2021/"
    "StockClassifyUse_stock.xls"
)

# 前缀→一级行业名的补充映射（API不可用的6个行业，由Excel前缀推断）
# 前缀规则: 申万内部行业代码的前2位 = 一级行业标识
FALLBACK_PREFIX_MAP: dict[str, str] = {
    "28": "汽车",
    "64": "机械设备",
    "74": "煤炭",
    "75": "石油石化",
    "76": "环保",
    "77": "美容护理",
}


# ───────────────────────── 步骤1：获取申万一级行业列表 ─────────────────────────

def fetch_sw_industries() -> pd.DataFrame:
    """获取申万一级行业代码和名称列表。"""
    print("=" * 60)
    print("  步骤 1/4：获取申万一级行业列表")
    print("=" * 60)

    df = ak.sw_index_first_info()
    df = df[df["行业代码"].str.match(r"^801\d{3}\.SI$")].copy()
    print(f"  标准一级行业: {len(df)} 个\n")

    for _, row in df.iterrows():
        print(f"    {row['行业代码']}  {row['行业名称']}  ({row['成份个数']}只)")

    return df


# ───────────────────────── 步骤2：API获取各行业成分股 ─────────────────────────

def fetch_via_api(industries: pd.DataFrame) -> dict[str, list[str]]:
    """通过 API 拉取各申万一级行业成分股。

    返回: {industry_name: [code6, ...]}
    """
    print("\n" + "=" * 60)
    print("  步骤 2/4：API 拉取行业成分股")
    print("=" * 60)

    industry_stocks: dict[str, list[str]] = {}
    failed: list[tuple[str, str]] = []

    for i, (_, row) in enumerate(industries.iterrows(), 1):
        name = row["行业名称"]
        symbol = row["行业代码"].replace(".SI", "")

        try:
            df = ak.index_component_sw(symbol)
            if len(df) == 0:
                failed.append((name, symbol))
                print(f"  [{i:2d}/{len(industries)}] {name}: 空结果 → 待Excel补充")
            else:
                stocks = sorted(df["证券代码"].astype(str).str.zfill(6).tolist())
                industry_stocks[name] = stocks
                print(f"  [{i:2d}/{len(industries)}] {name}: {len(stocks)} 只")
        except Exception:
            failed.append((name, symbol))
            print(f"  [{i:2d}/{len(industries)}] {name}: API不可用 → 待Excel补充")
        time.sleep(0.15)

    n_ok = len(industry_stocks)
    n_total = sum(len(v) for v in industry_stocks.values())
    print(f"\n  API成功: {n_ok} 个行业, {n_total} 只股票（含跨行业重复）")
    if failed:
        print(f"  待补充: {len(failed)} 个行业: {[f[0] for f in failed]}")

    return industry_stocks


# ───────────────────────── 步骤3：Excel补充失败的行业 ─────────────────────────

def fetch_via_excel(api_stocks: dict[str, list[str]]) -> dict[str, list[str]]:
    """下载申万Excel全量分类文件，用前缀映射补充API失败的行业。

    同时用Excel数据修正API结果中前缀映射不完全的问题（部分股票在API中
    属于某行业，但Excel中最新的分类已变更）。
    """
    print("\n" + "=" * 60)
    print("  步骤 3/4：Excel 全量分类补充")
    print("=" * 60)

    # ── 下载Excel ──
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    print("    下载申万全量分类文件...", end=" ", flush=True)
    r = requests.get(SW_EXCEL_URL, headers=headers, verify=False, timeout=30)
    if r.status_code != 200 or len(r.content) < 1000:
        print(f"失败 (status={r.status_code}, size={len(r.content)})")
        return {}

    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as f:
        f.write(r.content)
        tmp_path = f.name
    excel_df = pd.read_excel(tmp_path, engine="xlrd")
    _os.unlink(tmp_path)
    print(f"{len(excel_df)} 行, {excel_df['股票代码'].nunique()} 只股票")

    excel_df["code6"] = excel_df["股票代码"].astype(str).str.zfill(6)
    excel_df["ind_str"] = excel_df["行业代码"].astype(str)

    # ── 构建前缀→行业名映射 ──
    # 先用API已有的股票→行业关系反推前缀
    prefix_to_name: dict[str, str] = {}
    for ind_name, stocks in api_stocks.items():
        subset = excel_df[excel_df["code6"].isin(set(stocks))]
        if len(subset) == 0:
            continue
        top_prefix = subset["ind_str"].str[:2].value_counts().index[0]
        prefix_to_name[top_prefix] = ind_name

    # 加入手动补充的6个行业
    prefix_to_name.update(FALLBACK_PREFIX_MAP)
    print(f"    前缀映射: {len(prefix_to_name)} 个（API自动{len(prefix_to_name)-len(FALLBACK_PREFIX_MAP)} + 手动补充{len(FALLBACK_PREFIX_MAP)}）")

    # ── 对每只股票，取其最新行业分类（按更新日期） ──
    latest = excel_df.sort_values("更新日期").groupby("code6").last()
    latest["prefix"] = latest["ind_str"].str[:2]
    latest["sw_name"] = latest["prefix"].map(prefix_to_name)

    # ── 构建完整的行业→代码映射 ──
    excel_stocks: dict[str, list[str]] = {}
    for ind_name in set(prefix_to_name.values()):
        codes = sorted(latest[latest["sw_name"] == ind_name].index.tolist())
        if codes:
            excel_stocks[ind_name] = codes

    # 统计未分类的股票
    unclassified = latest[latest["sw_name"].isna()]
    print(f"    分类成功: {len(latest) - len(unclassified)} 只")
    print(f"    未分类: {len(unclassified)} 只")

    return excel_stocks


# ───────────────────────── 步骤4：合并并输出 ─────────────────────────

def merge_and_generate(
    api_stocks: dict[str, list[str]],
    excel_stocks: dict[str, list[str]],
) -> dict[str, list[str]]:
    """合并API和Excel的行业分类，优先使用API（权重信息更准确）。"""
    print("\n" + "=" * 60)
    print("  步骤 4/4：合并并生成 industry.py")
    print("=" * 60)

    # API优先，Excel补充API缺失的行业
    merged: dict[str, list[str]] = {}
    for name, stocks in excel_stocks.items():
        merged[name] = stocks
    for name, stocks in api_stocks.items():
        merged[name] = stocks  # API覆盖Excel（权重更准）

    print(f"\n  行业数: {len(merged)}")
    for name in sorted(merged.keys()):
        print(f"    {name}: {len(merged[name])} 只")

    # ── 构建 code→industry 的唯一映射 ──
    code_to_ind: dict[str, str] = {}
    for ind_name, stocks in merged.items():
        for s in stocks:
            if s not in code_to_ind:
                code_to_ind[s] = ind_name

    total_unique = len(code_to_ind)
    print(f"\n  去重后唯一股票: {total_unique} 只")

    # ── 检查 CN800 覆盖 ──
    uncovered_codes: set[str] = set()
    try:
        from quant.data.universe_cn800 import CN800_POOL
        cn800 = set(CN800_POOL)
        covered = cn800 & set(code_to_ind.keys())
        uncovered_codes = cn800 - set(code_to_ind.keys())
        print(f"  CN800 池覆盖: {len(covered)}/{len(cn800)} ({len(covered)/len(cn800)*100:.1f}%)")
        if uncovered_codes:
            print(f"  未覆盖: {len(uncovered_codes)} 只 → 归入「其他」")
            print(f"    示例: {sorted(uncovered_codes)[:15]}")
    except ImportError:
        print("  （CN800 池未构建，跳过覆盖统计）")

    # ── 构建行业→代码列表 ──
    ind_to_codes: dict[str, list[str]] = {}
    for code, ind in code_to_ind.items():
        ind_to_codes.setdefault(ind, []).append(code)
    for codes in ind_to_codes.values():
        codes.sort()

    # ── 写入文件 ──
    industry_entries = ",\n".join(
        f'    "{name}": {sorted(codes)!r}'
        for name, codes in sorted(ind_to_codes.items())
    )

    uncovered_repr = sorted(uncovered_codes)[:20] if uncovered_codes else []

    content = f'''# -*- coding: utf-8 -*-
"""industry —— A股股票行业分类映射（申万一级行业，{len(ind_to_codes)}类）。

由 scripts/build_cn_industry_map.py 自动生成。
生成日期: {pd.Timestamp.now().strftime("%Y-%m-%d")}
覆盖股票: {total_unique} 只

来源:
  - 申万宏源研究所-指数发布-申万一级行业指数
  - 主力: akshare.index_component_sw 成分股API
  - 补充: 申万2021版Excel全量分类（前缀映射补充API不可用的行业）

行业列表（{len(ind_to_codes)} 个）:
{chr(10).join(f'  {name}: {len(codes)} 只' for name, codes in sorted(ind_to_codes.items()))}

与旧版差异:
  - 旧版: 手工维护18个行业，仅覆盖 ~90 只 DEFAULT_POOL 股票
  - 新版: 申万一级{len(ind_to_codes)}行业，覆盖 {total_unique} 只全市场股票
  - 池外股票归入「其他」（与旧版一致）

CN800覆盖: {len(ind_to_codes)} 行业, 未覆盖示例: {uncovered_repr}

用法:
    from quant.data.industry import industry_series, symbol_to_industry
    s2i = symbol_to_industry()  # {{code6: industry_name}}
    series = industry_series(symbols)  # pd.Series, 池外 → "其他"
"""

from __future__ import annotations

import pandas as pd


# 行业 → 代码列表（申万一级行业，{len(ind_to_codes)}类）
INDUSTRY_MAP: dict[str, list[str]] = {{
{industry_entries}
}}

# 池外股票的兜底分类
_UNKNOWN = "其他"


def symbol_to_industry() -> dict[str, str]:
    """反转 INDUSTRY_MAP：代码 → 行业名。

    跨行业股票只属于首次出现的行业（API优先，权重更高），
    与申万指数主要行业分类逻辑一致。
    """
    out: dict[str, str] = {{}}
    for ind, syms in INDUSTRY_MAP.items():
        for s in syms:
            if s not in out:
                out[s] = ind
    return out


def industry_series(symbols: list[str]) -> pd.Series:
    """给定股票列表，返回 index=代码、value=行业名 的 Series（供 groupby）。

    池外股票归入「其他」，保证每只股票都有行业、不会在中性化时被丢掉。
    """
    s2i = symbol_to_industry()
    return pd.Series({{sym: s2i.get(sym, _UNKNOWN) for sym in symbols}})
'''
    OUT_PATH.write_text(content, encoding="utf-8")
    print(f"\n  ✅ 已写入: {OUT_PATH}")

    return code_to_ind


# ───────────────────────── 主流程 ─────────────────────────

def main() -> None:
    t0 = time.time()

    # 1: 行业列表
    industries = fetch_sw_industries()

    # 2: API拉取
    api_stocks = fetch_via_api(industries)

    # 3: Excel补充
    excel_stocks = fetch_via_excel(api_stocks)

    # 4: 合并生成
    code_to_ind = merge_and_generate(api_stocks, excel_stocks)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成！耗时 {elapsed:.0f} 秒")
    print(f"  行业映射覆盖: {len(code_to_ind)} 只股票")
    print(f"{'='*60}")

    # ── 快速验证 ──
    print("\n  验证常用股票分类:")
    from quant.data.industry import symbol_to_industry
    s2i = symbol_to_industry()
    for c in ["000001", "600519", "300750", "000858", "601318", "688981",
              "600104", "601899", "600028", "002594"]:
        print(f"    {c}: {s2i.get(c, '其他')}")


if __name__ == "__main__":
    main()
