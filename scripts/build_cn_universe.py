#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_cn_universe —— 构建A股扩展股票池：沪深300+中证500成分股。

从手工维护的 ~90~152 只静态池，升级到规则驱动的 ~700-750 只动态池。
池子构成规则：
  ① 沪深300 + 中证500 最新成分股（互斥，约800只）
  ② 剔除 ST/*ST/PT
  ③ 剔除 B 股（9xxxxx）
  ④ 代码转换为本地无后缀格式 + JoinQuant .XSHG/.XSHE 格式
  ⑤ 输出到 quant/data/universe_cn800.py，供本地回测和聚宽策略引用

用法：
    conda activate quant
    NO_PROXY='*' PYTHONPATH=. python scripts/build_cn_universe.py

输出：
    quant/data/universe_cn800.py          # 股票池 Python 模块（本地无后缀 + JQ 格式）
    quant/data/universe_cn800_stats.csv   # 行业/市值覆盖统计

后续步骤：
    NO_PROXY='*' PYTHONPATH=. python scripts/refetch_history.py --pool cn800
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# akshare 联网需要绕过本地代理（必须在 import akshare 之前设置）
import os
os.environ["NO_PROXY"] = "*"

import akshare as ak

from quant.data.industry import symbol_to_industry as _get_industry_map


# ───────────────────────── 配置 ─────────────────────────

CSI300_CODE = "000300"
CSI500_CODE = "000905"

# 代码前缀 → JoinQuant 后缀的映射
def _to_jq_format(code6: str) -> str:
    """6 位代码 → JoinQuant 格式（.XSHG / .XSHE）"""
    if code6.startswith("6"):   # 60xxxx, 688xxx, 689xxx
        return f"{code6}.XSHG"
    elif code6.startswith(("0", "3")):  # 00xxxx, 30xxxx
        return f"{code6}.XSHE"
    else:
        # 兜底：不应该出现
        return f"{code6}.XSHG"


def _from_jq_format(jq_code: str) -> str:
    """JoinQuant 格式 → 6 位代码"""
    return jq_code.split(".")[0]


OUT_DIR = PROJECT_ROOT / "quant" / "data"
OUT_PY = OUT_DIR / "universe_cn800.py"
OUT_CSV = OUT_DIR / "universe_cn800_stats.csv"


# ───────────────────────── 步骤1：获取成分股 ─────────────────────────

def fetch_constituents() -> pd.DataFrame:
    """获取沪深300 + 中证500 最新成分股，去重后返回 DataFrame。"""
    print("=" * 60)
    print("  步骤 1/4：获取指数成分股")
    print("=" * 60)

    all_rows = []
    for idx_code, idx_name in [(CSI300_CODE, "沪深300"), (CSI500_CODE, "中证500")]:
        print(f"    拉取 {idx_name} ({idx_code}) ...", end=" ", flush=True)
        try:
            df = ak.index_stock_cons_csindex(idx_code)
            df["来源指数"] = idx_name
            all_rows.append(df)
            print(f"{len(df)} 只")
        except Exception as e:
            print(f"失败: {e}")
            raise

    combined = pd.concat(all_rows, ignore_index=True)
    # 去重（沪深300和中证500互斥，但安全起见）
    combined = combined.drop_duplicates(subset=["成分券代码"])
    print(f"    合并去重后: {len(combined)} 只\n")
    return combined


# ───────────────────────── 步骤2：获取当前市场状态 ─────────────────────────

def fetch_market_status() -> pd.DataFrame:
    """获取全市场实时快照，用于 ST 检测和基础信息。"""
    print("=" * 60)
    print("  步骤 2/4：获取全市场状态（ST检测）")
    print("=" * 60)
    print("    拉取 stock_zh_a_spot_em（~5800只，约70秒）...", flush=True)

    spot = ak.stock_zh_a_spot_em()
    print(f"    获取 {len(spot)} 只股票快照")

    # 标记 ST
    spot["is_st"] = spot["名称"].str.contains(r"ST|\*ST|PT", na=False)
    st_count = spot["is_st"].sum()
    print(f"    其中 ST/*ST/PT: {st_count} 只")

    # 标记 B 股（9xxxxx 开头）
    spot["is_b_share"] = spot["代码"].str.startswith("9")
    b_count = spot["is_b_share"].sum()
    print(f"    其中 B 股(9开头): {b_count} 只")

    return spot[["代码", "名称", "is_st", "is_b_share", "总市值", "市盈率-动态", "市净率"]]


# ───────────────────────── 步骤3：筛选 ─────────────────────────

def filter_universe(
    constituents: pd.DataFrame, spot: pd.DataFrame
) -> pd.DataFrame:
    """将成分股与市场状态对齐，应用过滤规则。"""
    print("\n" + "=" * 60)
    print("  步骤 3/4：应用过滤规则")
    print("=" * 60)

    # 对齐：成分券代码（6位无后缀） ↔ spot 代码
    codes = constituents[["成分券代码", "成分券名称", "交易所", "来源指数"]].copy()
    codes.columns = ["code6", "name", "exchange", "source_index"]

    # 合并市场状态
    merged = codes.merge(spot, left_on="code6", right_on="代码", how="left")

    # 统计匹配情况
    matched = merged["名称"].notna().sum()
    unmatched = merged["名称"].isna().sum()
    print(f"    成分股与市场快照匹配: {matched}/{len(merged)}")
    if unmatched > 0:
        print(f"    未匹配（可能已退市/更名）: {unmatched} 只")
        unmatched_codes = merged[merged["名称"].isna()]["code6"].tolist()
        print(f"      {unmatched_codes[:10]}")

    # ── 过滤1: ST/*ST/PT ──
    st_mask = merged["is_st"].fillna(False)
    print(f"    剔除 ST/*ST/PT: {st_mask.sum()} 只")

    # ── 过滤2: B 股 ──
    b_mask = merged["is_b_share"].fillna(False)
    print(f"    剔除 B 股: {b_mask.sum()} 只")

    # ── 过滤3: 未匹配（退市/更名） ──
    na_mask = merged["名称"].isna()
    print(f"    剔除未匹配（退市/更名）: {na_mask.sum()} 只")

    # ── 综合过滤 ──
    keep = ~st_mask & ~b_mask & ~na_mask
    filtered = merged[keep].copy()
    print(f"\n    过滤后: {len(filtered)} 只（从 {len(merged)} 剔除 {len(merged) - len(filtered)}）")

    # ── 统计 ──
    print(f"\n    来源分布:")
    for src in ["沪深300", "中证500"]:
        n = (filtered["source_index"] == src).sum()
        print(f"      {src}: {n} 只")

    print(f"    交易所分布:")
    for ex, n in filtered["exchange"].value_counts().items():
        print(f"      {ex}: {n} 只")

    # 市值统计
    mv = pd.to_numeric(filtered["总市值"], errors="coerce")
    print(f"\n    总市值范围: {mv.min()/1e8:.0f}亿 ~ {mv.max()/1e8:.0f}亿")
    print(f"    总市值中位: {mv.median()/1e8:.0f}亿")

    return filtered


# ───────────────────────── 步骤4：生成输出 ─────────────────────────

def generate_output(filtered: pd.DataFrame) -> tuple[list[str], list[str]]:
    """生成两种格式的股票列表并写入文件。"""
    print("\n" + "=" * 60)
    print("  步骤 4/4：生成输出文件")
    print("=" * 60)

    codes_plain = sorted(filtered["code6"].tolist())
    codes_jq = [_to_jq_format(c) for c in codes_plain]

    # ── 行业分布统计（复用现有手工行业映射） ──
    code_to_ind = _get_industry_map()  # {code6: industry_name}，池外返回"其他"
    industry_counts: dict[str, int] = {}
    industry_stocks: dict[str, list[str]] = {}
    for c in codes_plain:
        ind = code_to_ind.get(c, "其他")
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
        industry_stocks.setdefault(ind, []).append(c)

    print(f"\n  行业分布（现有 INDUSTRY_MAP 覆盖）:")
    for ind, n in sorted(industry_counts.items(), key=lambda x: -x[1]):
        pct = n / len(codes_plain) * 100
        stocks_str = ",".join(industry_stocks[ind][:6])
        if len(industry_stocks[ind]) > 6:
            stocks_str += f"...({len(industry_stocks[ind])}只)"
        print(f"    {ind}: {n} 只 ({pct:.1f}%)  [{stocks_str}]")

    uncovered = industry_counts.get("其他", 0)
    covered = len(codes_plain) - uncovered
    print(f"\n  行业映射覆盖率: {covered}/{len(codes_plain)} ({covered/len(codes_plain)*100:.1f}%)")
    if uncovered > 0:
        print(f"  ⚠️ {uncovered} 只落入「其他」，需要扩展 INDUSTRY_MAP")

    # ── 写入 Python 模块 ──
    py_content = f'''# -*- coding: utf-8 -*-
"""A股扩展股票池：沪深300 + 中证500 成分股。

由 scripts/build_cn_universe.py 自动生成。
生成日期: {pd.Timestamp.now().strftime("%Y-%m-%d")}
股票数量: {len(codes_plain)} 只

构成规则:
  - 来源: 中证指数官网最新沪深300(000300) + 中证500(000905)成分股
  - 过滤: 剔除ST/*ST/PT、B股、退市/更名
  - 行业: 手工映射覆盖 {covered}/{len(codes_plain)} 只（{covered/len(codes_plain)*100:.1f}%）
  - 更新: 每半年指数调仓后重新运行本脚本即可

用法:
    from quant.data.universe_cn800 import CN800_POOL, CN800_POOL_JQ

    # 本地回测（无后缀6位代码）
    pool = CN800_POOL

    # 聚宽策略（.XSHG/.XSHE 后缀）
    jq_pool = CN800_POOL_JQ
"""

# 本地无后缀池
CN800_POOL = {codes_plain!r}

# 聚宽格式池（.XSHG/.XSHE 后缀）
CN800_POOL_JQ = {codes_jq!r}

# 按来源指数分组
CN800_HS300 = {sorted([_to_jq_format(c) for c in filtered[filtered["source_index"]=="沪深300"]["code6"].tolist()])!r}
CN800_ZZ500 = {sorted([_to_jq_format(c) for c in filtered[filtered["source_index"]=="中证500"]["code6"].tolist()])!r}

# 行业分布（参考）
CN800_INDUSTRY_COUNTS = {dict(sorted(industry_counts.items(), key=lambda x: -x[1]))!r}
'''
    OUT_PY.write_text(py_content, encoding="utf-8")
    print(f"\n  ✅ 已写入: {OUT_PY}")

    # ── 写入 CSV 统计 ──
    stats_rows = []
    for _, row in filtered.iterrows():
        stats_rows.append({
            "code6": row["code6"],
            "name": row["名称"] if pd.notna(row.get("名称")) else row.get("name", ""),
            "jq_code": _to_jq_format(row["code6"]),
            "exchange": row["exchange"],
            "source_index": row["source_index"],
            "total_mv_yuan": row.get("总市值", np.nan),
            "pe": row.get("市盈率-动态", np.nan),
            "pb": row.get("市净率", np.nan),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✅ 已写入: {OUT_CSV}")

    return codes_plain, codes_jq


# ───────────────────────── 主流程 ─────────────────────────

def main() -> None:
    t0 = time.time()

    # 1: 获取成分股
    constituents = fetch_constituents()

    # 2: 获取市场状态
    spot = fetch_market_status()

    # 3: 过滤
    filtered = filter_universe(constituents, spot)

    # 4: 输出
    codes_plain, codes_jq = generate_output(filtered)

    # ── 后续操作提示 ──
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成！耗时 {elapsed:.0f} 秒")
    print(f"  最终池: {len(codes_plain)} 只股票")
    print(f"{'='*60}")

    # 对比原有池
    try:
        from quant.data.universe import DEFAULT_POOL
        old_set = set(DEFAULT_POOL)
        new_set = set(codes_plain)
        retained = old_set & new_set
        added = new_set - old_set
        removed = old_set - new_set
        print(f"\n  与旧池(DEFAULT_POOL={len(DEFAULT_POOL)}只)对比:")
        print(f"    保留: {len(retained)} 只")
        print(f"    新增: {len(added)} 只")
        print(f"    移除: {len(removed)} 只")
        if removed:
            print(f"    移除列表: {sorted(removed)}")
    except Exception:
        pass

    print(f"\n  ═══ 后续步骤 ═══")
    print(f"  1. 拉取数据到本地缓存:")
    print(f"     NO_PROXY='*' PYTHONPATH=. python scripts/refetch_history.py --pool cn800")
    print(f"     （或修改 refetch_history.py 引用 CN800_POOL）")
    print(f"  2. 本地 walk-forward 验证:")
    print(f"     PYTHONPATH=. python scripts/cn_walkforward_honest.py --pool cn800")
    print(f"  3. 更新聚宽策略文件（替换 STOCK_POOL）:")
    print(f"     from quant.data.universe_cn800 import CN800_POOL_JQ")
    print(f"  4. 聚宽回测验证（2019-2025 全周期）")
    print(f"  5. 与 v9 基线对比，确认改善幅度")
    print()


if __name__ == "__main__":
    main()
