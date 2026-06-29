"""cn_factor_spec —— A股策略「单一事实源」(SSOT)：因子定义 + 参数 + 池/行业/费用。

M20 引入。目的：本地回测、RQAlpha 第三方复现、聚宽模拟盘三方此前各写一套因子和
参数，口径漂移导致选股大幅分叉（实测同日目标股平均只重合 5.16/12，Jaccard 0.29）。
本模块把 v3 策略的因子构造和全部参数固化成一份，三方共同引用，消除口径漂移。

⚠️ 本模块刻意**只依赖 quant 包内已有的因子函数和数据面板**，不直接联网。聚宽云端
无法 import 本包，故聚宽策略文件仍是自包含副本，但必须**逐字对照本模块**保持口径
一致（成长因子口径、参数、行业映射、费用）——这是「对齐」的纪律，不是自动同步。

—— 关键对齐决策（M20，逐一对应诊断根因）——

1. 成长因子口径（最大乖离源，M20→6.5 两次裁决）：
   - M20 一度从「1/PEG（东财日频）」改为「季报净利润同比/PE」以对齐聚宽实现并可复现。
   - 阶段6.5 诚实 walk-forward（5/5 单轴消融窗口一致）证伪了「同比更好」：日频 PEG
     倒数夏普 0.79/回撤 29% 全面优于季报同比 0.62/44%。**根因是数据覆盖率**——季报表
     stock_yjbb_em 仅 68% 覆盖，日频估值 94%，缺口使等权合成 growth 维度大量缺失。
   - **关键**：二者概念同一（日频估值无独立增速字段，1/PEG = 增速/PE）。故生产口径
     回到日频 PEG 倒数（`factors.growth_peg`，覆盖高 + OOS 优 + 概念仍是增速/PE 可对齐聚宽）。
     `growth_mode="peg"` 为默认；`"yoy"` 仅留作对比。详见 build_factor_library 文档。
2. 参数固定为 v3（项目实测分散结构最稳）：top10 / 40 日再平衡 / 单行业 1 只 / 95% 仓位。
3. 行业映射唯一来源 = `quant.data.industry.INDUSTRY_MAP`，聚宽策略手抄须同步。
4. 费用：佣金万 2.5 + 过户万 0.1（并入双边佣金）、印花万 5 卖出单边、滑点 0.0005。

—— 已知数据缺口（诚实记录）——
   池中 600837(海通证券)、601989(中国重工)、002013(中航机电) 三票在东财行情/估值/
   业绩报表接口均返回空——它们已被合并/改名/退市（如 601989 于 2024 并入中国船舶）。
   本地有效池实为 89/92。聚宽侧这些票合并后同样不可交易，故除外口径一致，不影响对齐。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.factor import factors as F
from quant.factor.neutralize import neutralize
from quant.factor.factors import combine_factors


# ============================ 参数（v3 基准） ============================

TOP_N = 10
REBALANCE_DAYS = 40
INDUSTRY_CAP = 1
MAX_EXPOSURE = 0.95
INITIAL_CAPITAL = 60_000

# 费用（与聚宽 set_order_cost 对齐）
COMMISSION_RATE = 0.00025      # 佣金 万2.5
TRANSFER_FEE_RATE = 0.00001    # 过户费 万0.1（聚宽无独立字段，并入双边佣金）
STAMP_DUTY_RATE = 0.0005       # 印花税 万5，卖出单边
SLIPPAGE = 0.0005              # PriceRelatedSlippage(0.0005)
EFFECTIVE_COMMISSION_RATE = COMMISSION_RATE + TRANSFER_FEE_RATE
MIN_COMMISSION = 5.0           # 单笔最低佣金 5 元

# 已知数据缺口（合并/退市，三方均不可交易）
KNOWN_MISSING = ("600837", "601989", "002013")

# v3 五因子名（合成顺序固定，三方一致）
V3_FACTORS = ("earnings_yield", "cashflow_yield", "sales_yield", "growth", "amihud")

# v4 六因子名（M21：在 v3 基础上加筹码集中度，实测正交且压回撤）。
# 验证结论（scripts/cn_holder_factor_eval.py，诚实 walk-forward）：
#   与 v3 五因子相关全部 |corr|<0.08（极正交）；加入后 OOS 夏普 0.62→0.71、
#   回撤 43.8%→39.9%、超额 +0.3%→+24.8%，三项全改善。
V4_FACTORS = V3_FACTORS + ("holder_concentration",)


# ============================ 因子库（本地口径，SSOT） ============================

def build_factor_library(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    value: dict[str, pd.DataFrame],
    quarterly: dict[str, pd.DataFrame],
    industry: pd.Series,
    holder: dict[str, pd.DataFrame] | None = None,
    neutralize_mode: str = "full",
    growth_mode: str = "peg",
) -> dict[str, pd.DataFrame]:
    """构建 v3/v4 因子库（已做行业/市值中性化），键为因子名，值为「日期 × 股票」面板。

    这是本地与 RQAlpha 共享的唯一因子构造入口。聚宽自包含副本须逐字对照本函数。

    参数:
        close:  收盘价面板（前复权）。
        amount: 成交额面板（amihud 用）。
        value:  日频估值面板字典（build_value_panels），需含 pe_ttm/pcf/ps/peg。
        quarterly: 季报增速面板字典（build_cn_quarterly_panels），需含 net_profit_yoy。
        industry: 代码 -> 行业名 的 Series（industry_series），中性化分组用。
        holder: 股东户数面板字典（build_cn_holder_panels），需含 change_ratio。
            传入则额外构造 holder_concentration 因子（v4）；None 则只出 v3 五因子（向后兼容）。
        neutralize_mode: 中性化模式，默认 "full"（行业+市值双中性，见 neutralize.py）。
        growth_mode: 成长因子口径（M20 阶段6.5 裁决，默认改回 "peg"）。
            "peg"=东财日频 PEG 倒数（= 增速/PE，覆盖 94%，生产默认）；
            "yoy"=季报净利润同比/PE（覆盖仅 68%，仅留作对比/历史口径）。
            ⚠️ 二者概念同一（日频估值无独立增速字段，成长信息编码在 PEG 里：1/PEG=增速/PE）。
            诚实 walk-forward 5/5 窗口一致显示 peg 优于 yoy（夏普 0.79 vs 0.62、回撤 29% vs 44%）；
            根因是季报表 stock_yjbb_em 覆盖太稀疏（68%）使等权合成 growth 维度大量缺失。
            详见 [[m20-local-jq-alignment]] 阶段6.5、scripts/cn_walkforward_honest.py --growth。
    """
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    if growth_mode == "yoy":
        # 季报同比/PE（覆盖稀疏，OOS 较差，仅留作对比；见上方 growth_mode 说明）
        growth = F.growth_yoy_over_pe(quarterly["net_profit_yoy"], value["pe_ttm"])
    else:
        # 生产默认：东财日频 PEG 倒数 = 增速/PE，覆盖 94%，OOS 更优且概念可对齐聚宽
        growth = F.growth_peg(value["peg"])
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "sales_yield": F.sales_yield(value["ps"]),
        "growth": growth,
        "amihud": F.amihud_illiquidity(close, amount, 20),
    }
    # M21：传入股东户数则加筹码集中度因子（v4），实测正交且压回撤。
    if holder is not None:
        raw["holder_concentration"] = F.holder_concentration(holder["change_ratio"])
    return {
        name: neutralize(factor, industry=industry, log_mv=log_mv, mode=neutralize_mode)
        for name, factor in raw.items()
    }


def equal_weight_composite(lib: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """等权合成：库内全部因子各做横截面 rank 后取均值（M14：先验稳时等权胜 IC 加权）。

    方向先验：因子都已是「越高越好」量纲，等权合成不预测方向（绕开 regime 翻号陷阱）。
    自动适配 v3（5 因子）或 v4（6 因子，含筹码）——按 lib 实际包含的因子合成。
    """
    return combine_factors(*lib.values())
