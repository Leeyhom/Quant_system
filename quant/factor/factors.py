"""factors —— 横截面因子库。

输入统一为面板：index=日期，columns=股票，值=对应字段（收盘价/成交量/...）。
输出与输入同形状：每个日期、每只股票一个因子分数，约定**越高越好**。
组合/分层层会在每个再平衡日对当天的因子分数横向排序。

⚠️ 防未来函数：所有因子只用 rolling/pct_change 等向后看的历史数据，
且**不在因子内部 shift**。回测/IC 层会用「上一交易日因子」对齐未来收益，
若在因子里再 shift 会造成双重滞后，悄悄削弱 IC。

方向约定（见 docs/09）：低波动、低流动性冲击这类「越小越好」的指标统一取负，
使所有因子都满足「分数越高越值得买」。某因子方向是否真有效，由 IC 的正负揭示。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """动量因子：过去 window 日累计收益，越高越好。"""
    return close.pct_change(window)


def reversal(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """反转因子：过去 window 日收益取负，过去跌得多的分数更高。"""
    return -close.pct_change(window)


def low_volatility(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """低波动因子：过去 window 日日收益波动取负，波动越低分数越高。"""
    ret = close.pct_change()
    return -ret.rolling(window).std()


def combine_factors(*factors: pd.DataFrame) -> pd.DataFrame:
    """简单多因子合成：每个因子先做横截面 rank，再取平均。

    为什么 rank：不同因子量纲不同（收益率、波动率），直接相加不合理。
    rank(pct=True) 把每个因子统一到 0~1 分位数，越高越好。
    """
    if not factors:
        raise ValueError("至少传入一个因子")
    ranked = [f.rank(axis=1, pct=True) for f in factors]
    return sum(ranked) / len(ranked)


# ───────────────────────── 量能因子 ─────────────────────────

def volume_trend(volume: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """量能趋势：短期均量 / 长期均量。>1 表示近期放量。"""
    short_ma = volume.rolling(short).mean()
    long_ma = volume.rolling(long).mean().replace(0, np.nan)
    return short_ma / long_ma


def amount_liquidity(amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """流动性（成交额）：过去 window 日平均成交额，越大越易进出。"""
    return amount.rolling(window).mean()


# ───────────────────────── 波动因子 ─────────────────────────

def parkinson_volatility(high: pd.DataFrame, low: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Parkinson 高低价波动率（取负，低波得分高）。

    用日内 ln(high/low) 估计波动，比仅用收盘价更充分利用了日内振幅信息。
    """
    log_hl = np.log(high / low.replace(0, np.nan))
    vol = (log_hl ** 2).rolling(window).mean() ** 0.5
    return -vol


def atr_volatility(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """ATR 真实波幅（取负，低波得分高）。

    真实波幅 = max(high-low, |high-prev_close|, |low-prev_close|)，
    比单纯 high-low 更稳健（考虑了隔夜跳空）。这里用相对 close 的比例，去量纲。
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ]).groupby(level=0).max()
    atr = tr.rolling(window).mean() / close.replace(0, np.nan)
    return -atr


# ───────────────────────── 趋势因子 ─────────────────────────

def ma_slope(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """均线斜率：均线相对 window 日前的变化率，衡量趋势方向与强度。"""
    ma = close.rolling(window).mean()
    return ma / ma.shift(window).replace(0, np.nan) - 1.0


def price_to_ma(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """价格相对均线偏离：close / SMA - 1。>0 表示在均线上方。"""
    ma = close.rolling(window).mean().replace(0, np.nan)
    return close / ma - 1.0


# ─────────────────────── 流动性冲击因子 ───────────────────────

def amihud_illiquidity(close: pd.DataFrame, amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Amihud 非流动性（取负，流动性好得分高）。

    Amihud = 平均(|日收益| / 成交额)，衡量单位成交额造成的价格冲击。
    值越大越不流动。取负后「越高越好」=「越流动越好」。

    局限：缺流通股本数据，无法算真正的换手率，这里用成交额近似流动性，
    数值受股票绝对成交额规模影响（大盘股天然成交额大）。仅作研究参考。
    """
    ret = close.pct_change().abs()
    illiq = (ret / amount.replace(0, np.nan)).rolling(window).mean()
    return -illiq


# ───────────────────────── 基本面因子（M10） ─────────────────────────
# 输入为估值/股本面板（来自 build_value_panels），与价格面板严格对齐。
# 沿用统一约定：分数越高越值得买；方向先验仅是经济假设，真伪由 IC 验证。

def earnings_yield(pe_ttm: pd.DataFrame) -> pd.DataFrame:
    """盈利收益率 = 1 / PE(TTM)，经典价值因子：越便宜（低PE）分越高。

    用倒数而非 -PE：PE 的尺度高度偏态（几倍到几百倍），倒数把它压到
    [0, 较小] 的稳健区间，横截面排序更合理。负 PE 已在 loader 置 NaN。
    """
    return 1.0 / pe_ttm.replace(0, np.nan)


def book_to_price(pb: pd.DataFrame) -> pd.DataFrame:
    """账面市值比 = 1 / 市净率（B/P）：低 PB（破净/便宜）分高。"""
    return 1.0 / pb.replace(0, np.nan)


def sales_yield(ps: pd.DataFrame) -> pd.DataFrame:
    """营收市值比 = 1 / 市销率（S/P）：低 PS 分高。对盈利波动大的行业更稳健。"""
    return 1.0 / ps.replace(0, np.nan)


def small_size(total_mv: pd.DataFrame) -> pd.DataFrame:
    """小市值因子 = -ln(总市值)：市值越小分越高（A股经典小市值效应）。

    取对数：市值跨数量级，对数后分布更对称、排序更稳。方向可能被 IC 推翻
    （近年小市值效应时强时弱），故仅作先验。
    """
    return -np.log(total_mv.replace(0, np.nan))


def turnover_rate(
    volume: pd.DataFrame, float_share: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """真实换手率 = 成交量 / 流通股本 的 window 日均值。

    M8 时缺流通股本只能用成交额近似，现在 stock_value_em 提供了流通股本，
    可算真正的换手率。**方向不预设**：高换手既可能是关注度（正向）也可能是
    投机过热（反向），交给 IC 判断；demo 的双方向候选机制会同时试正/反。
    注意 volume 单位为股、float_share 单位为股，比值即换手率。
    """
    return (volume / float_share.replace(0, np.nan)).rolling(window).mean()


# ─────────────────── 质量/成长/现金流因子（M14，零新增数据源） ───────────────────
# 关键洞察：现有估值面板里已隐含「价值之外」的正交维度，无需新接口即可导出：
#   - 质量(盈利能力) ROE = PB / PE  —— 会计恒等式：
#       PB = 价格/每股净资产, PE = 价格/每股收益
#       PB / PE = 每股收益 / 每股净资产 = 净资产收益率 ROE。
#     实测全池 ROE 中位 ≈10%（p10 4%, p90 20%），数值合理。ROE 越高=越能用
#     净资产赚钱=质量越好 → 越高越好，无需取负。
#   - 成长 1/PEG —— PEG = PE / 盈利增速，低 PEG = 相对成长便宜。取倒数使「越高越好」。
#   - 现金流收益率 1/PCF —— 市现率倒数，现金流相对市值越高越好（盈利质量的现金验证）。
# 为什么是「正交」：质量/成长回答「这公司好不好/快不快」，价值回答「便不便宜」，
# 经济含义不同。实测双中性后与 earnings_yield 相关仅 0.08~0.41，确为新信息。
# 方向仍是先验，真伪由 OOS IC 验证（M14 发现它们在不同 regime 时效性不同）。

def quality_roe(pe_ttm: pd.DataFrame, pb: pd.DataFrame) -> pd.DataFrame:
    """质量因子 ROE = PB / PE（盈利能力，越高越好）。

    用恒等式从已有的 PB、PE(TTM) 面板直接导出净资产收益率，零新增数据源。
    PE 已在 loader 把负值置 NaN（亏损股不参与），PB 同理，故比值天然只在
    「正盈利且正净资产」的票上有定义，避免亏损股污染排序。
    """
    return pb.replace(0, np.nan) / pe_ttm.replace(0, np.nan)


def growth_peg(peg: pd.DataFrame) -> pd.DataFrame:
    """成长因子 = 1 / PEG：低 PEG（相对盈利增速便宜）分高。

    PEG 把估值(PE)和成长(盈利增速)合在一起：同样 PE，增速越快 PEG 越低越划算。
    取倒数统一为「越高越好」。负/零 PEG 已在 loader 置 NaN（增速为负无意义）。
    """
    return 1.0 / peg.replace(0, np.nan)


def cashflow_yield(pcf: pd.DataFrame) -> pd.DataFrame:
    """现金流收益率 = 1 / 市现率（PCF）：经营现金流相对市值越高分越高。

    现金流比账面利润更难粉饰，是盈利质量的「现金验证」。与价值(1/PE)的差别：
    1/PE 看账面利润便宜度，1/PCF 看现金流便宜度，对应计/造假敏感度不同。
    """
    return 1.0 / pcf.replace(0, np.nan)


def growth_yoy_over_pe(net_profit_yoy: pd.DataFrame, pe_ttm: pd.DataFrame) -> pd.DataFrame:
    """成长因子（对齐聚宽口径）= 净利润同比增速 / PE，越高越好。

    M20 对齐聚宽的核心修复。背景：本地此前用东财日频 PEG 值取倒数（`growth_peg`）
    作成长因子，但聚宽用的是 `indicator.净利润同比增速 / valuation.pe`——两者口径
    完全不同（东财 PEG 的增速来源不透明），导致本地与聚宽选股大幅分叉。本函数改用
    与聚宽一致的「季报净利润同比增速 ÷ PE(TTM)」，让两套策略口径统一。

    经济含义同 PEG 倒数：同样 PE，增速越快越划算（成长相对估值便宜）。
    - net_profit_yoy 来自季报、已按公告日防前视对齐（build_cn_quarterly_panels），
      单位为「%」（如 38.93 表示 +38.93%），横截面排序只看相对高低，单位不影响。
    - 只在增速为正、PE 为正时有定义：负增速（盈利下滑）/负 PE（亏损）置 NaN，
      避免「负÷负=正」的假信号，与旧 growth_peg「PEG 非正置 NaN」的处理一致。
    """
    g = net_profit_yoy.where(net_profit_yoy > 0)
    pe = pe_ttm.replace(0, np.nan).where(pe_ttm > 0)
    return g / pe


def holder_concentration(change_ratio: pd.DataFrame) -> pd.DataFrame:
    """筹码集中度因子（M21）= 股东户数环比变化取负，越高越好。

    经济假设（与价值/质量/成长正交的新维度）：
      股东户数减少（change_ratio<0）= 筹码向少数人集中 = 主力吸筹 → 看多；
      股东户数增加（change_ratio>0）= 筹码分散 = 散户接盘 → 价值陷阱预警。
    取负使「户数降得越多分越高」，符合「越高越好」约定。

    change_ratio 来自季报、已按公告日防前视对齐（build_cn_holder_panels），
    单位为「%」（如 -4.98 表示户数环比 -4.98%）。方向是先验，真伪由 IC 验证：
    A股「主力吸筹」叙事广为流传，是否真有 OOS alpha、能否压价值陷阱回撤，看数据。
    """
    return -change_ratio


# ─────────────────── 美股季报基本面因子（M17，Stage 2b） ───────────────────
# 输入为美股基本面面板（来自 build_us_fundamental_panels，已按公告日防前视对齐）。
# 沿用统一约定：分数越高越值得买；方向是经济先验，真伪由 IC 验证。
# 与 A股基本面因子的差异：美股无日频估值源，价值因子靠 TTM EPS / 价格 自算；
# 质量/成长直接用财报披露的 ROE / 同比增速（已是「越高越好」的量纲，无需变换）。

def us_earnings_yield(eps_ttm: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """美股价值因子：盈利收益率 = TTM EPS / 价格（= 1/PE，越便宜分越高）。

    A股有日频 PE 可直接取倒数；美股无日频估值源，故用「最近 4 个已公告季度 EPS 之和」
    除以当日收盘价自算。eps_ttm 已在 loader 按公告日防前视，close 是前复权价——
    注意分子是名义 EPS、分母是复权价，二者口径不完全一致（复权会缩放历史价），但
    横截面排序只看同一日各股的相对高低，整体缩放不改变当日排序，可接受。负 EPS
    （亏损）会得到负盈利收益率，横截面里自然排在最后，符合「不便宜」的经济含义。
    """
    return eps_ttm / close.replace(0, np.nan)


def us_quality_roe(roe: pd.DataFrame) -> pd.DataFrame:
    """美股质量因子：直接用财报披露的平均 ROE（净资产收益率，越高越好）。

    与 A股用恒等式 PB/PE 反推 ROE 不同，美股财报直接给了 ROE_AVG，更直接可靠。
    ROE 越高 = 越能用净资产赚钱 = 质量越好，已是「越高越好」量纲，无需变换。
    """
    return roe


def us_growth(yoy: pd.DataFrame) -> pd.DataFrame:
    """美股成长因子：同比增速（营收同比或净利同比，越高越好）。

    薄包装：财报披露的同比增速本身就是「越高越好」的成长信号，直接返回。
    单列出来是为了语义清晰、与其它因子统一从工厂函数取用。方向是先验，
    真伪由 IC 验证（高增速既可能延续，也可能已被price-in，交给数据判断）。
    """
    return yoy
