"""neutralize —— 因子中性化：剥离行业 / 市值 beta，留下纯选股能力。

为什么要中性化（见 docs/12，由 M12 诊断实验驱动）：
   原始价值因子选出的 top-10 里 9 只是银行+地产——它其实在「押低估值行业」，
   而非「行业内选好公司」。一旦这些行业回调，组合就崩。中性化把因子里
   「整个行业便宜 / 整个盘子大小」的成分减掉，只留下「同行业、同市值档位里，
   这只股票相对更好」的纯 alpha。实验证明：行业中性后 earnings_yield 的
   ICIR 0.30→0.39、t 6.46→8.40，选股力反而更干净更稳。

防未来函数：所有中性化都在**单个交易日的横截面内**完成（减同日行业均值、
对同日市值回归），不使用任何跨期/未来信息，因此安全。回测层仍会对中性化后
的因子统一 shift(1)。

方法论借鉴（Barra/WorldQuant 风格，只借鉴思想不抄因子）：
   - 行业中性 = 减去所属行业的截面均值（等价于对行业哑变量回归取残差）。
   - 市值中性 = 对 ln(市值) 做截面 OLS 回归，取残差。
   - 顺序中性 = 先行业去均值，再对市值回归取残差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def industry_neutralize(factor: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """行业中性化：每个交易日，因子值减去所在行业的当日截面均值。

    参数:
        factor: 因子面板（日期 × 股票）。
        industry: index=股票代码、value=行业名 的 Series（来自 industry_series）。

    返回:
        同形状面板，每个交易日每个行业内的因子均值≈0（NaN 不参与）。
    """
    # 按列（股票）对应的行业分组，逐日减组内均值。
    ind = industry.reindex(factor.columns)
    # 转置成 股票 × 日期，按行业分组对每列（每天）去均值，再转置回来。
    fT = factor.T
    demeaned = fT.groupby(ind).transform(lambda g: g - g.mean())
    return demeaned.T


def size_neutralize(factor: pd.DataFrame, log_mv: pd.DataFrame) -> pd.DataFrame:
    """市值中性化：每个交易日，把因子对 ln(市值) 做横截面 OLS 回归，取残差。

    残差 = 因子中无法被市值线性解释的部分，即剥离了「大盘/小盘」风格暴露。

    参数:
        factor: 因子面板。
        log_mv: ln(总市值) 面板，与 factor 同形状对齐。

    返回:
        残差面板（同形状）。当日有效样本 < 3 时该日保持原值（无法稳健回归）。
    """
    out = factor.copy()
    log_mv = log_mv.reindex_like(factor)
    for dt in factor.index:
        y = factor.loc[dt]
        x = log_mv.loc[dt]
        mask = y.notna() & x.notna()
        if mask.sum() < 3:
            continue
        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)
        # 设计矩阵 [1, x]，最小二乘解 beta，残差 = y - X·beta
        X = np.column_stack([np.ones_like(xv), xv])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        out.loc[dt, mask.to_numpy()] = resid
    return out


def neutralize(
    factor: pd.DataFrame,
    industry: pd.Series | None = None,
    log_mv: pd.DataFrame | None = None,
    mode: str = "full",
) -> pd.DataFrame:
    """组合中性化（Barra 风格顺序：先行业去均值，再对市值回归取残差）。

    参数:
        industry: 行业 Series；None 则跳过行业中性。
        log_mv:   ln(市值) 面板；None 则跳过市值中性。
        mode:
            "industry" —— 只做行业中性；
            "size"     —— 只做市值中性；
            "full"     —— 行业 + 市值双中性（默认）。

    返回:
        中性化后的因子面板。便于做「原始 / 行业 / 双中性」消融对比。
    """
    out = factor
    if mode in ("industry", "full") and industry is not None:
        out = industry_neutralize(out, industry)
    if mode in ("size", "full") and log_mv is not None:
        out = size_neutralize(out, log_mv)
    return out
