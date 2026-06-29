#!/usr/bin/env python
"""因子方向稳定性分析——为什么Walk-Forward跑输基准？
核心问题：因子方向在不同 regime 是否会系统性翻转？
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, pandas as pd
from quant.backtest.ic_analysis import daily_ic, ic_summary, forward_returns

print("="*80)
print("  因子方向稳定性分析")
print("="*80)

# === A股 ===
print("\n【A股】因子IC方向逐年变化：")
from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.industry import industry_series
from scripts.wf_runner import build_factors_cn

symbols = DEFAULT_POOL
panels = build_ohlcv_panels(symbols)
close = panels['close']
value = build_value_panels(symbols, align_to=close)
raw = build_factors_cn(close, value, panels)
fwd = forward_returns(close, horizon=20)
ind = industry_series(list(close.columns))
log_mv = np.log(value['total_mv'].replace(0, np.nan))

from quant.factor.neutralize import neutralize
for name in list(raw.keys()):
    raw[name] = neutralize(raw[name], industry=ind, log_mv=log_mv, mode='full')

years = sorted(set(close.index.year))
all_ics = {}
for name, fac in raw.items():
    year_ics = []
    for y in years:
        mask = fwd.index.year == y
        if mask.sum() > 20:
            s = ic_summary(daily_ic(fac.loc[mask], fwd.loc[mask]))
            year_ics.append(s['mean_ic'])
        else:
            year_ics.append(np.nan)
    all_ics[name] = year_ics

print(f'{"因子":20s} ' + ' '.join([f'{y%100:4d}' for y in years]) + '   稳定率')
print('-'*80)

stable_factors = []
for name, ics in all_ics.items():
    signs = ['  +' if pd.notna(x) and x>0 else '  -' if pd.notna(x) else '   ' for x in ics]
    valid = [x for x in ics if pd.notna(x) and abs(x) > 1e-8]
    pos_rate = sum(1 for x in valid if x > 0) / len(valid) if valid else 0
    stab = max(pos_rate, 1-pos_rate)
    print(f'{name:20s} ' + ''.join(signs) + f'  {stab:.0%}')
    if stab >= 0.80 and len(valid) >= 5:
        stable_factors.append(name)

print(f'\n方向稳定率 >= 80% 的因子：{stable_factors}')

# === 美股 ===
print("\n\n" + "="*80)
print("【美股】因子IC方向逐年变化：")
from quant.data import us_loader
from quant.data.universe_us_expanded import EXPANDED_US_POOL
from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
from scripts.wf_runner import build_factors_us

symbols = EXPANDED_US_POOL[:300]
panels = build_ohlcv_panels(symbols, loader=us_loader)
close = panels['close']
fund = build_us_fundamental_panels(symbols, align_to=close)
raw_us = build_factors_us(close, fund, panels)
fwd_us = forward_returns(close, horizon=20)

years_us = sorted(set(close.index.year))
all_ics_us = {}
for name, fac in raw_us.items():
    year_ics = []
    for y in years_us:
        mask = fwd_us.index.year == y
        if mask.sum() > 20:
            s = ic_summary(daily_ic(fac.loc[mask], fwd_us.loc[mask]))
            year_ics.append(s['mean_ic'])
        else:
            year_ics.append(np.nan)
    all_ics_us[name] = year_ics

print(f'{"因子":20s} ' + ' '.join([f'{y%100:4d}' for y in years_us]) + '   稳定率')
print('-'*80)

stable_factors_us = []
for name, ics in all_ics_us.items():
    signs = ['  +' if pd.notna(x) and x>0 else '  -' if pd.notna(x) else '   ' for x in ics]
    valid = [x for x in ics if pd.notna(x) and abs(x) > 1e-8]
    pos_rate = sum(1 for x in valid if x > 0) / len(valid) if valid else 0
    stab = max(pos_rate, 1-pos_rate)
    print(f'{name:20s} ' + ''.join(signs) + f'  {stab:.0%}')
    if stab >= 0.80 and len(valid) >= 5:
        stable_factors_us.append(name)

print(f'\n方向稳定率 >= 80% 的因子：{stable_factors_us}')
print("\n结论：方向不稳定的因子是WF跑输的核心原因——这些因子在train段和test段方向翻转！")
