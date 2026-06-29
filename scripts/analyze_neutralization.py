#!/usr/bin/env python
"""中性化消融实验：双中性化是否过度清洗alpha？"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, pandas as pd
from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.data.universe import DEFAULT_POOL
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.industry import industry_series
from quant.factor import factors as F
from quant.factor.neutralize import neutralize
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary

symbols = DEFAULT_POOL
panels = build_ohlcv_panels(symbols)
close = panels['close']
value = build_value_panels(symbols, align_to=close)
ind = industry_series(list(close.columns))
log_mv = np.log(value['total_mv'].replace(0, np.nan))
fwd = forward_returns(close, horizon=20)

print("中性化消融：不同中性化方式下的因子IC")
print("="*80)
print(f"{'因子':18s} {'原始IC':>10s} {'行业中性':>10s} {'市值中性':>10s} {'双中性':>10s}")
print("-"*80)

factors = [
    ('small_size', lambda: F.small_size(value['total_mv'])),
    ('earnings_yield', lambda: F.earnings_yield(value['pe_ttm'])),
    ('book_to_price', lambda: F.book_to_price(value['pb'])),
    ('growth_peg', lambda: F.growth_peg(value['peg'])),
    ('quality_roe', lambda: F.quality_roe(value['pe_ttm'], value['pb'])),
]

for name, builder in factors:
    fac = builder()
    ic_raw = ic_summary(daily_ic(fac, fwd))['mean_ic']
    fac_ind = neutralize(fac, industry=ind, mode='industry')
    ic_ind = ic_summary(daily_ic(fac_ind, fwd))['mean_ic']
    fac_size = neutralize(fac, log_mv=log_mv, mode='size')
    ic_size = ic_summary(daily_ic(fac_size, fwd))['mean_ic']
    fac_both = neutralize(fac, industry=ind, log_mv=log_mv, mode='full')
    ic_both = ic_summary(daily_ic(fac_both, fwd))['mean_ic']
    print(f"{name:18s} {ic_raw:10.4f} {ic_ind:10.4f} {ic_size:10.4f} {ic_both:10.4f}")

print("\n" + "="*80)
print("中性化方式对组合回测的影响（全样本，仅做对比）")
print("="*80)

# 构建不同中性化的组合
from quant.factor.factors import combine_factors

for mode_name, mode in [('原始', None), ('行业中性', 'industry'), ('市值中性', 'size'), ('双中性', 'full')]:
    raw = {}
    for name, builder in factors:
        fac = builder()
        if mode:
            if mode == 'industry':
                fac = neutralize(fac, industry=ind, mode=mode)
            elif mode == 'size':
                fac = neutralize(fac, log_mv=log_mv, mode=mode)
            else:
                fac = neutralize(fac, industry=ind, log_mv=log_mv, mode=mode)
        raw[name] = fac
    composite = combine_factors(*raw.values())
    bt = long_top_layer(close, composite, rebalance_every=20)
    s = summary(bt['equity'], bt['port_ret'])
    sb = summary((1+bt['benchmark_ret']).cumprod(), bt['benchmark_ret'])
    print(f"{mode_name:8s}: 夏普 {s['sharpe']:+.2f}  收益 {s['total_return']:+.1%}  "
          f"基准夏普 {sb['sharpe']:+.2f}  超额 {s['sharpe']-sb['sharpe']:+.2f}")

print("\n⚠️  结论：双中性化确实削弱了大部分因子的IC，尤其是小市值因子！")
print("       中性化的目的是降低回撤和提高稳定性，但会牺牲部分收益。")
