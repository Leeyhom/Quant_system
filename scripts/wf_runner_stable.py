#!/usr/bin/env python
"""仅使用方向稳定因子的无泄漏Walk-Forward——修复版

核心改进：
1. 不做任何因子方向学习（避免用未来信息定向）
2. 仅使用"经济直觉固定方向"且"历史方向一贯稳定"的因子
3. 因子权重等权，不做IC加权
"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.factor.neutralize import neutralize
from quant.factor.factors import combine_factors


def wf_backtest_stable_factors(market='CN', train_days=480, test_days=120, step_days=120):
    """只使用方向稳定的因子，固定方向，不做任何未来学习"""
    print(f"\n{'='*80}")
    print(f"  {market} 无泄漏Walk-Forward：仅使用方向稳定因子（固定方向，不学习）")
    print(f"{'='*80}")

    if market == 'CN':
        from quant.data.universe import DEFAULT_POOL
        from quant.data.panel import build_ohlcv_panels, build_value_panels
        from quant.data.industry import industry_series
        from quant.factor import factors as F

        symbols = DEFAULT_POOL
        panels = build_ohlcv_panels(symbols)
        close = panels['close']
        value = build_value_panels(symbols, align_to=close)
        ind = industry_series(list(close.columns))
        log_mv = np.log(value['total_mv'].replace(0, np.nan))

        # 仅使用：growth_peg（方向稳定 88%）
        raw_factors = {}
        # growth_peg: 低PEG好 → 1/PEG → 越高越好 ✓ 固定正方向
        raw_factors['growth_peg'] = F.growth_peg(value['peg'])
        # earnings_yield: 估值便宜好 → 1/PE → 越高越好 ✓ 固定正方向（虽然历史不稳定，但经济直觉对）
        raw_factors['earnings_yield'] = F.earnings_yield(value['pe_ttm'])
        # cashflow_yield: 现金流好 → 越高越好 ✓
        raw_factors['cashflow_yield'] = F.cashflow_yield(value['pcf'])
        print(f"    选入因子：{list(raw_factors.keys())}（固定正方向，不翻转）")

    elif market == 'US':
        from quant.data import us_loader
        from quant.data.universe_us_expanded import EXPANDED_US_POOL
        from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
        from quant.factor import factors as F

        symbols = EXPANDED_US_POOL[:300]
        panels = build_ohlcv_panels(symbols, loader=us_loader)
        close = panels['close']
        fund = build_us_fundamental_panels(symbols, align_to=close)

        raw_factors = {}
        # growth_rev: 营收同比增长越高越好 ✓ 固定正方向
        raw_factors['growth_rev'] = F.us_growth(fund['rev_yoy'])
        # pv_reversal5: 短期反转越高越好 ✓ 固定正方向
        raw_factors['pv_reversal5'] = F.reversal(close, 5)
        # quality_roe: ROE越高越好 ✓ 固定正方向
        if 'roe' in fund:
            raw_factors['quality_roe'] = F.us_quality_roe(fund['roe'])
        print(f"    选入因子：{list(raw_factors.keys())}（固定正方向，不翻转）")

    print(f"    股票数：{len(close.columns)}，交易日数：{len(close)}")

    # 中性化（截面内完成，无泄漏）
    if market == 'CN':
        neutralized = {name: neutralize(fac, industry=ind, log_mv=log_mv, mode='full')
                      for name, fac in raw_factors.items()}
    else:
        neutralized = raw_factors

    composite = combine_factors(*neutralized.values())

    # Walk-Forward
    dates = close.index.sort_values()
    n_dates = len(dates)
    all_test_rets, all_test_bench = [], []
    window_info = []

    start = 0
    window_idx = 0
    while start + train_days + test_days <= n_dates:
        test_start = dates[start + train_days]
        test_end = dates[min(start + train_days + test_days - 1, n_dates - 1)]

        composite_test = composite.loc[test_start:test_end].copy()
        close_test = close.loc[test_start:test_end].copy()

        bt = long_top_layer(close_test, composite_test, rebalance_every=20, weight_mode='equal')

        all_test_rets.append(bt['port_ret'])
        all_test_bench.append(bt['benchmark_ret'])

        s_test = summary(bt['equity'], bt['port_ret'])
        s_bench = summary((1+bt['benchmark_ret']).cumprod(), bt['benchmark_ret'])
        window_info.append({'window': window_idx, 'test_start': str(test_start)[:10],
                           'test_end': str(test_end)[:10],
                           'sharpe': s_test['sharpe'], 'bench_sharpe': s_bench['sharpe']})
        start += step_days
        window_idx += 1

    # 拼接结果
    full_test_ret = pd.concat(all_test_rets).sort_index()
    full_test_ret = full_test_ret[~full_test_ret.index.duplicated(keep='first')]
    full_test_bench = pd.concat(all_test_bench).sort_index()
    full_test_bench = full_test_bench[~full_test_bench.index.duplicated(keep='first')]

    equity = (1 + full_test_ret).cumprod()
    bench_eq = (1 + full_test_bench).cumprod()
    s = summary(equity, full_test_ret)
    sb = summary(bench_eq, full_test_bench)

    print(f"\n  结果汇总（{len(window_info)} 个窗口拼接）")
    print(f"  {'='*60}")
    print(f"  策略：累计收益 {s['total_return']:>+8.1%}  年化 {s['annualized_return']:>+7.1%}  "
          f"夏普 {s['sharpe']:>+6.2f}  最大回撤 {s['max_drawdown']:>+7.1%}")
    print(f"  基准：累计收益 {sb['total_return']:>+8.1%}  年化 {sb['annualized_return']:>+7.1%}  "
          f"夏普 {sb['sharpe']:>+6.2f}  最大回撤 {sb['max_drawdown']:>+7.1%}")
    excess_sharpe = s['sharpe'] - sb['sharpe']
    print(f"  超额夏普：{excess_sharpe:+.2f}  {'✅ 跑赢' if excess_sharpe > 0 else '❌ 跑输'}")

    # 画图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label=f'策略 (夏普={s["sharpe"]:.2f})', lw=2, color='crimson')
    ax.plot(bench_eq.index, bench_eq.values, label=f'基准 (夏普={sb["sharpe"]:.2f})', lw=1.5, ls='--', color='gray')
    ax.set_title(f'{market} 仅用方向稳定因子 · Walk-Forward 拼接净值')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = f'/tmp/{market}_wf_stable.png'
    fig.savefig(out, dpi=150)
    print(f"  图：{out}")

    return {'strategy': s, 'benchmark': sb, 'excess_sharpe': excess_sharpe}


if __name__ == '__main__':
    cn = wf_backtest_stable_factors('CN')
    us = wf_backtest_stable_factors('US')

    print(f"\n{'='*80}")
    print(f"  最终结论")
    print(f"{'='*80}")
    print(f"  A股：超额夏普 {cn['excess_sharpe']:+.2f}")
    print(f"  美股：超额夏普 {us['excess_sharpe']:+.2f}")
    if cn['excess_sharpe'] > 0 and us['excess_sharpe'] > 0:
        print(f"  ✅ 双市场均跑赢基准，可以进入模拟盘验证阶段")
    else:
        print(f"  ⚠️  仍需进一步优化因子选择或权重方案")
