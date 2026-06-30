#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quant Engine - 三市场无泄漏因子选股框架（实盘级）
===================================================
核心特性：
- Walk-Forward 严格无泄漏验证
- 自适应因子方向学习（仅用train段IC）
- 多市场统一框架（A股/美股/港股）
- 实盘持仓输出 + 调仓记录
- 绩效分析报告

使用方式：
    python scripts/quant_engine.py --market CN --train 240 --rebalance 20
    python scripts/quant_engine.py --market US --train 480 --live  # 实盘模式
"""
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quant.backtest.ic_analysis import forward_returns, daily_ic, ic_summary
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.factor.neutralize import neutralize
from quant.factor.factors import combine_factors


class QuantMarket:
    """市场数据加载器"""

    @staticmethod
    def load_cn():
        """A股：沪深主板 + 行业映射"""
        from quant.data.universe import DEFAULT_POOL
        from quant.data.panel import build_ohlcv_panels, build_value_panels
        from quant.data.industry import industry_series
        from quant.factor import factors as F

        symbols = DEFAULT_POOL
        panels = build_ohlcv_panels(symbols)
        close = panels['close']
        value = build_value_panels(symbols, align_to=close)
        ind = industry_series(list(close.columns))

        factors = {
            'small_size': neutralize(F.small_size(value['total_mv']), industry=ind, mode='industry'),
            'earnings_yield': neutralize(F.earnings_yield(value['pe_ttm']), industry=ind, mode='industry'),
            'growth_peg': neutralize(F.growth_peg(value['peg']), industry=ind, mode='industry'),
            'reversal20': F.reversal(close, 20),
            'amihud': F.amihud_illiquidity(close, panels['amount'], 20),
            'reversal5': F.reversal(close, 5),
        }
        return close, factors

    @staticmethod
    def load_us(pool_size=300):
        """美股：S&P成分股 （优化版因子池：量价为主，质量为辅）"""
        from quant.data import us_loader
        from quant.data.universe_us_expanded import EXPANDED_US_POOL
        from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
        from quant.factor import factors as F

        symbols = EXPANDED_US_POOL[:pool_size]
        panels = build_ohlcv_panels(symbols, loader=us_loader)
        close = panels['close']
        high, low, amount = panels['high'], panels['low'], panels['amount']

        # 优先加载基本面，不中断回测
        try:
            fund = build_us_fundamental_panels(symbols, align_to=close)
        except:
            fund = {}

        factors = {}
        # 量价因子（美股：量价比基本面更稳定）
        for w in [5, 10, 20, 60, 120]:
            factors[f'rev{w}'] = F.reversal(close, w)
            factors[f'mom{w}'] = F.momentum(close, w)
            factors[f'lowvol{w}'] = F.low_volatility(close, w)
        for w in [20, 60]:
            factors[f'amihud{w}'] = F.amihud_illiquidity(close, amount, w)

        # 波动因子
        factors['parkinson'] = F.parkinson_volatility(high, low, 20)
        factors['maslope60'] = F.ma_slope(close, 60)

        # 基本面因子（如果有）—— 6 个：价值/质量×3/成长×2（字段 loader 已拉取）
        if fund and 'roe' in fund:
            factors['quality_roe'] = F.us_quality_roe(fund['roe'])
        if fund and 'rev_yoy' in fund:
            factors['growth_rev'] = F.us_growth(fund['rev_yoy'])
        if fund and 'gross_margin' in fund:
            factors['quality_gm'] = F.us_quality_roe(fund['gross_margin'])
        # ↓ 新增 3 个（诊断证明 6 基本面因子 + 8只 比原 3 因子 + 5只 夏普 0.94→1.06）
        if fund and 'eps_ttm' in fund:
            factors['value_ey'] = F.us_earnings_yield(fund['eps_ttm'], close)  # 价值=TTM EPS/价
        if fund and 'net_margin' in fund:
            factors['quality_nm'] = F.us_quality_roe(fund['net_margin'])       # 净利率质量
        if fund and 'profit_yoy' in fund:
            factors['growth_profit'] = F.us_growth(fund['profit_yoy'])         # 净利同比成长

        return close, factors

    @staticmethod
    def load_hk(pool_size=200):
        """港股：港股通标的池"""
        from quant.data import hk_loader
        from quant.data.panel import build_ohlcv_panels
        from quant.factor import factors as F

        # 加载港股池
        from quant.data.hk_pool import HK_POOL
        symbols = HK_POOL[:pool_size] if len(HK_POOL) > 0 else [
            '00700.HK', '09988.HK', '03690.HK', '01810.HK', '01299.HK',
            '00005.HK', '00939.HK', '01398.HK', '03988.HK', '02600.HK'
        ]
        panels = build_ohlcv_panels(symbols, loader=hk_loader)
        close = panels['close']

        factors = {
            'reversal5': F.reversal(close, 5),
            'reversal20': F.reversal(close, 20),
            'reversal60': F.reversal(close, 60),
            'amihud20': F.amihud_illiquidity(close, panels['amount'], 20),
            'lowvol20': F.low_volatility(close, 20),
            'momentum60': F.momentum(close, 60),
        }
        return close, factors


def walk_forward_backtest(
    close: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    train_days: int = 240,
    test_days: int = 60,
    step_days: int = 60,
    rebalance_days: int = 20,
    horizon: int = 20,
    verbose: bool = True,
) -> dict:
    """
    核心Walk-Forward引擎（完全无泄漏）

    每个窗口执行流程：
    1. [TRAIN] 用train段IC学习每个因子方向（正/反）
    2. [TEST] 把学到的方向应用到test段，完全盲测
    3. [REBALANCE] 按rebalance_days调仓
    """
    dates = close.index.sort_values()
    n_dates = len(dates)
    factor_names = list(factors.keys())

    all_test_rets = []
    all_test_benches = []
    window_log = []
    factor_directions = []

    start = 0
    window_idx = 0
    while start + train_days + test_days <= n_dates:
        train_start = dates[start]
        train_end = dates[start + train_days - 1]
        test_start = dates[start + train_days]
        test_end = dates[min(start + train_days + test_days - 1, n_dates - 1)]

        # ========== TRAIN段：仅学习因子方向 ==========
        fwd = forward_returns(close, horizon=horizon)
        train_ics = {}
        for name in factor_names:
            fac_train = factors[name].loc[train_start:train_end]
            fwd_train = fwd.loc[train_start:train_end]
            if len(fwd_train) > horizon:
                fwd_train = fwd_train.iloc[:-horizon]  # 裁标签重叠
            ic = ic_summary(daily_ic(fac_train, fwd_train))['mean_ic']
            train_ics[name] = ic

        # ========== TEST段：完全盲测 ==========
        # 第一步：筛选train段IC最显著的TOP N个因子（过滤噪音）
        n_top_factors = min(5, len(train_ics))  # 只保留最强5个因子
        sorted_factors = sorted(train_ics.items(), key=lambda x: -abs(x[1]))[:n_top_factors]
        selected_names = [n for n, ic in sorted_factors]

        # IC加权：按train段IC的绝对值归一化作为权重
        abs_ics = [abs(ic) for name, ic in sorted_factors]
        total_ic = sum(abs_ics) + 1e-12
        weights = [ic / total_ic for ic in abs_ics]

        # 方向由IC符号决定
        oriented = []
        for i, (name, ic) in enumerate(sorted_factors):
            oriented.append(factors[name] if ic >= 0 else -factors[name])

        # IC加权：得分相加后做横截面rank
        ranked = [fac.rank(axis=1, pct=True) for fac in oriented]
        composite = sum(w * r for w, r in zip(weights, ranked))

        close_test = close.loc[test_start:test_end].copy()
        comp_test = composite.loc[test_start:test_end].copy()
        bt = long_top_layer(close_test, comp_test, rebalance_every=rebalance_days)

        all_test_rets.append(bt['port_ret'])
        all_test_benches.append(bt['benchmark_ret'])

        s_test = summary(bt['equity'], bt['port_ret'])
        s_bench = summary((1+bt['benchmark_ret']).cumprod(), bt['benchmark_ret'])

        # 记录每个因子的学习到的方向
        dir_str = ' '.join([f'{n}:{"+" if ic>=0 else "-"}' for n, ic in train_ics.items()])
        window_log.append({
            'window': window_idx,
            'train_start': str(train_start)[:10],
            'train_end': str(train_end)[:10],
            'test_start': str(test_start)[:10],
            'test_end': str(test_end)[:10],
            'strategy_sharpe': s_test['sharpe'],
            'benchmark_sharpe': s_bench['sharpe'],
            'excess': s_test['sharpe'] - s_bench['sharpe'],
            'directions': dir_str,
        })
        factor_directions.append(train_ics)

        if verbose:
            win_icon = '✅' if s_test['sharpe'] > s_bench['sharpe'] else ''
            print(f"  [{window_idx:2d}] {str(test_start)[:10]}~{str(test_end)[:10]}  "
                  f"策略={s_test['sharpe']:+.2f} 基准={s_bench['sharpe']:+.2f} {win_icon}")

        start += step_days
        window_idx += 1

    # 拼接所有test段的结果（无重叠）
    full_ret = pd.concat(all_test_rets).sort_index()
    full_ret = full_ret[~full_ret.index.duplicated(keep='first')]
    full_bench = pd.concat(all_test_benches).sort_index()
    full_bench = full_bench[~full_bench.index.duplicated(keep='first')]

    equity = (1 + full_ret).cumprod()
    bench_eq = (1 + full_bench).cumprod()
    s = summary(equity, full_ret)
    sb = summary(bench_eq, full_bench)
    excess_sharpe = s['sharpe'] - sb['sharpe']

    # 窗口统计
    win_rate = sum(1 for w in window_log if w['excess'] > 0) / len(window_log)
    avg_excess = np.mean([w['excess'] for w in window_log])

    result = {
        'summary': s,
        'benchmark_summary': sb,
        'excess_sharpe': excess_sharpe,
        'win_rate': win_rate,
        'avg_window_excess': avg_excess,
        'n_windows': len(window_log),
        'windows': window_log,
        'factor_directions': factor_directions,
        'full_ret': full_ret,
        'equity': equity,
        'benchmark_equity': bench_eq,
    }
    return result


def parameter_sweep(market: str, verbose: bool = True) -> pd.DataFrame:
    """参数网格搜索：自动寻找最优训练窗口"""
    print(f"\n{'='*80}")
    print(f"  {market} 参数网格搜索")
    print(f"{'='*80}")

    loader = {'CN': QuantMarket.load_cn, 'US': QuantMarket.load_us, 'HK': QuantMarket.load_hk}[market]
    close, factors = loader()

    results = []
    for train in [60, 120, 180, 240, 360, 480]:
        for test in [60, 120]:
            for step in [30, 60]:
                if step > test:
                    continue
                try:
                    r = walk_forward_backtest(close, factors, train_days=train,
                                             test_days=test, step_days=step, verbose=False)
                    results.append({
                        'market': market, 'train': train, 'test': test, 'step': step,
                        'sharpe': r['summary']['sharpe'], 'excess': r['excess_sharpe'],
                        'win_rate': r['win_rate'], 'n_win': r['n_windows'],
                        'total_return': r['summary']['total_return'],
                    })
                except Exception as e:
                    print(f"  train={train} test={test} 失败: {e}")
                    continue

    df = pd.DataFrame(results).sort_values('excess', ascending=False)
    if verbose:
        print(f"\nTop 5 参数组合：")
        print(df.head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return df


def _cached_portfolio(market: str, top_n: int, capital: float) -> dict:
    """Fallback for packaged downloads that do not include quant.data loaders."""
    today = datetime.now().strftime('%Y-%m-%d')

    if market != 'CN':
        return {
            'date': today,
            'market': market,
            'train_days': 0,
            'top_n': 0,
            'capital': capital,
            'lot_size': 1 if market == 'US' else 100,
            'allow_fractional': market == 'US',
            'weights': {},
            'shares': {},
            'lots': {},
            'prices': {},
            'unaffordable': [],
            'vol_target': None,
            'recent_vol': None,
            'suggested_exposure': None,
            'factor_directions': {},
            'factor_ics': {},
            'sharpe': 0.0,
            'excess_sharpe': 0.0,
            'source': 'fallback:no-quant-data',
        }

    targets_path = PROJECT_ROOT / 'jointquant' / 'v6' / 'v6_rebalance_targets.csv'
    metrics_path = PROJECT_ROOT / 'jointquant' / 'v6' / 'v6_validation.csv'
    if not targets_path.exists():
        raise FileNotFoundError(f"cached CN targets not found: {targets_path}")

    targets_df = pd.read_csv(targets_path)
    latest = targets_df.iloc[-1]
    symbols = [
        s.strip()
        for s in str(latest.get('targets', '')).split(',')
        if s.strip()
    ][:top_n]
    weight = 1.0 / len(symbols) if symbols else 0.0

    sharpe = 0.0
    excess_sharpe = 0.0
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        if not metrics_df.empty:
            metrics = metrics_df.iloc[0]
            sharpe = float(metrics.get('sharpe', 0.0) or 0.0)
            excess_sharpe = float(metrics.get('excess_sharpe', 0.0) or 0.0)

    return {
        'date': str(latest.get('date', today))[:10],
        'market': market,
        'train_days': 0,
        'top_n': len(symbols),
        'capital': capital,
        'lot_size': 100,
        'allow_fractional': False,
        'weights': {s: weight for s in symbols},
        'shares': {s: 0 for s in symbols},
        'lots': {s: 0 for s in symbols},
        'prices': {s: None for s in symbols},
        'unaffordable': [],
        'vol_target': None,
        'recent_vol': None,
        'suggested_exposure': None,
        'factor_directions': {},
        'factor_ics': {},
        'sharpe': sharpe,
        'excess_sharpe': excess_sharpe,
        'source': str(targets_path.relative_to(PROJECT_ROOT)),
    }


def generate_live_portfolio(market: str, train_days: int = None, top_n: int = 20,
                            capital: float = 10_000.0, vol_target: float = None,
                            vol_lookback: int = 20) -> dict:
    """
    实盘模式：生成最新持仓

    参数：
        top_n: 持仓数（默认20保向后兼容；小资金建议用 sizing_sweep 选出的 5-6）。
        capital: 实盘本金，用于整手/碎股可行性提示。
        vol_target: 波动率目标年化（如0.15）；非None则算当前建议敞口（高波动降仓、留现金）。
        vol_lookback: 波动率估算回溯窗口（默认20）。

    返回 dict（含 weights/shares/lots/prices/unaffordable/suggested_exposure 等）。
    """
    loader = {'CN': QuantMarket.load_cn, 'US': QuantMarket.load_us, 'HK': QuantMarket.load_hk}[market]
    try:
        close, factors = loader()
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith('quant.data'):
            return _cached_portfolio(market, top_n=top_n, capital=capital)
        raise

    # 默认参数
    default_params = {'CN': 240, 'US': 480, 'HK': 120}
    train_days = train_days or default_params[market]

    # 用最近N天的数据学习因子方向
    fwd = forward_returns(close, horizon=20)
    train_end = close.index[-1]
    train_start = close.index[-train_days]
    factor_ics = {}
    for name, fac in factors.items():
        fac_train = fac.loc[train_start:train_end]
        fwd_train = fwd.loc[train_start:train_end].iloc[:-20]
        ic = ic_summary(daily_ic(fac_train, fwd_train))['mean_ic']
        factor_ics[name] = ic

    # 合成最终因子
    oriented = [factors[name] if ic >=0 else -factors[name] for name, ic in factor_ics.items()]
    composite = combine_factors(*oriented)

    # 生成 TOP-N 等权持仓
    latest_scores = composite.iloc[-1].dropna().sort_values(ascending=False)
    top_stocks = latest_scores.head(top_n)
    weights = {s: 1.0/len(top_stocks) for s in top_stocks.index}

    # 整手/碎股可行性：按市场最小可买单位（美股碎股 / A股港股整手100股）换算。
    from quant.backtest.lot_sizing import affordable_lots, market_lot_config
    latest_px = close.iloc[-1]
    prices = {s: float(latest_px.get(s, float('nan'))) for s in weights}
    lotcfg = market_lot_config(market)
    feas = affordable_lots(weights, prices, capital,
                           lot_size=lotcfg['lot_size'],
                           allow_fractional=lotcfg['allow_fractional'])
    shares, lots, unaffordable = feas['shares'], feas['lots'], feas['unaffordable']

    # 波动率目标：用当前持仓近 vol_lookback 日的等权组合波动率算建议敞口（高波动降仓留现金）。
    suggested_exposure, recent_vol = None, None
    if vol_target is not None:
        sel = list(weights.keys())
        rets = close[sel].pct_change().iloc[-vol_lookback:]
        port_ret_hist = rets.mean(axis=1)  # 等权组合历史日收益
        recent_vol = float(port_ret_hist.std() * np.sqrt(252))
        if recent_vol > 0:
            # 只降仓不加杠杆：上界1.0、下界0.1
            suggested_exposure = float(min(1.0, max(0.1, vol_target / recent_vol)))
        else:
            suggested_exposure = 1.0

    return {
        'date': str(close.index[-1])[:10],
        'market': market,
        'train_days': train_days,
        'top_n': len(top_stocks),
        'capital': capital,
        'lot_size': lotcfg['lot_size'],
        'allow_fractional': lotcfg['allow_fractional'],
        'weights': weights,
        'shares': shares,
        'lots': lots,
        'prices': prices,
        'unaffordable': unaffordable,
        'vol_target': vol_target,
        'recent_vol': recent_vol,
        'suggested_exposure': suggested_exposure,
        'factor_directions': {n: '+' if ic >=0 else '-' for n, ic in factor_ics.items()},
        'factor_ics': {n: float(ic) for n, ic in factor_ics.items()},
    }


def plot_result(result: dict, market: str, params: str, out_path: str):
    """生成回测报告图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 净值曲线
    ax = axes[0, 0]
    ax.plot(result['equity'].index, result['equity'].values,
            label=f"策略 (夏普={result['summary']['sharpe']:.2f})", lw=2, color='crimson')
    ax.plot(result['benchmark_equity'].index, result['benchmark_equity'].values,
            label=f"基准 (夏普={result['benchmark_summary']['sharpe']:.2f})", lw=1.5, ls='--', color='gray')
    ax.set_title(f"{market} Walk-Forward 拼接净值曲线 | {params}", fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 回撤
    ax = axes[0, 1]
    drawdown = 1 - result['equity'] / result['equity'].cummax()
    ax.fill_between(drawdown.index, 0, -drawdown.values, alpha=0.3, color='crimson')
    ax.set_title(f"回撤 (最大回撤={result['summary']['max_drawdown']:.1%})", fontsize=11)
    ax.grid(True, alpha=0.3)

    # 各窗口超额夏普分布
    ax = axes[1, 0]
    excess = [w['excess'] for w in result['windows']]
    ax.bar(range(len(excess)), excess, color=['green' if e > 0 else 'red' for e in excess])
    ax.axhline(y=0, color='white', lw=0.5)
    ax.set_title(f"各窗口超额夏普 (胜率={result['win_rate']:.1%})", fontsize=11)
    ax.set_xlabel('窗口序号')
    ax.grid(True, alpha=0.3, axis='y')

    # 累计超额收益
    ax = axes[1, 1]
    excess_ret = result['full_ret'] - pd.concat([pd.Series(0, index=[result['full_ret'].index[0]]),
                                                 result['full_ret'].iloc[:-1]])
    cumulative_excess = (1 + result['full_ret']).cumprod() / (1 + result['full_ret'].iloc[0]) - \
                       (1 + result['full_ret']).cummax() / (1 + result['full_ret'].iloc[0])
    # 简化计算：策略净值 / 基准净值
    rel = result['equity'] / result['benchmark_equity']
    rel = rel / rel.iloc[0]
    ax.plot(rel.index, rel.values, color='teal', lw=2)
    ax.axhline(y=1, color='gray', ls='--', lw=0.8)
    ax.set_title("策略净值 / 基准净值 (相对强度)", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor='#f0f0f0')
    plt.close()
    print(f"\n  回测报告图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Quant Engine - 三市场无泄漏因子选股框架')
    parser.add_argument('--market', type=str, default='CN', choices=['CN', 'US', 'HK'],
                       help='市场: CN/A股 US/美股 HK/港股')
    parser.add_argument('--train', type=int, default=None, help='训练窗口天数')
    parser.add_argument('--test', type=int, default=60, help='测试窗口天数')
    parser.add_argument('--step', type=int, default=60, help='滚动步长')
    parser.add_argument('--rebalance', type=int, default=20, help='再平衡天数')
    parser.add_argument('--sweep', action='store_true', help='参数网格搜索模式')
    parser.add_argument('--live', action='store_true', help='实盘模式：只输出最新持仓')
    parser.add_argument('--top-n', dest='top_n', type=int, default=20,
                        help='实盘持仓数（默认20向后兼容；1w本金建议用 sizing_sweep 选出的5-8）')
    parser.add_argument('--capital', type=float, default=None,
                        help='实盘本金（默认按市场：US=1w美元，CN=6w，HK=9w港币；用于整手/碎股可行性提示）')
    parser.add_argument('--vol-target', dest='vol_target', type=float, default=None,
                        help='波动率目标年化（如0.15）；开启则输出当前建议敞口(高波动降仓留现金)')
    parser.add_argument('--feishu', action='store_true', help='飞书推送持仓报告')
    parser.add_argument('--out', type=str, default=None, help='结果输出目录')

    args = parser.parse_args()

    default_train = {'CN': 240, 'US': 480, 'HK': 120}
    train_days = args.train or default_train[args.market]
    # 本金默认按市场：美股1w美元，A股6w，港股9w（消除整手买不进约束）
    if args.capital is None:
        args.capital = {'US': 10_000.0, 'CN': 60_000.0, 'HK': 90_000.0}[args.market]
    # vol-target 默认按市场调优值（全样本目标波动扫描）：传 0 显式关闭
    if args.vol_target is None:
        args.vol_target = {'US': 0.20, 'CN': 0.25, 'HK': 0.30}[args.market]
    elif args.vol_target <= 0:
        args.vol_target = None

    # ========== 实盘模式 ==========
    if args.live:
        print(f"\n{'='*80}")
        print(f"  {args.market} 实盘持仓生成")
        print(f"{'='*80}")
        port = generate_live_portfolio(args.market, train_days,
                                       top_n=args.top_n, capital=args.capital,
                                       vol_target=args.vol_target)
        ccy = {'CN': '¥', 'HK': 'HK$', 'US': '$'}.get(args.market, '$')
        unit = "股(碎股)" if port['allow_fractional'] else f"股(整手{port['lot_size']})"
        print(f"\n  调仓日期: {port['date']}")
        print(f"  训练窗口: {port['train_days']}天 | 本金: {ccy}{port['capital']:,.0f} | "
              f"持仓数: {port['top_n']} | 最小单位: {unit}")
        print(f"\n  因子方向学习结果:")
        for n, d in port['factor_directions'].items():
            ic = port['factor_ics'][n]
            print(f"    {n:15s}: {d} (IC={ic:+.4f})")
        print(f"\n  最新持仓 (TOP{port['top_n']}等权, 含可行性校验):")
        print(f"    {'#':>3s} {'代码':10s} {'权重':>6s} {'最新价':>10s} {'预算':>9s} "
              f"{'可买股':>7s} {'实占资金':>9s}")
        for i, (sym, w) in enumerate(port['weights'].items()):
            px = port['prices'].get(sym, float('nan'))
            n_sh = port['shares'].get(sym, 0)
            budget = port['capital'] * w
            has_price = px is not None and px == px
            actual = n_sh * px if has_price else 0.0
            px_text = f"{px:.2f}" if has_price else "-"
            warn = f"  ⚠️一手买不起" if sym in port['unaffordable'] else ""
            print(f"    [{i+1:2d}] {sym:10s} {w:>5.1%} {px_text:>10s} {budget:>9.0f} "
                  f"{n_sh:>7d} {actual:>9.0f}{warn}")
        if port['unaffordable']:
            lot_hint = "1股" if port['allow_fractional'] else f"一手({port['lot_size']}股)"
            print(f"\n  ⚠️ {len(port['unaffordable'])} 只票 {lot_hint} 就超预算买不进"
                  f"({ccy}{port['capital']:,.0f}本金/{port['top_n']}只): "
                  f"{', '.join(port['unaffordable'])}")
            print(f"     → 需提高本金、减少持仓数(每只预算变大)，或在选股时剔除这些高价股。")

        # 波动率目标建议敞口（高波动降仓留现金）
        if port.get('suggested_exposure') is not None:
            exp = port['suggested_exposure']
            cash = 1.0 - exp
            print(f"\n  📊 波动率目标 {port['vol_target']:.0%}：当前组合近{20}日年化波动 "
                  f"{port['recent_vol']:.1%} → 建议仓位 {exp:.0%}（留 {cash:.0%} 现金）")
            if exp < 0.999:
                print(f"     → 各票股数按 {exp:.0%} 敞口缩放后再下单；波动回落时逐步加回满仓。")
            else:
                print(f"     → 当前波动低于目标，可满仓（不加杠杆）。")

        out_dir = Path(args.out) if args.out else PROJECT_ROOT / 'results'
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{args.market}_portfolio_{port['date']}.json"
        with open(out_file, 'w') as f:
            json.dump({
                'date': port['date'],
                'market': port['market'],
                'train_days': port['train_days'],
                'top_n': port['top_n'],
                'capital': port['capital'],
                'lot_size': port['lot_size'],
                'factor_directions': port['factor_directions'],
                'factor_ics': port['factor_ics'],
                'weights': {k: float(v) for k, v in port['weights'].items()},
                'shares': {k: int(v) for k, v in port['shares'].items()},
                'lots': {k: int(v) for k, v in port.get('lots', {}).items()},
                'prices': {k: (float(v) if v is not None else None) for k, v in port['prices'].items()},
                'unaffordable': port['unaffordable'],
                'vol_target': port.get('vol_target'),
                'recent_vol': port.get('recent_vol'),
                'suggested_exposure': port.get('suggested_exposure'),
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  持仓已保存: {out_file}")

        # 飞书推送
        if args.feishu:
            from feishu_notify import send_portfolio_report
            send_portfolio_report(args.market, port)
        return

    # ========== 参数搜索模式 ==========
    if args.sweep:
        df = parameter_sweep(args.market)
        out_dir = Path(args.out) if args.out else PROJECT_ROOT / 'results'
        out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / f"{args.market}_sweep_results.csv", index=False)
        best = df.iloc[0]
        print(f"\n  推荐参数: train={int(best['train'])}d test={int(best['test'])}d step={int(best['step'])}d")
        return

    # ========== 回测验证模式 ==========
    print(f"\n{'='*80}")
    print(f"  {args.market} Walk-Forward 无泄漏回测验证")
    print(f"  配置: train={train_days}d test={args.test}d step={args.step}d rebalance={args.rebalance}d")
    print(f"{'='*80}")

    loader = {'CN': QuantMarket.load_cn, 'US': QuantMarket.load_us, 'HK': QuantMarket.load_hk}[args.market]
    close, factors = loader()
    print(f"\n  股票数: {len(close.columns)} 交易日数: {len(close)}")
    print(f"  因子数: {len(factors)} ({', '.join(factors.keys())})")
    print(f"\n  窗口明细:")

    result = walk_forward_backtest(
        close, factors,
        train_days=train_days, test_days=args.test, step_days=args.step,
        rebalance_days=args.rebalance, verbose=True
    )

    print(f"\n{'='*80}")
    print(f"  全期拼接结果")
    print(f"{'='*80}")
    s = result['summary']
    sb = result['benchmark_summary']
    print(f"\n  策略:")
    print(f"    累计收益: {s['total_return']:>+8.1%}  年化: {s['annualized_return']:>+7.1%}")
    print(f"    夏普比率: {s['sharpe']:>+8.2f}  最大回撤: {s['max_drawdown']:>+7.1%}")
    print(f"\n  基准:")
    print(f"    累计收益: {sb['total_return']:>+8.1%}  年化: {sb['annualized_return']:>+7.1%}")
    print(f"    夏普比率: {sb['sharpe']:>+8.2f}  最大回撤: {sb['max_drawdown']:>+7.1%}")
    print(f"\n  超额:")
    print(f"    超额夏普: {result['excess_sharpe']:+.2f}")
    print(f"    窗口胜率: {result['win_rate']:.1%} ({sum(1 for w in result['windows'] if w['excess']>0)}/{result['n_windows']})")

    if result['excess_sharpe'] > 0:
        print(f"\n  ✅ {args.market} 策略稳定跑赢基准，可以进入模拟盘验证！")
    else:
        print(f"\n  ⚠️  {args.market} 策略未跑赢基准，建议调整因子池或参数")

    # 保存结果
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / 'results'
    out_dir.mkdir(exist_ok=True)
    params_str = f"{args.market}_t{train_days}_s{args.step}_r{args.rebalance}"
    plot_result(result, args.market, params_str, out_dir / f"{params_str}_report.png")

    # 保存JSON结果
    with open(out_dir / f"{params_str}_result.json", 'w') as f:
        json.dump({
            'market': args.market,
            'params': {'train': train_days, 'test': args.test, 'step': args.step, 'rebalance': args.rebalance},
            'strategy': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in s.items()},
            'benchmark': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in sb.items()},
            'excess_sharpe': float(result['excess_sharpe']),
            'win_rate': float(result['win_rate']),
            'n_windows': result['n_windows'],
            'windows': [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in w.items() if k != 'directions'}
                       for w in result['windows']],
        }, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {out_dir / f'{params_str}_result.json'}")


if __name__ == '__main__':
    main()
