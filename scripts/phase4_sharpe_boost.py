"""phase4_sharpe_boost —— 夏普提升：风险平价 + 动态因子权重 + 仓位缩放（Phase 4）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/phase4_sharpe_boost.py

目标：把夏普从 1.04 推到 1.5+。

武器库：
    ① 风险平价（Risk Parity）：top quintile 内按等风险贡献配权，替代等权。
       理论预期夏普提升 +0.2~0.4。
    ② 动态因子权重：用60日滚动IC加权因子（短窗口=响应快=避开方向锁定陷阱）。
    ③ 自适应仓位缩放：根据近期波动率调整整体仓位，目标年化波动率 25%
       （比15%更接近美股实际，15%太保守把收益也杀了）。

对比：等权L5基线 vs 风险平价 vs 风险平价+动态因子 vs 全武器。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant.config import RAW_DATA_DIR
from quant.data import us_loader
from quant.data.universe_us_expanded import EXPANDED_US_POOL
from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
from quant.data.industry_us import industry_series
from quant.factor import factors as F
from quant.factor.neutralize import neutralize as neut_fn
from quant.factor.factors import combine_factors
from quant.factor.composite import factor_correlation
from quant.backtest.ic_analysis import forward_returns, daily_ic
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

REBALANCE = 20
US_COST_FN = make_layered_cost_fn()
TOP_FRAC = 0.20
VOL_TARGET = 0.25  # 目标年化波动率 25%（比15%更符合美股实际）
VOL_LOOKBACK = 60


def build_15_factors(close, fund, panels, ind):
    """Phase 3b 的 15 因子（行业中性化+方向定向）。"""
    amount = panels["amount"]; high, low = panels["high"], panels["low"]
    raw = {}
    # 基本面 3
    raw["quality_roe"] = F.us_quality_roe(fund["roe"])
    raw["quality_gm"] = F.us_quality_roe(fund["gross_margin"])
    raw["growth_rev"] = F.us_growth(fund["rev_yoy"])
    # 量价 5（方向已定向）
    raw["pv_momentum60"] = -F.momentum(close, 60)
    raw["pv_reversal20"] = F.reversal(close, 20)
    raw["pv_lowvol20"] = -F.low_volatility(close, 20)
    raw["pv_amihud"] = -F.amihud_illiquidity(close, amount, 20)
    raw["pv_parkinson"] = -F.parkinson_volatility(high, low, 20)
    # 多周期 5
    raw["pv_reversal5"] = F.reversal(close, 5)
    raw["pv_lowvol5"] = -F.low_volatility(close, 5)
    raw["pv_amihud5"] = -F.amihud_illiquidity(close, amount, 5)
    raw["pv_maslope60"] = -F.ma_slope(close, 60)
    raw["pv_lowvol60"] = -F.low_volatility(close, 60)
    # 变动 2
    raw["gm_yoy_change"] = fund["gross_margin"] - fund["gross_margin"].shift(252)
    raw["rev_growth_accel"] = fund["rev_yoy"] - fund["rev_yoy"].shift(252)
    return {n: neut_fn(f, industry=ind, mode="industry") for n, f in raw.items()}


def risk_parity_weights(cov: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """风险平价权重：每只股票的边际风险贡献相等。

    最小化 Σ_i Σ_j (RC_i - RC_j)²，其中 RC_i = w_i * (Σw)_i / √(w'Σw)。
    约束：w_i ≥ 0, Σw_i = 1。

    若协方差矩阵奇异，退回等权。
    """
    n = len(cov)
    if n <= 1:
        return np.ones(n) / n

    def risk_contributions(w):
        sigma_w = cov @ w
        port_vol = np.sqrt(w @ sigma_w)
        if port_vol < 1e-12:
            return np.ones(n) / n
        return w * sigma_w / port_vol

    def objective(w):
        rc = risk_contributions(w)
        target = 1.0 / n
        return np.sum((rc - target) ** 2)

    # 初始值：等权
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    try:
        result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": max_iter, "ftol": 1e-10})
        if result.success:
            w = result.x
            w = np.maximum(w, 0)
            return w / w.sum()
    except Exception:
        pass
    return np.ones(n) / n


def run_backtest(close, eq_factor, method="equal", dynamic_weights=None,
                 vol_target=None):
    """通用回测：等权 or 风险平价，可选动态因子权重和仓位缩放。

    返回 DataFrame（port_ret, equity, benchmark_ret, benchmark）。
    """
    ret = close.pct_change().fillna(0.0).values
    dates = close.index
    stocks = close.columns
    n_dates = len(dates)
    n_stocks = len(stocks)

    port_ret = np.zeros(n_dates)
    turnover_arr = np.zeros(n_dates)
    current_w = np.zeros(n_stocks)

    # 预计算动态因子权重（若提供）
    dyn_factor_weights = None
    if dynamic_weights is not None:
        dyn_factor_weights = dynamic_weights

    for i in range(n_dates):
        should_rebalance = i > 0 and (i % REBALANCE == 0)

        if should_rebalance:
            # 获取当日因子分数
            date = dates[i]
            scores = eq_factor.loc[date].values.copy()
            valid_mask = ~np.isnan(scores)
            n_valid = valid_mask.sum()

            if n_valid < 10:
                current_w = np.zeros(n_stocks)
            else:
                # 选 top_frac 最高分
                n_select = max(1, int(n_valid * TOP_FRAC))
                valid_indices = np.where(valid_mask)[0]
                top_idx = valid_indices[np.argsort(scores[valid_mask])[-n_select:]]

                if method == "equal":
                    w = np.zeros(n_stocks)
                    w[top_idx] = 1.0 / n_select
                elif method == "risk_parity":
                    # 用过去60天收益估算协方差
                    start_idx = max(0, i - 60)
                    ret_window = ret[start_idx:i, :][:, top_idx]
                    ret_window = ret_window[~np.isnan(ret_window).any(axis=1)]
                    if len(ret_window) < 20:
                        w = np.zeros(n_stocks)
                        w[top_idx] = 1.0 / n_select
                    else:
                        cov = np.cov(ret_window, rowvar=False)
                        # 正则化
                        cov = cov + np.eye(len(cov)) * 1e-6
                        rp_w = risk_parity_weights(cov)
                        w = np.zeros(n_stocks)
                        for j, idx in enumerate(top_idx):
                            w[idx] = rp_w[j]

                # 计算换手和费用
                weight_change = np.abs(w - current_w)
                turnover_arr[i] = weight_change.sum()
                prices = close.iloc[i].values
                if turnover_arr[i] > 0:
                    notional = weight_change * 1_000_000
                    shares = notional / np.maximum(prices, 1e-6)
                    shares = np.nan_to_num(shares, 0)
                    traded = shares[shares > 0]
                    if len(traded) > 0:
                        cost_dollars = np.sum(np.maximum(traded * 0.005, 1.0))
                        cost_frac = cost_dollars / 1_000_000
                    else:
                        cost_frac = 0.0
                else:
                    cost_frac = 0.0

                current_w = w
        else:
            cost_frac = 0.0
            turnover_arr[i] = 0.0

        gross = np.dot(current_w, ret[i])
        port_ret[i] = gross - cost_frac

    # 仓位缩放
    if vol_target is not None:
        port_ret_series = pd.Series(port_ret, index=dates)
        rolling_vol = port_ret_series.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
        scale = (vol_target / rolling_vol.replace(0, np.nan)).clip(upper=2.0, lower=0.1)
        scale = scale.fillna(1.0).values
        port_ret = port_ret * np.roll(scale, 1)
        port_ret[0] = port_ret[0]  # 第一天不动

    # 基准
    available = close.notna().astype(float).values
    bench_w = available / available.sum(axis=1, keepdims=True)
    bench_ret = np.sum(bench_w * ret, axis=1)

    out = pd.DataFrame(index=dates)
    out["port_ret"] = port_ret
    out["equity"] = (1.0 + port_ret).cumprod()
    out["benchmark_ret"] = bench_ret
    out["benchmark"] = (1.0 + bench_ret).cumprod()
    return out


def _stats(bt):
    return summary(bt["equity"], bt["port_ret"])


def main():
    print("=" * 72)
    print("  Phase 4: 夏普提升实验（风险平价 + 动态因子 + 仓位缩放）")
    print("=" * 72)

    print("\n[1] 构建面板 + 15因子...", flush=True)
    panels = build_ohlcv_panels(EXPANDED_US_POOL, loader=us_loader)
    close = panels["close"]
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    ind = industry_series(list(close.columns))
    facs = build_15_factors(close, fund, panels, ind)

    # 静态等权合成因子
    eq_15f = combine_factors(*facs.values())

    # 动态因子权重：60日滚动IC
    print("\n[2] 计算动态因子权重（60日滚动IC）...", flush=True)
    fwd = forward_returns(close, horizon=20)
    dynamic_factor_weights = pd.DataFrame(1.0 / len(facs), index=close.index, columns=list(facs.keys()))
    for i in range(120, len(close.index)):
        date = close.index[i]
        start_date = close.index[max(0, i - 60)]
        ic_window = {}
        for name, fac in facs.items():
            fac_slice = fac.loc[start_date:date]
            fwd_slice = fwd.loc[start_date:date]
            ic_series = daily_ic(fac_slice, fwd_slice)
            if len(ic_series) > 10:
                ic_window[name] = abs(ic_series.mean())
            else:
                ic_window[name] = 0
        total_ic = sum(ic_window.values())
        if total_ic > 0:
            for name in ic_window:
                dynamic_factor_weights.loc[date, name] = ic_window[name] / total_ic

    # 每天用动态权重合成因子
    print("      构建动态加权合成因子...", flush=True)
    eq_dynamic = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for date in close.index:
        w = dynamic_factor_weights.loc[date]
        for name, fac in facs.items():
            if date in fac.index:
                eq_dynamic.loc[date] += w[name] * fac.loc[date].fillna(0)

    # 标准化
    eq_dynamic = eq_dynamic.rank(axis=1, pct=True)

    print("\n[3] 四种方案对比...\n", flush=True)
    results = {}

    bt = run_backtest(close, eq_15f, method="equal")
    results["① 等权L5 + 等权因子"] = _stats(bt)

    bt = run_backtest(close, eq_15f, method="risk_parity")
    results["② 风险平价L5 + 等权因子"] = _stats(bt)

    bt = run_backtest(close, eq_dynamic, method="risk_parity")
    results["③ 风险平价L5 + 动态因子(60dIC)"] = _stats(bt)

    bt = run_backtest(close, eq_dynamic, method="risk_parity", vol_target=VOL_TARGET)
    results["④ RP + 动态因子 + VolTarget25%"] = _stats(bt)

    print(f"  {'方案':35s} {'累计收益':>8s} {'年化':>7s} {'夏普':>6s} {'回撤':>7s} {'Calmar':>6s}")
    print(f"  {'-'*75}")
    best_sharpe = 0
    best_name = ""
    for name, s in results.items():
        calmar = s["annualized_return"] / max(s["max_drawdown"], 0.001)
        print(f"  {name:35s} {s['total_return']:>+8.1%} {s['annualized_return']:>+7.1%} "
              f"{s['sharpe']:>+6.2f} {s['max_drawdown']:>+7.1%} {calmar:>+6.2f}")
        if s["sharpe"] > best_sharpe:
            best_sharpe = s["sharpe"]
            best_name = name

    # 等权基准
    bench_ret = bt["benchmark_ret"]
    bench_eq = (1.0 + bench_ret).cumprod()
    bench_s = summary(bench_eq, bench_ret)
    calmar_b = bench_s["annualized_return"] / max(bench_s["max_drawdown"], 0.001)
    print(f"  {'📊 等权基准':35s} {bench_s['total_return']:>+8.1%} {bench_s['annualized_return']:>+7.1%} "
          f"{bench_s['sharpe']:>+6.2f} {bench_s['max_drawdown']:>+7.1%} {calmar_b:>+6.2f}")

    print(f"\n  → 最优方案: {best_name}（夏普 {best_sharpe:.2f}）")
    sharpe_gain = best_sharpe - 1.04
    print(f"  → 相对 Phase 3b 基线(夏普1.04)提升: {sharpe_gain:+.2f}")

    # 图
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ["steelblue", "darkorange", "green", "red"]
    for (name, s), c in zip(results.items(), colors):
        bt = run_backtest(close, eq_15f if "等权因子" in name else eq_dynamic,
                          method="risk_parity" if "风险平价" in name else "equal",
                          vol_target=VOL_TARGET if "VolTarget" in name else None)
        ax.plot(bt["equity"].index, bt["equity"], label=f"{name} (SR={s['sharpe']:.2f})", lw=1.5, color=c)
    ax.plot(bench_eq.index, bench_eq, label=f"Benchmark (SR={bench_s['sharpe']:.2f})", lw=1.0, ls="--", color="gray")
    ax.set_title(f"Phase 4: Sharpe Boost — Risk Parity + Dynamic Weights + Vol Targeting")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = RAW_DATA_DIR / "phase4_sharpe_boost.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n  图: {png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
