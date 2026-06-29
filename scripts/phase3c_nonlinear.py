"""phase3c_nonlinear —— 非线性因子合成（Ridge回归 + Walk-Forward，Phase 3c）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/phase3c_nonlinear.py

逻辑：
    ① 15个因子的截面数据，按日期拉平成 (N×D) 面板
    ② Walk-forward：每 120 个交易日重训练一次 Ridge 回归
       - 训练数据：ALL past data up to train_cutoff
       - 特征：15原始 + 15平方 = 30维（省略交互项，避免105维过拟合）
       - 目标：forward 20-day return
    ③ 用训练好的 beta 预测 train_cutoff 之后所有日期的因子合成得分
    ④ 与线性等权对比分层多头绩效

防前视铁律：每期只用 train_cutoff 之前的数据训练，之后的数据从未被模型看到。
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

from quant.config import RAW_DATA_DIR
from quant.data import us_loader
from quant.data.universe_us_expanded import EXPANDED_US_POOL
from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
from quant.data.industry_us import industry_series
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize as neut_fn
from quant.backtest.ic_analysis import forward_returns
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
US_COST_FN = make_layered_cost_fn()
RIDGE_ALPHA = 1.0
RETRAIN_EVERY = 120


def build_all_15_factors(close, fund, panels, ind):
    """Phase 3b 确认的 15 因子（行业中性化）。"""
    amount = panels["amount"]
    high, low = panels["high"], panels["low"]
    raw = {}
    raw["quality_roe"] = F.us_quality_roe(fund["roe"])
    raw["quality_gm"] = F.us_quality_roe(fund["gross_margin"])
    raw["growth_rev"] = F.us_growth(fund["rev_yoy"])
    raw["pv_momentum60"] = -F.momentum(close, 60)
    raw["pv_reversal20"] = F.reversal(close, 20)
    raw["pv_lowvol20"] = -F.low_volatility(close, 20)
    raw["pv_amihud"] = -F.amihud_illiquidity(close, amount, 20)
    raw["pv_parkinson"] = -F.parkinson_volatility(high, low, 20)
    raw["pv_reversal5"] = F.reversal(close, 5)
    raw["pv_lowvol5"] = -F.low_volatility(close, 5)
    raw["pv_amihud5"] = -F.amihud_illiquidity(close, amount, 5)
    raw["pv_maslope60"] = -F.ma_slope(close, 60)
    raw["pv_lowvol60"] = -F.low_volatility(close, 60)
    raw["gm_yoy_change"] = fund["gross_margin"] - fund["gross_margin"].shift(252)
    raw["rev_growth_accel"] = fund["rev_yoy"] - fund["rev_yoy"].shift(252)
    return {n: neut_fn(f, industry=ind, mode="industry") for n, f in raw.items()}


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    """Ridge: β = (X'X + αI)⁻¹X'y"""
    d = X.shape[1]
    try:
        return np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X.T @ X + alpha * np.eye(d), X.T @ y, rcond=None)[0]


def _perf(ret, label):
    eq = (1.0 + ret).cumprod()
    m = summary(eq, ret)
    return f"{label:30s} {m['total_return']:>+8.1%} {m['sharpe']:>+6.2f} {m['max_drawdown']:>+7.1%}"


def main():
    print("=" * 72)
    print("  Phase 3c: Ridge多项式非线性因子合成")
    print("=" * 72)

    print("\n[1] 构建面板 + 15因子...", flush=True)
    panels = build_ohlcv_panels(EXPANDED_US_POOL, loader=us_loader)
    close = panels["close"]
    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    ind = industry_series(list(close.columns))
    fwd = forward_returns(close, horizon=HORIZON)
    facs_15 = build_all_15_factors(close, fund, panels, ind)
    eq_linear = combine_factors(*facs_15.values())

    factor_names = list(facs_15.keys())
    n_factors = len(factor_names)
    print(f"      因子: {n_factors} | 特征: {n_factors}原始 + {n_factors}平方 = {n_factors*2}维")

    # ── 拉平成面板 ──
    print("\n[2] 拉平截面数据为训练面板...", flush=True)
    fac_stack = {}
    for name in factor_names:
        s = facs_15[name].stack()
        s.name = name
        fac_stack[name] = s
    X_df = pd.DataFrame(fac_stack)  # (date, stock) MultiIndex
    y_s = fwd.stack()
    y_s.name = "fwd_ret"

    # 合并并对齐
    full = X_df.join(y_s, how="inner").dropna()
    dates = sorted(set(full.index.get_level_values("date")))
    stocks = list(facs_15[factor_names[0]].columns)
    print(f"      有效训练样本: {len(full):,} (日期×股票) | 日期: {len(dates)} | 股票: {len(stocks)}")

    # ── Walk-forward 训练 + 预测 ──
    print(f"\n[3] Walk-forward Ridge（retrain每{RETRAIN_EVERY}天）...", flush=True)
    composite_ml = pd.DataFrame(np.nan, index=close.index, columns=stocks)

    train_cutoffs = dates[::RETRAIN_EVERY]  # 每 RETRAIN_EVERY 天重训练一次
    n_models = 0
    for cutoff in train_cutoffs:
        # 训练数据：所有 ≤ cutoff 的样本
        train_mask = full.index.get_level_values("date") <= cutoff
        train = full[train_mask]
        if len(train) < 10000:
            continue

        X_raw = train[factor_names].values
        y_raw = train["fwd_ret"].values

        # 截面标准化（每日期内做，但全局用均值/std近似）
        x_mean = np.nanmean(X_raw, axis=0)
        x_std = np.nanstd(X_raw, axis=0) + 1e-12
        X_std = (X_raw - x_mean) / x_std
        # 多项式特征：原始 + 平方
        X_poly = np.column_stack([X_std, X_std ** 2])

        beta = ridge_fit(X_poly, y_raw, alpha=RIDGE_ALPHA)
        n_models += 1

        # 预测：cutoff 之后的所有日期（直到下一个 cutoff）
        future_dates = [d for d in dates if d > cutoff]
        for pred_date in future_dates:
            # 获取该日所有股票的因子值
            try:
                row = full.loc[pred_date]
            except KeyError:
                continue
            if isinstance(row, pd.Series):
                row = row.to_frame().T
            X_pred_raw = row[factor_names].values
            valid = ~np.isnan(X_pred_raw).any(axis=1)
            if valid.sum() < 10:
                continue

            X_pred_std = (X_pred_raw - x_mean) / x_std
            X_pred_poly = np.column_stack([X_pred_std, X_pred_std ** 2])
            scores = X_pred_poly @ beta

            # 填入对应股票
            stock_list = row.index.tolist() if hasattr(row.index, 'tolist') else [row.name]
            for j, s in enumerate(stock_list):
                if j < len(scores) and valid[j] and s in composite_ml.columns:
                    composite_ml.loc[pred_date, s] = scores[j]

    valid_dates_ml = composite_ml.dropna(how="all").index
    print(f"      训练了 {n_models} 个模型 | 有效预测日期: {len(valid_dates_ml)}")

    if len(valid_dates_ml) < 100:
        print("  ⚠️ 预测不足，终止。")
        return

    # 对齐回测起点
    start_d = valid_dates_ml[0]
    close_sub = close.loc[start_d:]
    eq_linear_sub = eq_linear.loc[start_d:]
    ml_sub = composite_ml.loc[start_d:]

    print(f"\n[4] 对比回测（从{start_d.date()}起）...\n", flush=True)
    bt_linear = long_top_layer(close_sub, eq_linear_sub, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt_ml = long_top_layer(close_sub, ml_sub, rebalance_every=REBALANCE, cost_fn=US_COST_FN)

    print(f"      {'方案':30s} {'累计收益':>8s} {'夏普':>6s} {'回撤':>7s}")
    print(f"      {'-'*55}")
    print(f"      {_perf(bt_linear['port_ret'], '线性等权 15F')}")
    print(f"      {_perf(bt_ml['port_ret'], f'Ridge多项式 15F')}")
    print(f"      {_perf(bt_linear['benchmark_ret'], '等权基准')}")

    # 图
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt_linear["equity"].index, bt_linear["equity"], label="Linear Equal-Weight 15F", lw=2.0, color="steelblue")
    ax.plot(bt_ml["equity"].index, bt_ml["equity"], label="Ridge Polynomial 15F", lw=2.0, color="darkorange")
    ax.plot(bt_linear["benchmark"].index, bt_linear["benchmark"], label="Benchmark", lw=1.0, ls="--", color="gray")
    ax.set_title(f"Phase 3c: Linear vs Ridge-Polynomial (15F × 30dim, retrain/{RETRAIN_EVERY}d)")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = RAW_DATA_DIR / "phase3c_nonlinear.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n      图: {png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
