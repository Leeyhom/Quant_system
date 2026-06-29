"""phase6_ensemble —— 时间序列因子 + XGBoost多horizon Ensemble（Phase 6）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/phase6_ensemble.py [--neutralize]

新增武器：
    ① 时间序列派生因子（从OHLCV+基本面零成本导出）：
       - vol_regime: 短/长期波动率比率（捕捉波动率扩张/收缩周期）
       - trend_strength: 60日对数价格的趋势R²
       - factor_momentum_roe: ROE因子的20日变化
       - factor_momentum_gm: 毛利率因子的20日变化
       - cross_sectional_disp: 截面因子离散度（市场整体信号）

    ② XGBoost多horizon Ensemble：
       - 3个模型分别预测 horizon=5d, 20d, 60d 的forward return
       - 预测取平均（ensemble mean）
       - 每120天 retrain，walk-forward严格防前视
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
import xgboost as xgb

from quant.config import RAW_DATA_DIR
from quant.data import us_loader
from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize as neut_fn
from quant.backtest.ic_analysis import forward_returns
from quant.backtest.layered import long_top_layer
from quant.backtest.metrics import summary
from quant.backtest.us_cost import make_layered_cost_fn

HORIZONS = [5, 20, 60]
REBALANCE = 20
N_LAYERS = 5
US_COST_FN = make_layered_cost_fn()
RETRAIN_EVERY = 120


def load_dynamic_tickers():
    with open(PROJECT_ROOT / "data" / "raw" / "us_dynamic_tickers.txt") as f:
        return [l.strip() for l in f if l.strip()]


def build_base_factors(close, fund, panels):
    """15基础因子（与Phase 5一致）。"""
    amount, high, low = panels["amount"], panels["high"], panels["low"]
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
    return raw


def add_ts_factors(close, panels, facs):
    """新增时间序列派生因子（从已有数据零成本导出）。

    这些因子捕捉的是「市场状态」和「因子动量」，与截面因子正交。
    """
    ret = close.pct_change()
    new = {}

    # 1. 波动率区间：短期/长期波动率比值（>1=波动扩张，<1=波动收缩）
    vol_short = ret.rolling(20).std()
    vol_long = ret.rolling(60).std().replace(0, np.nan)
    new["vol_regime"] = vol_short / vol_long

    # 2. 趋势强度：60日对数价格的线性回归 R²（趋势越明确的股票越可靠）
    log_p = np.log(close.replace(0, np.nan))
    def trend_r2(series, window=60):
        result = pd.Series(np.nan, index=series.index)
        x = np.arange(window)
        for i in range(window, len(series)):
            y = series.iloc[i-window:i].values
            mask = ~np.isnan(y)
            if mask.sum() < 30:
                continue
            xm, ym = x[mask].mean(), y[mask].mean()
            ss_tot = ((y[mask] - ym) ** 2).sum()
            if ss_tot < 1e-12:
                continue
            slope = ((x[mask]-xm)*(y[mask]-ym)).sum() / ((x[mask]-xm)**2).sum()
            y_pred = xm + slope * (x[mask] - xm)
            ss_res = ((y[mask] - y_pred) ** 2).sum()
            result.iloc[i] = 1 - ss_res / ss_tot
        return result
    new["trend_strength"] = log_p.apply(trend_r2)

    # 3. 因子动量：基础因子的20日变化
    if "quality_roe" in facs:
        new["factor_mom_roe"] = facs["quality_roe"].diff(20)
    if "quality_gm" in facs:
        new["factor_mom_gm"] = facs["quality_gm"].diff(20)

    # 4. 截面离散度：每日所有股票因子得分的标准差（市场整体信号）
    cs_disp = pd.Series(np.nan, index=close.index)
    for i, date in enumerate(close.index):
        vals = []
        for f in facs.values():
            if date in f.index:
                vals.append(f.loc[date].dropna().values)
        if vals:
            cs_disp.loc[date] = np.mean([np.std(v) for v in vals])
    disp_df = pd.DataFrame({c: cs_disp for c in close.columns}, index=close.index)
    new["cross_section_disp"] = disp_df

    # 合并
    all_facs = {**facs, **{k: v for k, v in new.items() if not v.isna().all().all()}}
    print(f"    时间序列因子: {len(new)} 个新增 (总{len(all_facs)}个)")
    return all_facs


def build_xgb_ensemble(facs, close):
    """多horizon XGBoost ensemble：分别对5d/20d/60d训练模型，预测取平均。"""
    factor_names = list(facs.keys())
    stocks = close.columns.tolist()

    # 拉平面板
    fac_stack = {n: facs[n].stack() for n in factor_names}
    X_df = pd.DataFrame(fac_stack)

    # 对每个 horizon 训练
    all_preds = []
    for horizon in HORIZONS:
        fwd = forward_returns(close, horizon=horizon)
        y_s = fwd.stack(); y_s.name = "target"
        full = X_df.join(y_s, how="inner").dropna()
        full_dates = sorted(set(full.index.get_level_values("date")))

        pred_df = pd.DataFrame(np.nan, index=close.index, columns=stocks)
        train_cutoffs = full_dates[::RETRAIN_EVERY]
        for cutoff in train_cutoffs:
            train = full[full.index.get_level_values("date") <= cutoff]
            if len(train) < 20000:
                continue
            X_train, y_train = train[factor_names].values, train["target"].values
            model = xgb.XGBRegressor(
                n_estimators=120, max_depth=5, learning_rate=0.05,
                subsample=0.8, reg_alpha=1.0, reg_lambda=1.5,
                random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            future = [d for d in full_dates if d > cutoff]
            for pred_date in future:
                try:
                    row = full.loc[pred_date]
                except KeyError:
                    continue
                X_pred = row[factor_names].values
                preds = model.predict(X_pred)
                for j, s in enumerate(row.index):
                    if s in pred_df.columns:
                        pred_df.loc[pred_date, s] = float(preds[j])
        all_preds.append(pred_df.rank(axis=1, pct=True))

    # Ensemble: 简单平均
    ensemble = sum(all_preds) / len(all_preds)
    valid = ensemble.dropna(how="all").index
    print(f"    Ensemble: {len(valid)}有效预测日 (horizons={HORIZONS})")
    return ensemble


def apply_neutralize(facs, close):
    """市值中性化。"""
    import pickle
    spot_df = pd.read_pickle(PROJECT_ROOT / "data" / "raw" / "us_spot_list.pkl")
    spot_df["mktcap"] = pd.to_numeric(spot_df["mktcap"], errors="coerce")
    spot_df["symbol_clean"] = spot_df["symbol"].str.upper().str.replace(".", "-", regex=False)
    mktcap_map = dict(zip(spot_df["symbol_clean"], spot_df["mktcap"]))
    log_mv_data = {}
    for s in close.columns:
        mc = mktcap_map.get(s, np.nan)
        log_mv_data[s] = np.log(max(mc, 1e6)) if mc > 0 else np.nan
    log_mv = pd.DataFrame([log_mv_data], index=close.index[:1]).reindex(close.index).ffill()
    return {n: neut_fn(f, log_mv=log_mv, mode="size") for n, f in facs.items()}


def filter_pool(close, panels):
    """质量过滤 + 极端收益剔除。"""
    vol = panels["volume"]
    avg_vol = vol.rolling(60).mean().iloc[-1]
    ok = (close.iloc[-1] >= 5.0) & (avg_vol >= 300000)
    ret = close.pct_change()
    ok &= (ret.abs() > 5.0).sum() <= 10
    ok &= (ret < -0.8).sum() <= 5
    keep = ok[ok].index.tolist()
    for k in ["close", "amount", "high", "low", "volume"]:
        panels[k] = panels[k][keep]
    close_new = panels["close"]
    for k in panels:
        panels[k] = panels[k].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    print(f"    质量过滤: {len(keep)}/{close.shape[1]}")
    return close_new


def _stats(bt):
    s = summary(bt["equity"], bt["port_ret"])
    return {"return": s["total_return"], "sharpe": s["sharpe"],
            "dd": s["max_drawdown"], "ann": s["annualized_return"]}


def main():
    print("=" * 72)
    print("  Phase 6: 时间序列因子 + XGBoost多Horizon Ensemble")
    print("=" * 72)

    do_neut = "--neutralize" in sys.argv

    tik = load_dynamic_tickers()
    panels = build_ohlcv_panels(tik, loader=us_loader)
    close = panels["close"]
    close = filter_pool(close, panels)
    print(f"    有效池: {close.shape[1]} 只")

    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    print(f"    基本面: {fund['roe'].notna().any().sum()} 只")

    # 因子
    base = build_base_factors(close, fund, panels)
    facs = add_ts_factors(close, panels, base)
    if do_neut:
        facs = apply_neutralize(facs, close)
        print("    ✅ 市值中性化")

    eq_linear = combine_factors(*facs.values())

    # 回测基线
    bt_ew = long_top_layer(close, eq_linear, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt_rp = long_top_layer(close, eq_linear, rebalance_every=REBALANCE, cost_fn=US_COST_FN, weight_mode="risk_parity")

    # XGBoost ensemble
    print(f"\n[2] XGBoost {len(HORIZONS)}-horizon Ensemble...")
    ensemble = build_xgb_ensemble(facs, close)
    start_d = ensemble.dropna(how="all").index[0]
    close_sub = close.loc[start_d:]
    ens_sub = ensemble.loc[start_d:]
    bt_xgb = long_top_layer(close_sub, ens_sub, rebalance_every=REBALANCE, cost_fn=US_COST_FN, weight_mode="risk_parity")

    print(f"\n{'方案':35s} {'累计':>8s} {'年化':>7s} {'夏普':>6s} {'回撤':>7s}")
    print(f"{'-'*70}")
    for label, bt in [("① 等权L5+等权因子", bt_ew),
                       ("② 风险平价L5+等权因子", bt_rp),
                       ("③ RP+XGBoost Ensemble", bt_xgb)]:
        s = _stats(bt)
        print(f"{label:35s} {s['return']:>+8.1%} {s['ann']:>+7.1%} {s['sharpe']:>+6.2f} {s['dd']:>+7.1%}")

    s_b = summary(bt_ew["benchmark"], bt_ew["benchmark_ret"])
    print(f"{'📊 等权基准':35s} {s_b['total_return']:>+8.1%} {s_b['annualized_return']:>+7.1%} {s_b['sharpe']:>+6.2f} {s_b['max_drawdown']:>+7.1%}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt_ew["equity"].index, bt_ew["equity"], label="EW L5", lw=1.5, color="steelblue")
    ax.plot(bt_rp["equity"].index, bt_rp["equity"], label="RP L5", lw=1.5, color="darkorange")
    ax.plot(bt_xgb["equity"].index, bt_xgb["equity"], label="RP+XGB Ensemble", lw=2.0, color="red")
    ax.plot(bt_ew["benchmark"].index, bt_ew["benchmark"], label="Bench", lw=1.0, ls="--", color="gray")
    ax.set_title(f"Phase 6: TS Factors + XGBoost Ensemble ({close.shape[1]} stocks, {len(facs)} factors)")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = RAW_DATA_DIR / "phase6_ensemble.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n图: {png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
