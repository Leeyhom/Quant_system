"""phase5_xgboost_dynamic —— XGBoost+风险平价：全市场动态池终极验证（Phase 5）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/phase5_xgboost_dynamic.py

目标：在全市场动态池(2073只)上用XGBoost非线性合成+风险平价组合，
      同时提升夏普和收益率。

对比基线：
    ① 等权L5 + 等权因子（Phase 3b复现）
    ② 风险平价L5 + 等权因子
    ③ 风险平价L5 + XGBoost合成因子
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

HORIZON = 20
N_LAYERS = 5
REBALANCE = 20
US_COST_FN = make_layered_cost_fn()
RETRAIN_EVERY = 120  # XGBoost 重训练间隔（交易日）


def load_dynamic_tickers():
    with open(PROJECT_ROOT / "data" / "raw" / "us_dynamic_tickers.txt") as f:
        return [l.strip() for l in f if l.strip()]


def build_all_factors(close, fund, panels):
    """Phase 3b 的 15 因子（原始方向定向，不做行业中性化——动态池行业映射太稀疏）。

    量价因子方向：Phase 3a/3b 在扩展池上已确认 IC 方向，这里复用。
    基本面因子方向：全样本均值 IC 为正时保留原方向。
    """
    amount = panels["amount"]; high, low = panels["high"], panels["low"]
    raw = {}
    # 基本面 3（方向为正，保留原方向）
    raw["quality_roe"] = F.us_quality_roe(fund["roe"])
    raw["quality_gm"] = F.us_quality_roe(fund["gross_margin"])
    raw["growth_rev"] = F.us_growth(fund["rev_yoy"])
    # 量价 5（IC 为负的翻转向）
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
    return raw


def build_xgboost_composite(facs, close, fwd):
    """Walk-forward XGBoost 合成因子。

    对流式：
      - 每 RETRAIN_EVERY 个交易日重新训练一次
      - 训练数据：train_cutoff 日期之前的所有 (日期×股票) 样本
      - 预测：train_cutoff 之后第一个交易日的因子得分
      - 各日独立做截面 rank → 输出 rank 分数

    返回：与 close 同 index 的合成因子 DataFrame。
    """
    factor_names = list(facs.keys())
    dates = sorted(close.index)

    # 拉平为 (date, stock) 面板
    fac_stack = {}
    for name in factor_names:
        s = facs[name].stack()
        s.name = name
        fac_stack[name] = s
    X_df = pd.DataFrame(fac_stack)
    y_s = fwd.stack(); y_s.name = "target"
    full = X_df.join(y_s, how="inner").dropna()
    full_dates = sorted(set(full.index.get_level_values("date")))

    stocks = close.columns.tolist()
    composite = pd.DataFrame(np.nan, index=close.index, columns=stocks)

    n_features = len(factor_names)
    train_cutoffs = full_dates[::RETRAIN_EVERY]
    n_models = 0

    for cutoff in train_cutoffs:
        # 训练：≤ cutoff 的所有样本
        train = full[full.index.get_level_values("date") <= cutoff]
        if len(train) < 20000:
            continue

        X_train = train[factor_names].values
        y_train = train["target"].values

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
            reg_lambda=1.0, random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        n_models += 1

        # 预测：cutoff 之后每个交易日
        future = [d for d in full_dates if d > cutoff]
        for pred_date in future:
            try:
                row = full.loc[pred_date]
            except KeyError:
                continue
            X_pred = row[factor_names].values
            preds = model.predict(X_pred)
            for j, s in enumerate(row.index):
                if s in composite.columns:
                    composite.loc[pred_date, s] = float(preds[j])

    valid = composite.dropna(how="all").index
    print(f"      XGBoost: {n_models}模型 | {len(valid)}有效预测日")
    # 截面 rank 归一化
    composite = composite.rank(axis=1, pct=True)
    return composite


def _stats(bt):
    s = summary(bt["equity"], bt["port_ret"])
    return {"return": s["total_return"], "sharpe": s["sharpe"],
            "dd": s["max_drawdown"], "ann": s["annualized_return"]}


def filter_quality(close, volume, min_price=5.0, min_daily_volume=300000):
    """过滤低价/低流动性/极端收益股票，避免 pct_change 数值爆炸。"""
    avg_vol = volume.rolling(60).mean().iloc[-1]
    last_price = close.iloc[-1]
    ok = (last_price >= min_price) & (avg_vol >= min_daily_volume)

    # 额外：检查是否有极端日收益（单日 > 200% 说明数据有问题）
    ret = close.pct_change()
    extreme_days = (ret.abs() > 2.0).sum()  # 单日涨跌 >200%
    ok &= (extreme_days <= 10)  # 全周期极端日不超过10天

    keep = ok[ok].index.tolist()
    print(f"    质量过滤: {len(keep)}/{len(close.columns)} 只 "
          f"(价格≥${min_price}, 日均量≥{min_daily_volume/1e6:.0f}M)")
    return keep


def main():
    print("=" * 72)
    print("  Phase 5: XGBoost + 风险平价 × 全市场动态池")
    print("=" * 72)

    # ── 数据 ──
    tickers = load_dynamic_tickers()
    print(f"\n[1] 动态池: {len(tickers)} 只")
    panels = build_ohlcv_panels(tickers, loader=us_loader)
    close = panels["close"]
    n_stocks = close.notna().any().sum()
    span = f"{close.index.min().date()}~{close.index.max().date()}"
    print(f"    行情: {close.shape[0]}天 × {n_stocks}只有数据 | {span}")

    # 质量过滤
    keep = filter_quality(close, panels["volume"])
    close = close[keep]
    n_after_price = len(close.columns)
    panels["close"] = close
    panels["amount"] = panels["amount"][keep]
    panels["high"] = panels["high"][keep]
    panels["low"] = panels["low"][keep]
    panels["volume"] = panels["volume"][keep]

    # 系统性数据清洗：剔除极端日收益的股票
    daily_ret = close.pct_change()
    # 单日涨 >500% 或跌 >80% 视为数据错误
    bad_stocks = (daily_ret > 5.0).any() | (daily_ret < -0.8).any()
    clean_stocks = bad_stocks[~bad_stocks].index.tolist()
    if len(clean_stocks) < len(close.columns):
        print(f"    剔除极端收益: {len(close.columns)-len(clean_stocks)} 只 (单日>500%或<-80%)")
        close = close[clean_stocks]
        for k in ["close", "amount", "high", "low", "volume"]:
            panels[k] = panels[k][clean_stocks]

    close = panels["close"]
    print(f"    最终有效: {len(close.columns)} 只 (从{len(tickers)}只经过滤)")

    # 清理 inf/NaN
    for k in ["close", "amount", "high", "low", "volume"]:
        panels[k] = panels[k].replace([np.inf, -np.inf], np.nan)
        panels[k] = panels[k].ffill().fillna(0.0)

    fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
    n_fund = fund["roe"].notna().any().sum()
    print(f"    基本面: {n_fund} 只有效")

    fwd = forward_returns(close, horizon=HORIZON)

    # ── 因子 ──
    do_neutralize = "--neutralize" in sys.argv
    print(f"\n[2] 构建 15 因子{'（市值中性化）' if do_neutralize else '（原始方向定向）'}...")
    facs = build_all_factors(close, fund, panels)

    # 市值中性化
    if do_neutralize:
        import pickle
        spot_df = pd.read_pickle(PROJECT_ROOT / "data" / "raw" / "us_spot_list.pkl")
        spot_df["mktcap"] = pd.to_numeric(spot_df["mktcap"], errors="coerce")
        spot_df["symbol_clean"] = spot_df["symbol"].str.upper().str.replace(".", "-", regex=False)
        mktcap_map = dict(zip(spot_df["symbol_clean"], spot_df["mktcap"]))
        log_mv_data = {}
        stocks_in_close = close.columns.tolist()
        for s in stocks_in_close:
            mc = mktcap_map.get(s, np.nan)
            log_mv_data[s] = np.log(max(mc, 1e6)) if mc > 0 else np.nan
        log_mv = pd.DataFrame([log_mv_data], index=close.index[:1])
        log_mv = log_mv.reindex(close.index).ffill()  # 静态广播到所有日期
        print(f"    市值覆盖: {log_mv.notna().sum().sum()} / {len(stocks_in_close)} 只")
        facs = {n: neut_fn(f, log_mv=log_mv, mode="size") for n, f in facs.items()}
        print("    ✅ 市值中性化完成")
    eq_linear = combine_factors(*facs.values())
    print(f"    因子数: {len(facs)}")

    # ── 线性等权基线 ──
    print(f"\n[3] 回测对比...\n")
    bt_ew_eq = long_top_layer(close, eq_linear, rebalance_every=REBALANCE, cost_fn=US_COST_FN)
    bt_rp_eq = long_top_layer(close, eq_linear, rebalance_every=REBALANCE, cost_fn=US_COST_FN, weight_mode="risk_parity")

    # ── XGBoost 合成 ──
    print(f"\n[4] XGBoost walk-forward 合成（retrain/{RETRAIN_EVERY}d）...")
    composite_xgb = build_xgboost_composite(facs, close, fwd)
    start_d = composite_xgb.dropna(how="all").index[0]
    print(f"    首条预测: {start_d.date()}")

    close_sub = close.loc[start_d:]
    eq_linear_sub = eq_linear.loc[start_d:]
    xgb_sub = composite_xgb.loc[start_d:]

    bt_rp_xgb = long_top_layer(close_sub, xgb_sub, rebalance_every=REBALANCE, cost_fn=US_COST_FN, weight_mode="risk_parity")

    # ── 比较 ──
    print(f"\n{'方案':35s} {'累计':>8s} {'年化':>7s} {'夏普':>6s} {'回撤':>7s}")
    print(f"{'-'*70}")
    for label, bt in [("① 等权L5+等权因子", bt_ew_eq),
                       ("② 风险平价L5+等权因子", bt_rp_eq),
                       ("③ 风险平价L5+XGBoost", bt_rp_xgb)]:
        s = _stats(bt)
        print(f"{label:35s} {s['return']:>+8.1%} {s['ann']:>+7.1%} {s['sharpe']:>+6.2f} {s['dd']:>+7.1%}")

    # 基准
    s_bench = _stats(bt_ew_eq)
    bench_ret = bt_ew_eq["benchmark_ret"]
    s_b = summary((1+bench_ret).cumprod(), bench_ret)
    print(f"{'📊 等权基准':35s} {s_b['total_return']:>+8.1%} {s_b['annualized_return']:>+7.1%} {s_b['sharpe']:>+6.2f} {s_b['max_drawdown']:>+7.1%}")

    # 图
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(bt_ew_eq["equity"].index, bt_ew_eq["equity"], label="① EW L5 + EW Factors", lw=1.8, color="steelblue")
    ax.plot(bt_rp_eq["equity"].index, bt_rp_eq["equity"], label="② RP L5 + EW Factors", lw=1.8, color="darkorange")
    ax.plot(bt_rp_xgb["equity"].index, bt_rp_xgb["equity"], label="③ RP L5 + XGBoost", lw=2.2, color="red")
    ax.plot((1+bench_ret).cumprod().index, (1+bench_ret).cumprod(), label="Benchmark", lw=1.0, ls="--", color="gray")
    ax.set_title(f"Phase 5: Dynamic Pool ({n_stocks} stocks) — XGBoost + Risk Parity")
    ax.set_ylabel("Net Value (log)"); ax.set_yscale("log"); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = RAW_DATA_DIR / "phase5_xgboost_dynamic.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n  图: {png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
