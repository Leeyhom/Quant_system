# -*- coding: utf-8 -*-
"""Validate the JoinQuant v2 A-share strategy in the local framework.

This script has two jobs:
1. Diagnose exported JoinQuant transaction/position files under jointquant/.
2. Re-test the proposed v2 strategy with local akshare/RQAlpha-style data.

It intentionally keeps the model close to the JoinQuant v2 cloud strategy:
full M14 factors, industry/size neutralization, fixed top-N, 100-share lots,
95% max exposure, A-share fees, and unfilled cash left idle.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.backtest.cn_cost import cn_trade_cost_yuan
from quant.backtest.metrics import summary
from quant.data.industry import industry_series
from quant.data.panel import build_ohlcv_panels, build_value_panels
from quant.data.universe import DEFAULT_POOL
from quant.factor import factors as F
from quant.factor.factors import combine_factors
from quant.factor.neutralize import neutralize


JQ_DIR = PROJECT_ROOT / "jointquant" / "v2"
OUT_CSV = JQ_DIR / "joinquant_v2_validation.csv"
OUT_MD = JQ_DIR / "joinquant_analysis.md"

CAPITAL = 60_000.0
MAX_EXPOSURE = 0.95


def _load_joinquant_exports() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trx = pd.read_csv(JQ_DIR / "transaction.csv", encoding="gbk")
    trx["date"] = pd.to_datetime(trx["日期"])
    trx["code"] = trx["标的"].astype(str).str.extract(r"\((\d+\.XS(?:HG|HE))\)")[0]
    trx["qty"] = trx["成交数量"].astype(str).str.replace("股", "", regex=False).astype(float)
    trx["amount"] = pd.to_numeric(trx["成交额"], errors="coerce")
    trx["fee"] = pd.to_numeric(trx["手续费"], errors="coerce").fillna(0.0)
    trx["pnl"] = pd.to_numeric(trx["平仓盈亏"], errors="coerce").fillna(0.0)

    pos = pd.read_csv(JQ_DIR / "position.csv", encoding="gbk").reset_index()
    pos.columns = [
        "date",
        "asset_type",
        "target",
        "side",
        "qty",
        "available",
        "close",
        "value",
        "float_pnl",
        "open_avg",
        "fut_avg",
        "margin",
        "day_pnl",
        "today_qty",
        "pnl_pct",
        "total_value",
        "weight_pct",
    ]
    pos["date"] = pd.to_datetime(pos["date"])
    for col in ["close", "value", "float_pnl", "day_pnl", "total_value"]:
        pos[col] = pd.to_numeric(pos[col], errors="coerce")
    pos["code"] = pos["target"].astype(str).str.extract(r"\((\d+\.XS(?:HG|HE))\)")[0]

    stock = pos[pos["asset_type"].eq("股票")].copy()
    cash = pos[pos["target"].eq("Cash")].copy()
    stock_daily = stock.groupby("date").agg(
        stock_value=("value", "sum"),
        holdings=("code", "nunique"),
        day_pnl=("day_pnl", "sum"),
    )
    total_daily = stock.groupby("date")["total_value"].max().rename("total_value")
    cash_daily = cash.groupby("date")["close"].last().rename("cash")
    daily = pd.concat([stock_daily, total_daily, cash_daily], axis=1).sort_index()
    daily["cash"] = daily["cash"].fillna(daily["total_value"] - daily["stock_value"])
    daily["exposure"] = daily["stock_value"] / daily["total_value"]
    daily["cash_ratio"] = daily["cash"] / daily["total_value"]
    daily["ret"] = daily["total_value"].pct_change().fillna(0.0)
    return trx, pos, daily


def _diagnose_joinquant() -> dict:
    trx, pos, daily = _load_joinquant_exports()
    sell = trx[trx["交易类型"].eq("卖")].copy()
    trade_by_date = trx.groupby("date").agg(
        orders=("code", "count"),
        buy_amount=("amount", lambda s: s[s > 0].sum()),
        sell_amount=("amount", lambda s: -s[s < 0].sum()),
        fee=("fee", "sum"),
        realized_pnl=("pnl", "sum"),
    )
    trade_by_date["gross_turnover"] = (
        trade_by_date["buy_amount"] + trade_by_date["sell_amount"]
    ) / daily["total_value"].reindex(trade_by_date.index)

    by_code = sell.groupby("code").agg(
        sells=("code", "count"),
        realized_pnl=("pnl", "sum"),
        sell_amount=("amount", lambda s: -s.sum()),
        fee=("fee", "sum"),
    )
    by_code["pnl_rate"] = by_code["realized_pnl"] / by_code["sell_amount"]

    wealth = daily["total_value"]
    dd = wealth / wealth.cummax() - 1.0
    dd_end = dd.idxmin()
    dd_start = wealth.loc[:dd_end].idxmax()

    return {
        "daily": daily,
        "trade_by_date": trade_by_date,
        "by_code": by_code,
        "start_value": float(wealth.iloc[0]),
        "end_value": float(wealth.iloc[-1]),
        "total_return": float(wealth.iloc[-1] / CAPITAL - 1.0),
        "ann_return": float((wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(wealth)) - 1),
        "sharpe_raw": float(np.sqrt(252) * daily["ret"].mean() / daily["ret"].std()),
        "max_dd": float(-dd.min()),
        "avg_exposure": float(daily["exposure"].mean()),
        "avg_cash": float(daily["cash_ratio"].mean()),
        "max_cash": float(daily["cash_ratio"].max()),
        "trade_dates": int(len(trade_by_date)),
        "orders": int(len(trx)),
        "total_fee": float(trx["fee"].sum()),
        "sell_winrate": float((sell["pnl"] > 0).mean()),
        "avg_win": float(sell.loc[sell["pnl"] > 0, "pnl"].mean()),
        "avg_loss": float(sell.loc[sell["pnl"] < 0, "pnl"].mean()),
        "dd_start": dd_start,
        "dd_end": dd_end,
    }


def _build_full_factor(close: pd.DataFrame, amount: pd.DataFrame, value: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ind = industry_series(list(close.columns))
    log_mv = np.log(value["total_mv"].replace(0, np.nan))
    raw = {
        "earnings_yield": F.earnings_yield(value["pe_ttm"]),
        "cashflow_yield": F.cashflow_yield(value["pcf"]),
        "growth_peg": F.growth_peg(value["peg"]),
        "amihud": F.amihud_illiquidity(close, amount, 20),
    }
    facs = {
        name: neutralize(factor, industry=ind, log_mv=log_mv, mode="full")
        for name, factor in raw.items()
    }
    return combine_factors(*facs.values())


def _build_simple_factor(close: pd.DataFrame, amount: pd.DataFrame, value: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return combine_factors(
        F.earnings_yield(value["pe_ttm"]),
        F.book_to_price(value["pb"]),
        F.small_size(value["total_mv"]),
        F.reversal(close, 20),
        F.reversal(close, 5),
        F.amihud_illiquidity(close, amount, 20),
    )


def _industry_limited_selection(
    scores: pd.Series,
    prices: pd.Series,
    industry: dict[str, str],
    top_n: int,
    industry_cap: int,
    slot_value: float,
) -> list[str]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for stock in scores.index:
        price = prices.get(stock, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        if price * 100 > slot_value * 1.15:
            continue
        ind = industry.get(stock, "其他")
        if industry_cap > 0 and counts.get(ind, 0) >= industry_cap:
            continue
        selected.append(stock)
        counts[ind] = counts.get(ind, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def lot_backtest(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
    industry_cap: int,
    capital: float = CAPITAL,
    max_exposure: float = MAX_EXPOSURE,
    first_rebalance: bool = False,
) -> dict:
    industry = industry_series(list(close.columns)).to_dict()
    cash = capital
    shares = pd.Series(0.0, index=close.columns)
    values: list[float] = []
    returns: list[float] = []
    turnovers: list[float] = []
    holdings: list[int] = []
    fees: list[float] = []
    cash_ratios: list[float] = []
    prev_value = capital

    for i, date in enumerate(close.index):
        prices = close.loc[date]
        stock_value = float((shares * prices.fillna(0.0)).sum())
        total_value = cash + stock_value

        if first_rebalance:
            should_rebalance = i > 0 and (i - 1) % rebalance_every == 0
        else:
            should_rebalance = i > 0 and i % rebalance_every == 0

        if should_rebalance:
            scores = factor.iloc[i - 1].dropna().sort_values(ascending=False)
            scores = scores[scores.index.isin(prices.dropna().index)]
            slot = total_value * max_exposure / top_n
            selected = _industry_limited_selection(scores, prices, industry, top_n, industry_cap, slot)

            target_shares = pd.Series(0.0, index=close.columns)
            if selected:
                slot = total_value * max_exposure / len(selected)
                for stock in selected:
                    target_shares[stock] = math.floor(slot / prices[stock] / 100.0) * 100.0

            delta = target_shares - shares
            notional = (delta.abs() * prices).fillna(0.0)
            fee = cn_trade_cost_yuan(notional, slippage=0.0005)
            cash = total_value - float((target_shares * prices.fillna(0.0)).sum()) - fee
            shares = target_shares
            stock_value = float((shares * prices.fillna(0.0)).sum())
            total_value = cash + stock_value

            turnovers.append(float(notional.sum() / max(total_value, 1.0)))
            holdings.append(int((shares > 0).sum()))
            fees.append(float(fee))

        values.append(total_value)
        returns.append(total_value / prev_value - 1.0 if len(values) > 1 else 0.0)
        cash_ratios.append(cash / total_value if total_value > 0 else 0.0)
        prev_value = total_value

    ret = pd.Series(returns, index=close.index)
    equity = pd.Series(values, index=close.index) / capital
    out = summary(equity, ret)
    out.update(
        {
            "final_value": float(values[-1]),
            "avg_turnover": float(np.mean(turnovers)),
            "avg_holdings": float(np.mean(holdings)),
            "avg_fee": float(np.mean(fees)),
            "avg_cash": float(np.mean(cash_ratios)),
        }
    )
    return out


def run_validation() -> pd.DataFrame:
    panels = build_ohlcv_panels(DEFAULT_POOL, start="20180101", end="20251231")
    close = panels["close"]
    amount = panels["amount"]
    value = build_value_panels(DEFAULT_POOL, start="20180101", end="20251231", align_to=close)

    full_factor = _build_full_factor(close, amount, value)
    simple_factor = _build_simple_factor(close, amount, value)

    ret = close.pct_change().fillna(0.0)
    available = close.notna().astype(float)
    bench_ret = (available.div(available.sum(axis=1), axis=0).fillna(0.0) * ret).sum(axis=1)
    bench_equity = (1.0 + bench_ret).cumprod()
    bench = summary(bench_equity, bench_ret)

    rows = []
    configs = [
        ("jq_v1_simple_like_late", simple_factor, 6, 60, 0, False),
        ("v2_full_top12_60_ind1_late", full_factor, 12, 60, 1, False),
        ("v2_full_top12_60_ind1_first", full_factor, 12, 60, 1, True),
        ("v2_full_top12_40_ind1_first", full_factor, 12, 40, 1, True),
        ("v2_full_top10_40_ind1_first", full_factor, 10, 40, 1, True),
        ("v2_full_top8_40_ind1_first", full_factor, 8, 40, 1, True),
    ]
    for name, factor, top_n, rebalance, industry_cap, first_rebalance in configs:
        m = lot_backtest(close, factor, top_n, rebalance, industry_cap, first_rebalance=first_rebalance)
        rows.append(
            {
                "name": name,
                "top_n": top_n,
                "rebalance_days": rebalance,
                "industry_cap": industry_cap,
                "first_rebalance": first_rebalance,
                **m,
                "bench_total_return": bench["total_return"],
                "bench_sharpe": bench["sharpe"],
                "excess_sharpe": m["sharpe"] - bench["sharpe"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False)
    df.to_csv(OUT_CSV, index=False)
    return df


def write_report(validation: pd.DataFrame, diag: dict) -> None:
    worst = diag["by_code"].sort_values("realized_pnl").head(8)
    best = diag["by_code"].sort_values("realized_pnl", ascending=False).head(8)
    rec = validation.iloc[0]
    lines = [
        "# JoinQuant 回测诊断与 v2 本地验证",
        "",
        "## 聚宽导出诊断",
        "",
        f"- 区间：{diag['daily'].index.min().date()} ~ {diag['daily'].index.max().date()}",
        f"- 总收益：{diag['total_return']:.2%}，年化：{diag['ann_return']:.2%}",
        f"- 原始日频夏普：{diag['sharpe_raw']:.2f}，最大回撤：{diag['max_dd']:.2%}",
        f"- 平均股票仓位：{diag['avg_exposure']:.2%}，平均现金：{diag['avg_cash']:.2%}，最高现金：{diag['max_cash']:.2%}",
        f"- 交易日数：{diag['trade_dates']}，订单数：{diag['orders']}，总手续费：{diag['total_fee']:.2f}",
        f"- 卖出胜率：{diag['sell_winrate']:.2%}，平均盈利：{diag['avg_win']:.2f}，平均亏损：{diag['avg_loss']:.2f}",
        f"- 最大回撤段：{diag['dd_start'].date()} ~ {diag['dd_end'].date()}",
        "",
        "## 主要问题",
        "",
        "1. v1 不是本地 M14 策略的等价复刻，只用了 PE/PB/市值/短反转/Amihud，缺少现金流、成长和中性化。",
        "2. 固定 6 只导致个股错误暴露过大，亏损集中在地产、光伏、银行等周期/困境价值票。",
        "3. 6 万本金下平均现金约 11%，100 股整数手与高价股不可买造成长期现金拖累。",
        "4. 60 日调仓可控，但行业不分散时容易连续押同一类 beta。",
        "",
        "## 最差已实现盈亏",
        "",
        worst[["sells", "realized_pnl", "pnl_rate"]].to_markdown(),
        "",
        "## 最好已实现盈亏",
        "",
        best[["sells", "realized_pnl", "pnl_rate"]].to_markdown(),
        "",
        "## v2 本地验证",
        "",
        validation[
            [
                "name",
                "first_rebalance",
                "total_return",
                "annualized_return",
                "sharpe",
                "max_drawdown",
                "avg_holdings",
                "avg_cash",
                "excess_sharpe",
            ]
        ].to_markdown(index=False),
        "",
        "## 推荐版本",
        "",
        f"- `{rec['name']}`：top_n={int(rec['top_n'])}, rebalance={int(rec['rebalance_days'])}, industry_cap={int(rec['industry_cap'])}",
        f"- 本地 100 股整数手验证：收益 {rec['total_return']:.2%}，年化 {rec['annualized_return']:.2%}，夏普 {rec['sharpe']:.2f}，最大回撤 {rec['max_drawdown']:.2%}",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    diag = _diagnose_joinquant()
    validation = run_validation()
    write_report(validation, diag)
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_MD}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
