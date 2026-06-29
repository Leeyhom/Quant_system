"""Compare local framework baseline with RQAlpha execution replay."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.backtest.metrics import summary

BRIDGE_DIR = PROJECT_ROOT / "data" / "rqalpha_bridge"
SELF_BT = BRIDGE_DIR / "cn_self_backtest.csv"
RQ_RESULT = BRIDGE_DIR / "rqalpha_result.pkl"
REPORT = BRIDGE_DIR / "cn_rqalpha_compare.md"
CSV = BRIDGE_DIR / "cn_rqalpha_compare.csv"


def _load_rqalpha():
    with RQ_RESULT.open("rb") as f:
        return pickle.load(f)


def _fmt_pct(x: float) -> str:
    return f"{x:+.2%}"


def _fmt_num(x: float) -> str:
    return f"{x:+.2f}"


def main() -> None:
    if not SELF_BT.exists():
        raise FileNotFoundError(SELF_BT)
    if not RQ_RESULT.exists():
        raise FileNotFoundError(RQ_RESULT)

    self_bt = pd.read_csv(SELF_BT, parse_dates=["date"], index_col="date")
    rq = _load_rqalpha()

    rq_port = rq["portfolio"].copy()
    rq_port.index = pd.to_datetime(rq_port.index)
    rq_equity = rq_port["unit_net_value"]
    rq_ret = rq_equity.pct_change().fillna(0.0)
    rq_summary = rq["summary"]

    self_s = summary(self_bt["equity"], self_bt["port_ret"])
    rq_s = summary(rq_equity, rq_ret)

    # RQAlpha summary uses its own annualization/risk implementation; keep both
    # but compare using our metrics function for apples-to-apples on daily NAV.
    rows = [
        {
            "engine": "local_framework",
            "total_return": self_s["total_return"],
            "annualized_return": self_s["annualized_return"],
            "sharpe": self_s["sharpe"],
            "max_drawdown": self_s["max_drawdown"],
            "final_nav": float(self_bt["equity"].iloc[-1]),
        },
        {
            "engine": "rqalpha_replay",
            "total_return": rq_s["total_return"],
            "annualized_return": rq_s["annualized_return"],
            "sharpe": rq_s["sharpe"],
            "max_drawdown": rq_s["max_drawdown"],
            "final_nav": float(rq_equity.iloc[-1]),
        },
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(CSV, index=False)

    trades = rq["trades"]
    positions = rq["stock_positions"]
    account = rq["stock_account"]
    total_cost = float(trades["transaction_cost"].sum()) if len(trades) else 0.0
    final_cash = float(rq_port["cash"].iloc[-1])
    final_total = float(rq_port["total_value"].iloc[-1])
    avg_cash_ratio = float((rq_port["cash"] / rq_port["total_value"]).mean())
    final_cash_ratio = final_cash / final_total if final_total else 0.0

    lines = [
        "# RQAlpha Execution Replay Comparison",
        "",
        "Purpose: replay the project-generated A-share target weights inside RQAlpha,",
        "so differences mainly reflect execution, lot size, cost, and RQAlpha data/matching.",
        "",
        "## Performance",
        "",
        "| Engine | Total Return | Annualized | Sharpe | Max Drawdown | Final NAV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in comp.iterrows():
        lines.append(
            f"| {r['engine']} | {_fmt_pct(r['total_return'])} | "
            f"{_fmt_pct(r['annualized_return'])} | {_fmt_num(r['sharpe'])} | "
            f"{r['max_drawdown']:.2%} | {r['final_nav']:.4f} |"
        )

    diff_total = comp.loc[1, "total_return"] - comp.loc[0, "total_return"]
    diff_sharpe = comp.loc[1, "sharpe"] - comp.loc[0, "sharpe"]
    diff_dd = comp.loc[1, "max_drawdown"] - comp.loc[0, "max_drawdown"]

    lines += [
        "",
        "## Difference",
        "",
        f"- RQAlpha total return minus local: {_fmt_pct(diff_total)}",
        f"- RQAlpha Sharpe minus local: {_fmt_num(diff_sharpe)}",
        f"- RQAlpha max drawdown minus local: {diff_dd:+.2%}",
        "",
        "## RQAlpha Execution Details",
        "",
        f"- Trades: {len(trades)}",
        f"- Position rows: {len(positions)}",
        f"- Total transaction cost: {total_cost:.2f}",
        f"- Final cash: {final_cash:.2f}",
        f"- Final cash ratio: {final_cash_ratio:.2%}",
        f"- Average cash ratio: {avg_cash_ratio:.2%}",
        f"- RQAlpha reported turnover: {rq_summary.get('turnover'):.4f}",
        f"- RQAlpha reported total return: {rq_summary.get('total_returns'):.2%}",
        f"- RQAlpha reported Sharpe: {rq_summary.get('sharpe'):.4f}",
        "",
        "## Interpretation",
        "",
        "The replay is close enough to confirm that the local signal/weight schedule is not wildly",
        "mis-executed, but RQAlpha still trims performance. The likely causes are 100-share lots,",
        "cash drag from unfilled target weights, RQAlpha bundle price/adjustment differences, and",
        "its transaction-cost/matching model.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved: {REPORT}")
    print(f"Saved: {CSV}")


if __name__ == "__main__":
    main()
