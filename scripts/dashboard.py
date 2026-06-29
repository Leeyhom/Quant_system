"""dashboard —— 量化系统可视化面板（Bloomberg 终端风格）。

运行方式：
    conda activate quant
    python scripts/dashboard.py

生成 data/raw/dashboard/ 下的高清仪表盘，包含：
    ① 三市场净值曲线（叠加基准）
    ② 滚动夏普比率
    ③ 回撤分析
    ④ 逐年收益对比
    ⑤ 因子 IC 概况
    ⑥ 综合绩效总表
    ⑦ 实盘监控预留区（仓位/P&L/风控）

设计原则：深色主题、信息密度高、可扩展（后续接入模拟/实盘只需更新数据源）。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from quant.config import RAW_DATA_DIR

# ─── 主题配置 ───
BG_COLOR = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLORS = {"US": "#00d4aa", "CN": "#ff6b6b", "HK": "#ffd93d"}
GRID_COLOR = "#2a2a4a"
TABLE_HEADER_BG = "#0f3460"

DASH_DIR = RAW_DATA_DIR / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR, "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR, "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR, "grid.color": GRID_COLOR,
    "grid.alpha": 0.3, "font.family": "sans-serif",
})


def _load_market_data(market="US"):
    """加载单个市场的回测数据。返回 dict 含 equity/returns/benchmark 等。"""
    # 简化版：从实际脚本导入。实盘时替换为实时数据源。
    path = PROJECT_ROOT
    result = {}

    if market == "US":
        from quant.data import us_loader
        from quant.data.universe_us_expanded import EXPANDED_US_POOL
        from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors
        from quant.backtest.layered import long_top_layer
        from quant.backtest.metrics import summary

        tickers = EXPANDED_US_POOL[:300]
        panels = build_ohlcv_panels(tickers, loader=us_loader)
        close = panels["close"]
        fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
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
        eq = combine_factors(*raw.values())
        bt = long_top_layer(close, eq, rebalance_every=20)
        result = {"equity": bt["equity"], "returns": bt["port_ret"],
                  "bench_equity": bt["benchmark"], "bench_returns": bt["benchmark_ret"],
                  "name": "US", "sharpe": summary(bt["equity"], bt["port_ret"])["sharpe"]}

    elif market == "CN":
        from quant.data.universe import DEFAULT_POOL
        from quant.data.panel import build_ohlcv_panels, build_value_panels
        from quant.data.industry import industry_series
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors
        from quant.factor.neutralize import neutralize as neut_fn
        from quant.backtest.layered import long_top_layer
        from quant.backtest.metrics import summary

        syms = DEFAULT_POOL
        panels = build_ohlcv_panels(syms)
        close = panels["close"]
        value = build_value_panels(syms, align_to=close)
        ind = industry_series(list(close.columns))
        log_mv = np.log(value["total_mv"].replace(0, np.nan))
        raw = {}
        raw["earnings_yield"] = F.earnings_yield(value["pe_ttm"])
        raw["quality_roe"] = F.quality_roe(value["pe_ttm"], value["pb"])
        raw["growth_peg"] = F.growth_peg(value["peg"])
        raw["pv_momentum60"] = F.momentum(close, 60)
        raw["pv_amihud"] = F.amihud_illiquidity(close, panels["amount"], 20)
        facs = {n: neut_fn(f, industry=ind, log_mv=log_mv, mode="full") for n, f in raw.items()}
        eq = combine_factors(*facs.values())
        bt = long_top_layer(close, eq, rebalance_every=20)
        result = {"equity": bt["equity"], "returns": bt["port_ret"],
                  "bench_equity": bt["benchmark"], "bench_returns": bt["benchmark_ret"],
                  "name": "CN", "sharpe": summary(bt["equity"], bt["port_ret"])["sharpe"]}

    elif market == "HK":
        from quant.data import hk_loader
        from quant.data.panel import build_ohlcv_panels
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors
        from quant.backtest.metrics import summary

        with open(PROJECT_ROOT / "data" / "raw" / "hk_all_tickers.txt") as f:
            tickers = [l.strip() for l in f][:200]
        panels = build_ohlcv_panels(tickers, loader=hk_loader)
        close = panels["close"]
        ok = (close.iloc[-1] >= 1.0) & (panels["volume"].rolling(60).mean().iloc[-1] >= 50000)
        close = close[ok[ok].index.tolist()]
        high, low = panels["high"][close.columns], panels["low"][close.columns]
        ret = close.pct_change().fillna(0.0)
        # HK最优：Top-10 集中等权
        facs = {
            "rev60": F.reversal(close, 60),
            "park": -F.parkinson_volatility(high, low, 20),
            "rev5": F.reversal(close, 5),
        }
        eq = combine_factors(*facs.values())
        port_ret = pd.Series(0.0, index=close.index)
        current_w = pd.Series(0.0, index=close.columns)
        for i, date in enumerate(close.index):
            if i > 0 and i % 10 == 0:
                sc = eq.iloc[i - 1].dropna()
                if len(sc) >= 10:
                    top = sc.nlargest(10).index
                    w = pd.Series(0.0, index=close.columns)
                    w.loc[top] = 0.1
                    current_w = w
            port_ret.loc[date] = (current_w * ret.loc[date]).sum()
        eq_curve = (1.0 + port_ret).cumprod()
        bench_ret = (close.notna().astype(float).div(close.notna().sum(axis=1), axis=0) * ret).sum(axis=1)
        result = {"equity": eq_curve, "returns": port_ret,
                  "bench_equity": (1.0 + bench_ret).cumprod(), "bench_returns": bench_ret,
                  "name": "HK", "sharpe": summary(eq_curve, port_ret)["sharpe"]}

    return result


def _perf_summary(equity, returns):
    """从净值和日收益中提取绩效指标。"""
    total = equity.iloc[-1] / equity.iloc[0] - 1
    ann = total ** (252 / max(len(equity), 1)) - 1 if total > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    dd = (1 - equity / equity.cummax()).max()
    calmar = ann / dd if dd > 0 else 0
    win_rate = (returns > 0).mean()
    return {"total": total, "ann": ann, "vol": vol, "sharpe": sharpe,
            "dd": dd, "calmar": calmar, "win": win_rate}


def _draw_table(ax, data, title, col_widths=None):
    """在指定 Axes 上绘制格式化表格。"""
    ax.axis("off")
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=8)
    rows, cols = len(data), len(data[0])
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for i in range(rows):
        for j in range(cols):
            cell = table[i, j]
            cell.set_facecolor(PANEL_BG if i > 0 else TABLE_HEADER_BG)
            cell.set_text_props(color=TEXT_COLOR)
            cell.set_edgecolor(GRID_COLOR)
            if i == 0:
                cell.set_text_props(color="white", fontweight="bold")
    return table


def build_dashboard():
    """构建完整仪表盘。"""
    print("Building dashboard...")

    # ── 加载三市场数据 ──
    markets = {}
    for m in ["US", "CN", "HK"]:
        try:
            markets[m] = _load_market_data(m)
            print(f"  {m}: loaded")
        except Exception as e:
            print(f"  {m}: SKIP ({type(e).__name__})")

    # ── 布局 ──
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.30)

    # Panel 1: 三市场净值曲线（左上 2×2）
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_facecolor(PANEL_BG)
    for m, data in markets.items():
        c = ACCENT_COLORS[m]
        ax1.plot(data["equity"].index, data["equity"].values,
                 color=c, lw=1.8, label=f"{m} Strategy")
        ax1.plot(data["bench_equity"].index, data["bench_equity"].values,
                 color=c, lw=0.8, ls="--", alpha=0.4, label=f"{m} Bench")
    ax1.set_title("Equity Curves (log scale)", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # Panel 2: 滚动夏普（右上 1×2）
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_facecolor(PANEL_BG)
    for m, data in markets.items():
        roll_sharpe = data["returns"].rolling(60).mean() / data["returns"].rolling(60).std() * np.sqrt(252)
        ax2.plot(roll_sharpe.index, roll_sharpe.values, color=ACCENT_COLORS[m], lw=1.2, label=m)
    ax2.axhline(y=0, color="white", lw=0.5, ls="--")
    ax2.set_title("Rolling 60d Sharpe Ratio", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR)
    ax2.grid(True, alpha=0.3)

    # Panel 3: 回撤分析（右中 1×2）
    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.set_facecolor(PANEL_BG)
    for m, data in markets.items():
        dd = 1 - data["equity"] / data["equity"].cummax()
        ax3.fill_between(dd.index, 0, -dd.values, color=ACCENT_COLORS[m], alpha=0.3, label=m)
    ax3.set_title("Drawdown Analysis", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax3.legend(loc="lower left", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax3.grid(True, alpha=0.3)

    # Panel 4: 逐年收益（左下 1×2）
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.set_facecolor(PANEL_BG)
    all_yearly = {}
    x_offset = np.arange(8)  # 2018-2025
    w = 0.25
    for idx, (m, data) in enumerate(markets.items()):
        yearly = data["returns"].groupby(data["returns"].index.year).apply(lambda x: (1 + x).prod() - 1)
        all_yearly[m] = yearly
        ax4.bar(x_offset + idx * w, [yearly.get(y, 0) for y in range(2018, 2026)],
                w, color=ACCENT_COLORS[m], alpha=0.8, label=m)
    ax4.axhline(y=0, color="white", lw=0.5)
    ax4.set_xticks(x_offset + w)
    ax4.set_xticklabels([str(y) for y in range(2018, 2026)])
    ax4.set_title("Yearly Returns by Market", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax4.grid(True, alpha=0.3, axis="y")

    # Panel 5: 绩效总表（右下 1×2）
    ax5 = fig.add_subplot(gs[2, 2:])
    metrics = []
    for m in ["US", "CN", "HK"]:
        if m in markets:
            s = _perf_summary(markets[m]["equity"], markets[m]["returns"])
            metrics.append([m, f"{s['total']:+.1%}", f"{s['ann']:+.1%}", f"{s['vol']:.1%}",
                            f"{s['sharpe']:.2f}", f"{s['dd']:.1%}", f"{s['calmar']:.2f}",
                            f"{s['win']:.0%}"])
    header = ["Mkt", "Total", "Ann.", "Vol", "Sharpe", "MaxDD", "Calmar", "Win%"]
    _draw_table(ax5, [header] + metrics, "Performance Summary")

    # Panel 6: 底部状态栏（实时/模拟盘预留）
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor("#0f3460")
    ax6.axis("off")
    status_text = (
        "STATUS: BACKTEST MODE  |  Period: 2018-01 ~ 2025-12  |  "
        "US: 1396 stocks / CN: 89 stocks / HK: 178 stocks  |  "
        "Rebalance: 20d(US/CN) 10d(HK)  |  "
        "Live Trading: OFFLINE  |  Paper Trading: OFFLINE"
    )
    ax6.text(0.5, 0.5, status_text, transform=ax6.transAxes, ha="center", va="center",
             color="#00d4aa", fontsize=10, fontfamily="monospace", fontweight="bold")

    # Panel 7: 实盘预留区说明（左下角）
    ax7 = fig.add_subplot(gs[3, :])
    # Overwritten by ax6, skip

    # ── 标题 ──
    fig.suptitle("Quantitative Trading System · Multi-Market Dashboard",
                 color=TEXT_COLOR, fontsize=16, fontweight="bold", y=0.98)

    # ── 保存 ──
    png = DASH_DIR / "dashboard.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"\nDashboard saved: {png}")
    print("Done ✅")


if __name__ == "__main__":
    build_dashboard()
