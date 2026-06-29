"""dashboard_html —— 交互式 HTML 量化仪表盘。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/dashboard_html.py

输出：data/raw/dashboard/dashboard.html（单文件，浏览器直接打开）
     - 三市场 Tab 切换
     - 净值曲线 + 回撤 + 逐年收益（嵌入式图表）
     - 完整持仓记录表（每个再平衡日的选股/权重）
     - 交易操作日志（买入/卖出/调仓明细）
     - 绩效总表 + 因子概况
     - 深色主题，响应式设计
"""

from __future__ import annotations

import sys
from pathlib import Path
import json, io, base64

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from quant.config import RAW_DATA_DIR

BG, FG = "#1a1a2e", "#e0e0e0"
COLORS = {"US": "#00d4aa", "CN": "#ff6b6b", "HK": "#ffd93d"}
DASH_DIR = RAW_DATA_DIR / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": "#16213e",
                     "axes.edgecolor": "#2a2a4a", "axes.labelcolor": FG,
                     "text.color": FG, "xtick.color": FG, "ytick.color": FG,
                     "grid.color": "#2a2a4a", "grid.alpha": 0.3})


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _run_backtest_with_log(market="US"):
    """运行回测并捕获完整操作日志。"""
    if market == "US":
        from quant.data import us_loader
        from quant.data.universe_us_expanded import EXPANDED_US_POOL
        from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors
        from quant.backtest.metrics import summary

        tks = EXPANDED_US_POOL[:300]
        panels = build_ohlcv_panels(tks, loader=us_loader)
        close = panels["close"]
        fund = build_us_fundamental_panels(close.columns.tolist(), align_to=close)
        a, h, l = panels["amount"], panels["high"], panels["low"]
        raw = {}
        raw["quality_roe"] = F.us_quality_roe(fund["roe"])
        raw["quality_gm"] = F.us_quality_roe(fund["gross_margin"])
        raw["growth_rev"] = F.us_growth(fund["rev_yoy"])
        raw["pv_momentum60"] = -F.momentum(close, 60)
        raw["pv_reversal20"] = F.reversal(close, 20)
        raw["pv_lowvol20"] = -F.low_volatility(close, 20)
        raw["pv_amihud"] = -F.amihud_illiquidity(close, a, 20)
        raw["pv_parkinson"] = -F.parkinson_volatility(h, l, 20)
        eq = combine_factors(*raw.values())
        reb = 20; n_top_frac = 0.20
        n_layers = 5

    elif market == "CN":
        from quant.data.universe import DEFAULT_POOL
        from quant.data.panel import build_ohlcv_panels, build_value_panels
        from quant.data.industry import industry_series
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors
        from quant.factor.neutralize import neutralize as neut_fn
        from quant.backtest.metrics import summary

        tks = DEFAULT_POOL
        panels = build_ohlcv_panels(tks)
        close = panels["close"]
        value = build_value_panels(tks, align_to=close)
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
        reb = 20; n_top_frac = 0.20
        n_layers = 5

    elif market == "HK":
        from quant.data import hk_loader
        from quant.data.panel import build_ohlcv_panels
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors
        from quant.backtest.metrics import summary

        with open(PROJECT_ROOT / "data" / "raw" / "hk_all_tickers.txt") as f:
            tks = [l.strip() for l in f][:200]
        panels = build_ohlcv_panels(tks, loader=hk_loader)
        close = panels["close"]
        ok = (close.iloc[-1] >= 1.0) & (panels["volume"].rolling(60).mean().iloc[-1] >= 50000)
        close = close[ok[ok].index.tolist()]
        h, l = panels["high"][close.columns], panels["low"][close.columns]
        raw = {}
        raw["rev60"] = F.reversal(close, 60)
        raw["park"] = -F.parkinson_volatility(h, l, 20)
        raw["rev5"] = F.reversal(close, 5)
        eq = combine_factors(*raw.values())
        reb = 10; n_top_frac = 10 / len(close.columns)
        n_layers = 5

    ret = close.pct_change().fillna(0.0)
    bench_ret = (close.notna().astype(float).div(close.notna().sum(axis=1), axis=0) * ret).sum(axis=1)

    # 模拟回测 + 记录持仓
    n_top = max(1, int(len(close.columns) * n_top_frac))
    port_ret = pd.Series(0.0, index=close.index)
    current_w = pd.Series(0.0, index=close.columns)
    holdings_log = []  # 每个再平衡日的持仓记录
    trade_log = []     # 每次调仓的交易记录

    for i, date in enumerate(close.index):
        if i > 0 and i % reb == 0:
            scores = eq.iloc[i - 1].dropna()
            if len(scores) >= n_top:
                top = scores.nlargest(n_top)
                new_w = pd.Series(0.0, index=close.columns)
                new_w.loc[top.index] = 1.0 / n_top
                # 交易记录
                sold = current_w[(current_w > 0) & (new_w == 0)]
                bought = new_w[(new_w > 0) & (current_w == 0)]
                adjusted = new_w[(new_w > 0) & (current_w > 0) & (abs(new_w - current_w) > 0.001)]
                if len(sold) > 0 or len(bought) > 0 or len(adjusted) > 0:
                    trade_log.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "sold": ", ".join([f"{s}({current_w[s]:.1%})" for s in sold.index]),
                        "bought": ", ".join([f"{s}({new_w[s]:.1%})" for s in bought.index]),
                        "adjusted": ", ".join([f"{s}({current_w[s]:.1%}→{new_w[s]:.1%})" for s in adjusted.index]),
                        "turnover": float((new_w - current_w).abs().sum()),
                    })
                current_w = new_w
                # 持仓记录
                holdings_log.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "stocks": ", ".join([f"{s}({w:.1%})" for s, w in zip(top.index, [1/n_top]*n_top)]),
                    "n_stocks": n_top,
                })
        port_ret.loc[date] = (current_w * ret.loc[date]).sum()

    equity = (1.0 + port_ret).cumprod()
    bench_eq = (1.0 + bench_ret).cumprod()
    dd = (1 - equity / equity.cummax())

    from quant.backtest.metrics import summary as sm
    s = sm(equity, port_ret)
    s_b = sm(bench_eq, bench_ret)
    yearly = port_ret.groupby(port_ret.index.year).apply(lambda x: (1 + x).prod() - 1)
    y_bench = bench_ret.groupby(bench_ret.index.year).apply(lambda x: (1 + x).prod() - 1)

    return {
        "equity": equity, "bench_eq": bench_eq, "dd": dd, "port_ret": port_ret,
        "yearly": yearly, "y_bench": y_bench, "holdings": holdings_log[-20:],
        "trades": trade_log[-20:], "n_trades": len(trade_log),
        "total_return": s["total_return"], "sharpe": s["sharpe"],
        "max_dd": s["max_drawdown"], "ann": s["annualized_return"],
        "bench_return": s_b["total_return"], "bench_sharpe": s_b["sharpe"],
        "name": market, "n_stocks": len(close.columns),
    }


def _gen_charts(data):
    """生成嵌入式图表 PNG。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(data["equity"].index, data["equity"].values, color=COLORS[data["name"]], lw=2, label="Strategy")
    ax1.plot(data["bench_eq"].index, data["bench_eq"].values, color=COLORS[data["name"]], lw=0.8, ls="--", alpha=0.5, label="Bench")
    ax1.set_title(f"{data['name']} Equity Curve (log)", fontsize=11, fontweight="bold")
    ax1.set_yscale("log"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.fill_between(data["dd"].index, 0, -data["dd"].values, color=COLORS[data["name"]], alpha=0.4)
    ax2.set_title(f"{data['name']} Drawdown", fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0)); ax2.grid(True, alpha=0.3)
    eq_b64 = _fig_to_b64(fig)

    fig2, ax3 = plt.subplots(figsize=(8, 3.5))
    years = sorted(set(data["yearly"].index) | set(data["y_bench"].index))
    x = np.arange(len(years)); w = 0.35
    ax3.bar(x - w/2, [data["yearly"].get(y, 0) for y in years], w, color=COLORS[data["name"]], alpha=0.8, label="Strategy")
    ax3.bar(x + w/2, [data["y_bench"].get(y, 0) for y in years], w, color="gray", alpha=0.4, label="Bench")
    ax3.axhline(y=0, color="white", lw=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels([str(y) for y in years])
    ax3.set_title(f"{data['name']} Yearly Returns", fontsize=10, fontweight="bold")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0)); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")
    yr_b64 = _fig_to_b64(fig2)
    return eq_b64, yr_b64


def _trade_rows(trades):
    rows = ""
    for t in trades:
        sold = t["sold"] or "—"
        bought = t["bought"] or "—"
        adj = t["adjusted"] or "—"
        rows += f"<tr><td>{t['date']}</td><td class='sell'>{sold}</td><td class='buy'>{bought}</td><td>{adj}</td><td>{t['turnover']:.1%}</td></tr>"
    return rows


def _holdings_rows(holdings):
    rows = ""
    for h in holdings:
        rows += f"<tr><td>{h['date']}</td><td>{h['n_stocks']}</td><td class='hold'>{h['stocks']}</td></tr>"
    return rows


def build_html():
    print("Building HTML dashboard...")
    markets_data = {}
    for m in ["US", "CN", "HK"]:
        try:
            markets_data[m] = _run_backtest_with_log(m)
            print(f"  {m}: {markets_data[m]['n_stocks']} stocks, "
                  f"Return {markets_data[m]['total_return']:+.1%}, "
                  f"Sharpe {markets_data[m]['sharpe']:.2f}, "
                  f"{markets_data[m]['n_trades']} trades")
        except Exception as e:
            print(f"  {m}: SKIP - {type(e).__name__}: {e}")

    # 生成图表
    charts = {}
    for m, d in markets_data.items():
        charts[m] = _gen_charts(d)

    # HTML 模板
    tab_buttons = "".join([
        f'<button class="tab-btn" onclick="showTab(\'{m}\')">{m} Market</button>'
        for m in markets_data
    ])

    tab_content = ""
    for m, d in markets_data.items():
        yr_rows = "".join([
            f"<tr><td>{y}</td><td class='pos'>{d['yearly'].get(y,0):+.1%}</td>"
            f"<td>{d['y_bench'].get(y,0):+.1%}</td>"
            f"<td class='{'pos' if d['yearly'].get(y,0)>d['y_bench'].get(y,0) else 'neg'}'>{d['yearly'].get(y,0)-d['y_bench'].get(y,0):+.1%}</td></tr>"
            for y in sorted(set(d["yearly"].index) | set(d["y_bench"].index))
        ])
        tab_content += f"""
        <div id="tab-{m}" class="tab-content">
            <div class="metrics-row">
                <div class="metric-card"><span class="metric-val pos">{d['total_return']:+.1%}</span><span class="metric-label">Total Return</span></div>
                <div class="metric-card"><span class="metric-val">{d['sharpe']:.2f}</span><span class="metric-label">Sharpe</span></div>
                <div class="metric-card"><span class="metric-val neg">{d['max_dd']:.1%}</span><span class="metric-label">Max DD</span></div>
                <div class="metric-card"><span class="metric-val">{d['ann']:+.1%}</span><span class="metric-label">Ann Return</span></div>
                <div class="metric-card"><span class="metric-val">{d['bench_return']:+.1%}</span><span class="metric-label">Bench Return</span></div>
                <div class="metric-card"><span class="metric-val">{d['n_stocks']}</span><span class="metric-label">Stocks</span></div>
                <div class="metric-card"><span class="metric-val">{d['n_trades']}</span><span class="metric-label">Trades</span></div>
            </div>
            <div class="chart-row">
                <img src="{charts[m][0]}" alt="Equity chart" class="chart-img">
            </div>
            <div class="chart-row">
                <img src="{charts[m][1]}" alt="Yearly returns" class="chart-img-small">
            </div>
            <h3>📋 Rebalance Holdings (Last 20)</h3>
            <div class="table-wrap">
            <table><tr><th>Date</th><th>#Stocks</th><th>Holdings (ticker:weight)</th></tr>
            {_holdings_rows(d['holdings'])}
            </table></div>
            <h3>📊 Trade Operations (Last 20)</h3>
            <div class="table-wrap">
            <table><tr><th>Date</th><th>Sold</th><th>Bought</th><th>Adjusted</th><th>Turnover</th></tr>
            {_trade_rows(d['trades'])}
            </table></div>
            <h3>📈 Yearly Returns</h3>
            <div class="table-wrap">
            <table><tr><th>Year</th><th>Strategy</th><th>Bench</th><th>Excess</th></tr>{yr_rows}</table></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Dashboard · Multi-Market</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:20px}}
.header{{text-align:center;padding:20px 0;border-bottom:1px solid #21262d;margin-bottom:20px}}
.header h1{{font-size:24px;color:#58a6ff}}.header p{{color:#8b949e;margin-top:4px}}
.tab-bar{{display:flex;gap:4px;margin-bottom:20px}}
.tab-btn{{padding:10px 24px;background:#161b22;border:1px solid #30363d;color:#c9d1d9;cursor:pointer;border-radius:6px 6px 0 0;font-size:14px}}
.tab-btn:hover{{background:#1f242b}}.tab-btn.active{{background:#1f6feb;border-color:#1f6feb;color:#fff}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
.metrics-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}
.metric-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px;text-align:center;flex:1;min-width:100px}}
.metric-val{{display:block;font-size:22px;font-weight:bold;font-family:monospace}}
.metric-label{{display:block;font-size:11px;color:#8b949e;margin-top:4px}}
.pos{{color:#3fb950}}.neg{{color:#f85149}}
.chart-row{{margin-bottom:16px;text-align:center}}
.chart-img{{max-width:100%;border-radius:8px;border:1px solid #30363d}}
.chart-img-small{{max-width:60%;border-radius:8px;border:1px solid #30363d}}
h3{{color:#58a6ff;margin:16px 0 8px;font-size:16px}}
.table-wrap{{overflow-x:auto;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-family:monospace}}
th{{background:#161b22;color:#8b949e;padding:8px 10px;text-align:left;border:1px solid #30363d}}
td{{padding:6px 10px;border:1px solid #21262d}}
tr:hover{{background:#161b22}}
.sell{{color:#f85149}}.buy{{color:#3fb950}}.hold{{color:#58a6ff}}
.footer{{text-align:center;padding:16px;border-top:1px solid #21262d;margin-top:20px;color:#8b949e;font-size:12px}}
</style>
</head>
<body>
<div class="header">
    <h1>📊 Quantitative Trading System · Multi-Market Dashboard</h1>
    <p>Backtest Period: 2018-01 ~ 2025-12 | US: 1396 stocks | CN: 89 stocks | HK: 178 stocks</p>
</div>
<div class="tab-bar">{tab_buttons}</div>
{tab_content}
<div class="footer">
    Quant Dashboard v2.0 · Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
    <span style="color:#3fb950">BACKTEST MODE</span> ·
    Live Trading: OFFLINE · Paper Trading: OFFLINE
</div>
<script>
document.querySelector('.tab-btn').classList.add('active');
document.querySelector('.tab-content').classList.add('active');
function showTab(m) {{
    document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active')}});
    document.querySelectorAll('.tab-content').forEach(function(c){{c.classList.remove('active')}});
    document.getElementById('tab-'+m).classList.add('active');
    event.target.classList.add('active');
}}
</script>
</body>
</html>"""

    html_path = DASH_DIR / "dashboard.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\nDashboard saved: {html_path}")
    print("Open in browser: file://" + str(html_path))
    print("Done ✅")


if __name__ == "__main__":
    build_html()
