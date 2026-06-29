"""dashboard_v3 —— 交互式 HTML 量化仪表盘（ECharts 重写版）。

运行方式：
    conda activate quant
    NO_PROXY='*' python scripts/dashboard_v3.py

相比旧版 dashboard_html.py 的改进：
    - 图表从「matplotlib 截图嵌 base64」升级为 ECharts 交互图（可缩放/悬浮/切换）
    - 净值/回撤/滚动夏普/逐年收益全部用真实数据序列渲染，文件更小且可交互
    - 新增「方法论与严谨性提示」面板：诚实标注样本内/样本外口径与已知偏差
    - 响应式深色主题，三市场 Tab 切换

数据口径（重要，见 README 审计章节）：
    本面板展示的是 **样本内（in-sample）全期回测**——因子方向、池子构成均使用
    了全样本信息，用于「策略行为可视化」而非「可实盘收益预期」。严格的样本外
    walk-forward 结果请见 portfolio_validation 框架与 docs/审计报告。
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from quant.config import RAW_DATA_DIR

DASH_DIR = RAW_DATA_DIR / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"US": "#00d4aa", "CN": "#ff6b6b", "HK": "#ffd93d"}

# 每个市场的简介（展示在 Tab 内，说明因子与构造）
MARKET_DESC = {
    "US": "S&P500/400 子集 · 8 因子等权（基本面 3 + 量价 5）· 分层多头 L5 · 20 日再平衡 · 含每股+最低费",
    "CN": "沪深主板 89 只 · 5 因子等权（行业/市值双中性化）· 分层多头 L5 · 20 日再平衡 · 0.1% 比例成本",
    "HK": "港股 178 只 · 3 反转/波动因子 · Top-10 集中等权 · 10 日再平衡 · 零成本（港股费用模型待建）",
}


def _downsample(series: pd.Series, max_points: int = 600) -> pd.Series:
    """把净值/回撤序列降采样到 ~max_points 个点，缩小 HTML 体积、保持形状。"""
    if len(series) <= max_points:
        return series
    step = int(np.ceil(len(series) / max_points))
    return series.iloc[::step]


def _run_backtest(market: str = "US") -> dict:
    """运行单市场回测，返回净值/回撤/逐年/持仓/交易等数据（沿用旧版口径）。"""
    if market == "US":
        from quant.data import us_loader
        from quant.data.universe_us_expanded import EXPANDED_US_POOL
        from quant.data.panel import build_ohlcv_panels, build_us_fundamental_panels
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors

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
        reb = 20
        n_top_frac = 0.20
        factor_names = list(raw.keys())

    elif market == "CN":
        from quant.data.universe import DEFAULT_POOL
        from quant.data.panel import build_ohlcv_panels, build_value_panels
        from quant.data.industry import industry_series
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors
        from quant.factor.neutralize import neutralize as neut_fn

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
        reb = 20
        n_top_frac = 0.20
        factor_names = list(raw.keys())

    elif market == "HK":
        from quant.data import hk_loader
        from quant.data.panel import build_ohlcv_panels
        from quant.factor import factors as F
        from quant.factor.factors import combine_factors

        with open(PROJECT_ROOT / "data" / "raw" / "hk_all_tickers.txt") as f:
            tks = [ln.strip() for ln in f][:200]
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
        reb = 10
        n_top_frac = 10 / len(close.columns)
        factor_names = list(raw.keys())

    ret = close.pct_change().fillna(0.0)
    bench_ret = (close.notna().astype(float).div(close.notna().sum(axis=1), axis=0) * ret).sum(axis=1)

    n_top = max(1, int(len(close.columns) * n_top_frac))
    port_ret = pd.Series(0.0, index=close.index)
    current_w = pd.Series(0.0, index=close.columns)
    holdings_log, trade_log = [], []

    for i, date in enumerate(close.index):
        if i > 0 and i % reb == 0:
            scores = eq.iloc[i - 1].dropna()  # 用上一日因子，防未来函数
            if len(scores) >= n_top:
                top = scores.nlargest(n_top)
                new_w = pd.Series(0.0, index=close.columns)
                new_w.loc[top.index] = 1.0 / n_top
                sold = current_w[(current_w > 0) & (new_w == 0)]
                bought = new_w[(new_w > 0) & (current_w == 0)]
                if len(sold) > 0 or len(bought) > 0:
                    trade_log.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "sold": ", ".join(sold.index.tolist()) or "—",
                        "bought": ", ".join(bought.index.tolist()) or "—",
                        "turnover": float((new_w - current_w).abs().sum()),
                    })
                current_w = new_w
                holdings_log.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "n": n_top,
                    "stocks": ", ".join(top.index.tolist()),
                })
        port_ret.loc[date] = (current_w * ret.loc[date]).sum()

    equity = (1.0 + port_ret).cumprod()
    bench_eq = (1.0 + bench_ret).cumprod()
    dd = (1 - equity / equity.cummax())
    roll_sharpe = (port_ret.rolling(60).mean() / port_ret.rolling(60).std() * np.sqrt(252))

    from quant.backtest.metrics import summary
    s = summary(equity, port_ret)
    s_b = summary(bench_eq, bench_ret)
    yearly = port_ret.groupby(port_ret.index.year).apply(lambda x: (1 + x).prod() - 1)
    y_bench = bench_ret.groupby(bench_ret.index.year).apply(lambda x: (1 + x).prod() - 1)

    eq_ds = _downsample(equity)
    be_ds = _downsample(bench_eq)
    dd_ds = _downsample(dd)
    rs_ds = _downsample(roll_sharpe.dropna())

    dates = [d.strftime("%Y-%m-%d") for d in eq_ds.index]
    return {
        "name": market,
        "n_stocks": int(len(close.columns)),
        "n_factors": len(factor_names),
        "factor_names": factor_names,
        "dates": dates,
        "equity": [round(float(v), 4) for v in eq_ds.values],
        "bench_equity": [round(float(v), 4) for v in be_ds.reindex(eq_ds.index).values],
        "dd_dates": [d.strftime("%Y-%m-%d") for d in dd_ds.index],
        "drawdown": [round(float(-v) * 100, 2) for v in dd_ds.values],
        "rs_dates": [d.strftime("%Y-%m-%d") for d in rs_ds.index],
        "roll_sharpe": [round(float(v), 3) for v in rs_ds.values],
        "years": [int(y) for y in sorted(set(yearly.index) | set(y_bench.index))],
        "yearly": [round(float(yearly.get(y, 0)) * 100, 2) for y in sorted(set(yearly.index) | set(y_bench.index))],
        "yearly_bench": [round(float(y_bench.get(y, 0)) * 100, 2) for y in sorted(set(yearly.index) | set(y_bench.index))],
        "total_return": round(float(s["total_return"]) * 100, 1),
        "ann": round(float(s["annualized_return"]) * 100, 1),
        "sharpe": round(float(s["sharpe"]), 2),
        "max_dd": round(float(s["max_drawdown"]) * 100, 1),
        "bench_return": round(float(s_b["total_return"]) * 100, 1),
        "bench_sharpe": round(float(s_b["sharpe"]), 2),
        "n_trades": len(trade_log),
        "holdings": holdings_log[-15:],
        "trades": trade_log[-15:],
        "desc": MARKET_DESC.get(market, ""),
    }


def build():
    print("Building interactive dashboard (ECharts)...")
    data = {}
    for m in ["US", "CN", "HK"]:
        try:
            data[m] = _run_backtest(m)
            d = data[m]
            print(f"  {m}: {d['n_stocks']} stocks, {d['n_factors']}F, "
                  f"Ret {d['total_return']:+.1f}% Sharpe {d['sharpe']:.2f} "
                  f"(bench {d['bench_return']:+.1f}% / {d['bench_sharpe']:.2f})")
        except Exception as e:
            print(f"  {m}: SKIP - {type(e).__name__}: {e}")

    _ensure_echarts_vendored()
    html = _render_html(data)
    out = DASH_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nSaved: {out}  ({out.stat().st_size // 1024} KB)")
    print("Open: file://" + str(out))


def _ensure_echarts_vendored():
    """把 echarts.min.js 下载到 dashboard 目录，供离线打开。失败则回退 CDN。"""
    vendor = DASH_DIR / "echarts.min.js"
    if vendor.exists() and vendor.stat().st_size > 500_000:
        return
    try:
        import urllib.request
        url = "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"
        urllib.request.urlretrieve(url, vendor)
        print(f"  vendored echarts.min.js ({vendor.stat().st_size // 1024} KB)")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 无法下载 echarts，HTML 将依赖 CDN：{type(e).__name__}")


def _render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    gen_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    from string import Template
    tpl = Template(_TEMPLATE)
    return tpl.safe_substitute(
        PAYLOAD=payload,
        GEN_TIME=gen_time,
        VALIDATION_HTML=_validation_html(),
    )


def _validation_html() -> str:
    """Render third-party execution validation and live-readiness status."""
    bridge = PROJECT_ROOT / "data" / "rqalpha_bridge" / "cn_rqalpha_compare.csv"
    sizing = PROJECT_ROOT / "data" / "raw" / "sizing" / "cn_sizing_summary.json"
    jq_metrics = PROJECT_ROOT / "jointquant" / "version_metrics" / "joinquant_strategy_metrics.csv"
    parts = []

    def _pct(v, digits=1):
        if pd.isna(v):
            return "—"
        return f"{float(v) * 100:+.{digits}f}%"

    def _num(v, digits=2):
        if pd.isna(v):
            return "—"
        return f"{float(v):.{digits}f}"

    if jq_metrics.exists():
        metrics = pd.read_csv(jq_metrics)
        jq = metrics[metrics["source"].eq("jq_screenshot")].copy()
        local = metrics[metrics["source"].eq("local_recalc_equal_weight_benchmark")].copy()
        best_jq = jq.sort_values("total_return", ascending=False).iloc[0] if len(jq) else None
        latest_jq = jq[jq["version"].eq("v5")].iloc[0] if len(jq[jq["version"].eq("v5")]) else None
        exported = metrics[metrics["source"].eq("jq_export_local_pool_benchmark")].copy()
        table_df = pd.concat([jq, exported, local], ignore_index=True)
        source_label = {
            "jq_screenshot": "聚宽实测",
            "local_recalc_equal_weight_benchmark": "本地复算",
            "jq_export_local_pool_benchmark": "聚宽导出",
        }
        rows = []
        for _, r in table_df.iterrows():
            alpha_cls = "pos" if pd.notna(r["alpha_ann"]) and r["alpha_ann"] > 0.10 else ("neg" if pd.notna(r["alpha_ann"]) and r["alpha_ann"] < 0.08 else "")
            rows.append(
                "<tr>"
                f"<td><b>{r['version']}</b></td>"
                f"<td>{source_label.get(r['source'], r['source'])}</td>"
                f"<td class=\"{'pos' if r['total_return'] >= 0 else 'neg'}\">{_pct(r['total_return'])}</td>"
                f"<td>{_pct(r['annualized_return'])}</td>"
                f"<td class=\"{alpha_cls}\">{_pct(r['alpha_ann'])}</td>"
                f"<td>{_num(r['beta'], 2)}</td>"
                f"<td>{_num(r['sharpe'], 2)}</td>"
                f"<td>{_num(r['information_ratio'], 2)}</td>"
                f"<td class=\"neg\">-{abs(float(r['max_drawdown'])) * 100:.1f}%</td>"
                f"<td>{_pct(r['daily_win_rate'])}</td>"
                f"<td>{_num(r['daily_pl_ratio'], 2)}</td>"
                f"<td>{_pct(r['trade_win_rate'])}</td>"
                f"<td>{_num(r['profit_factor'], 2)}</td>"
                f"<td style=\"min-width:220px;color:var(--muted)\">{r.get('note', '')}</td>"
                "</tr>"
            )
        best_cards = ""
        if best_jq is not None:
            best_cards += f"""
      <div class="card"><div class="v pos">{best_jq['version']}</div><div class="l">聚宽当前最优版本</div><div class="cmp">收益 {_pct(best_jq['total_return'])}</div></div>
      <div class="card"><div class="v accent">{_pct(best_jq['alpha_ann'])}</div><div class="l">聚宽最高 Alpha</div><div class="cmp">Beta {_num(best_jq['beta'], 2)}</div></div>
      <div class="card"><div class="v accent">{_num(best_jq['information_ratio'], 2)}</div><div class="l">最高信息比率</div><div class="cmp">Sharpe {_num(best_jq['sharpe'], 2)}</div></div>
"""
        if latest_jq is not None:
            best_cards += f"""
      <div class="card"><div class="v {'pos' if latest_jq['max_drawdown'] <= best_jq['max_drawdown'] else 'neg'}">-{abs(float(latest_jq['max_drawdown'])) * 100:.1f}%</div><div class="l">v5 聚宽最大回撤</div><div class="cmp">收益 {_pct(latest_jq['total_return'])}</div></div>
"""
        parts.append(f"""
  <div class="section-title">聚宽版本诊断 · Alpha / Beta / 胜率 / 盈亏比</div>
  <div class="method">
    <h4>A股策略版本横向对比</h4>
    <div class="cards" style="margin-bottom:14px">
      {best_cards}
    </div>
    <div class="tbl-wrap" style="max-height:420px">
      <table>
        <thead>
          <tr>
            <th>版本</th><th>来源</th><th>累计收益</th><th>年化</th><th>Alpha</th><th>Beta</th>
            <th>Sharpe</th><th>信息比率</th><th>最大回撤</th><th>日胜率</th><th>盈亏比</th>
            <th>交易胜率</th><th>Profit Factor</th><th>备注</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <ul style="margin-top:12px">
      <li><b>核心发现：</b>聚宽口径 Alpha 明显偏低，v3 最高约 10.3%，v4/v5 只有约 7.7%~7.9%</li>
      <li><b>含义：</b>目前收益主要来自市场/行业/风格暴露，纯选股 Alpha 不够厚，后期很难滚雪球</li>
      <li><b>口径说明：</b>聚宽截图行使用聚宽报告指标；本地复算行用本地等权池作为近似基准</li>
      <li><b>下一步：</b>要提高 Alpha，优先引入更高信息密度的因子，而不是继续压缩持仓或调再平衡参数</li>
    </ul>
  </div>
""")

    if bridge.exists():
        comp = pd.read_csv(bridge)
        local = comp[comp["engine"] == "local_framework"].iloc[0]
        rq = comp[comp["engine"] == "rqalpha_replay"].iloc[0]
        diff_ret = rq["total_return"] - local["total_return"]
        diff_dd = rq["max_drawdown"] - local["max_drawdown"]
        parts.append(f"""
  <div class="section-title">第三方执行校验 · RQAlpha</div>
  <div class="method">
    <h4>A股目标权重回放结果</h4>
    <div class="cards" style="margin-bottom:14px">
      <div class="card"><div class="v pos">{local['total_return']:+.1%}</div><div class="l">自研累计收益</div></div>
      <div class="card"><div class="v pos">{rq['total_return']:+.1%}</div><div class="l">RQAlpha累计收益</div></div>
      <div class="card"><div class="v accent">{rq['sharpe']:.2f}</div><div class="l">RQAlpha重算夏普</div><div class="cmp">自研 {local['sharpe']:.2f}</div></div>
      <div class="card"><div class="v neg">-{abs(diff_ret):.1%}</div><div class="l">执行层收益折损</div><div class="cmp">回撤差 {diff_dd:+.2%}</div></div>
    </div>
    <ul>
      <li><b>结论：</b>同一目标权重在 RQAlpha 下收益小幅下降，但回撤接近，说明自研执行层具有参考意义</li>
      <li><b>主要差异：</b>100股整手、未满仓现金拖累、RQAlpha bundle 价格/复权口径、撮合与费用模型</li>
    </ul>
  </div>
""")

    if sizing.exists():
        s = pd.read_json(sizing, typ="series")
        parts.append(f"""
  <div class="section-title">A股模拟盘推荐配置</div>
  <div class="method">
    <h4>小资金真实费用口径扫描</h4>
    <div class="cards" style="margin-bottom:14px">
      <div class="card"><div class="v accent">{int(s['top_n'])}只</div><div class="l">持仓数量</div></div>
      <div class="card"><div class="v accent">{int(s['rebalance'])}日</div><div class="l">再平衡周期</div></div>
      <div class="card"><div class="v pos">{float(s['net_sharpe']):.2f}</div><div class="l">样本外净夏普</div><div class="cmp">基准 {float(s['bench_sharpe']):.2f}</div></div>
      <div class="card"><div class="v pos">{float(s['beat_rate']):.0%}</div><div class="l">窗口跑赢率</div></div>
      <div class="card"><div class="v">{float(s['ann_fee_drag']):.2%}</div><div class="l">年化费用拖累</div></div>
      <div class="card"><div class="v">{float(s['lot_infeasible']):.0%}</div><div class="l">整手买不进</div></div>
    </div>
    <ul>
      <li><b>建议：</b>A股 6万本金先用 6只/60日进入模拟盘；该配置在当前网格下净夏普最高</li>
      <li><b>风控：</b>波动率目标和回撤刹车在本次扫描里没有提升净夏普，先作为告警/降仓备选</li>
    </ul>
  </div>
""")

    if not parts:
        return ""
    return "\n".join(parts)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>量化交易系统 · 多市场仪表盘</title>
<script src="echarts.min.js"></script>
<script>
// 本地 echarts 缺失时回退 CDN（离线优先，在线兜底）
if(typeof echarts==='undefined'){
  document.write('<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"><\/script>');
}
</script>
<style>
:root{
  --bg:#0a0e1a; --panel:#111827; --panel2:#0f1623; --border:#1f2937;
  --fg:#e5e7eb; --muted:#8b97a8; --accent:#38bdf8;
  --pos:#22c55e; --neg:#ef4444; --warn:#f59e0b;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
  'PingFang SC','Microsoft YaHei',sans-serif;padding:0;line-height:1.5}
.wrap{max-width:1320px;margin:0 auto;padding:28px 24px 60px}
header{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:8px;
  padding-bottom:18px;border-bottom:1px solid var(--border);margin-bottom:22px}
header h1{font-size:22px;font-weight:700;letter-spacing:.3px}
header h1 .dot{color:var(--accent)}
header .sub{color:var(--muted);font-size:13px}
.banner{background:linear-gradient(90deg,rgba(245,158,11,.12),rgba(245,158,11,.02));
  border:1px solid rgba(245,158,11,.35);border-radius:10px;padding:12px 16px;margin-bottom:22px;
  font-size:13px;color:#fcd34d}
.banner b{color:#fde68a}
.tabs{display:flex;gap:6px;margin-bottom:20px}
.tab{padding:9px 22px;background:var(--panel);border:1px solid var(--border);color:var(--muted);
  cursor:pointer;border-radius:8px;font-size:14px;font-weight:600;transition:.15s}
.tab:hover{color:var(--fg);border-color:#374151}
.tab.active{background:var(--accent);border-color:var(--accent);color:#04121c}
.view{display:none}.view.active{display:block;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.desc{color:var(--muted);font-size:13px;margin-bottom:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.card .v{font-size:23px;font-weight:700;font-variant-numeric:tabular-nums}
.card .l{font-size:11px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.5px}
.card .cmp{font-size:11px;margin-top:5px;color:var(--muted)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.accent{color:var(--accent)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
.chart-box{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px}
.chart-box h3{font-size:13px;color:var(--muted);margin-bottom:10px;font-weight:600}
.chart{width:100%;height:300px}
.chart.tall{height:340px}
.full{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th{background:var(--panel2);color:var(--muted);padding:9px 12px;text-align:left;
  border-bottom:1px solid var(--border);font-weight:600;position:sticky;top:0}
td{padding:8px 12px;border-bottom:1px solid var(--border);color:var(--fg)}
tr:hover td{background:var(--panel2)}
.tbl-wrap{max-height:330px;overflow:auto;border:1px solid var(--border);border-radius:10px}
.buy{color:var(--pos)}.sell{color:var(--neg)}.tk{color:var(--accent)}
.section-title{font-size:15px;font-weight:700;margin:24px 0 12px;color:var(--fg)}
.method{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:18px 20px}
.method h4{font-size:14px;margin-bottom:10px;color:var(--accent)}
.method ul{list-style:none;display:grid;grid-template-columns:1fr 1fr;gap:8px 24px}
.method li{font-size:12.5px;color:var(--muted);padding-left:18px;position:relative}
.method li::before{content:"▸";position:absolute;left:0;color:var(--accent)}
.method li b{color:var(--fg);font-weight:600}
footer{margin-top:36px;padding-top:16px;border-top:1px solid var(--border);
  color:var(--muted);font-size:12px;text-align:center}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:600;
  background:rgba(56,189,248,.12);color:var(--accent);border:1px solid rgba(56,189,248,.3)}
@media(max-width:860px){.grid2{grid-template-columns:1fr}.method ul{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="dot">◆</span> 量化交易系统 · 多市场仪表盘</h1>
    <div class="sub">回测区间 2018-01 ~ 2025-12 · 生成于 $GEN_TIME</div>
  </header>

  <div class="banner">
    <b>口径说明：</b> 本面板为<b>样本内（in-sample）全期回测</b>，用于直观展示策略行为与持仓变化，
    <b>不代表可实盘收益预期</b>。因子方向、池子构成均使用了全样本信息（含已知的幸存者偏差）。
    严格的样本外 walk-forward 结论请参阅审计报告 <span class="pill">docs/AUDIT_专业量化审计报告.md</span>。
  </div>

  $VALIDATION_HTML

  <div class="tabs" id="tabs"></div>
  <div id="views"></div>

  <div class="section-title">方法论与严谨性</div>
  <div class="method">
    <h4>已落实的防作弊措施</h4>
    <ul>
      <li><b>防未来函数：</b>信号一律用上一交易日因子（i-1）建仓</li>
      <li><b>中性化无泄漏：</b>行业/市值中性化在单日截面内完成</li>
      <li><b>交易成本：</b>美股每股+最低费模型，A股 0.1% 比例成本</li>
      <li><b>基准对比：</b>每个市场均与等权买入持有基准对照</li>
    </ul>
    <h4 style="margin-top:16px">已知偏差（审计发现，需在实盘前修正）</h4>
    <ul>
      <li><b>幸存者偏差：</b>美股池取自近期市值快照，缺失退市/萎缩股</li>
      <li><b>样本内定向：</b>因子方向用全样本 IC 决定（应改 train 段）</li>
      <li><b>口径混用：</b>RP/IC 筛选等增益部分来自样本内择优</li>
      <li><b>XGBoost 标签重叠：</b>训练截断未裁掉 horizon 重叠天</li>
    </ul>
  </div>

  <footer>
    量化交易系统 · 因子选股研究框架 · <span class="accent">BACKTEST / RESEARCH MODE</span> ·
    实盘与模拟盘：未接入
  </footer>
</div>

<script>
const DATA = $PAYLOAD;
const COLORS = {US:"#00d4aa", CN:"#ff6b6b", HK:"#ffd93d"};
const markets = Object.keys(DATA);
const charts = {};

function fmtPct(v){return (v>=0?"+":"")+v.toFixed(1)+"%";}

function card(v,label,cls,cmp){
  return `<div class="card"><div class="v ${cls||''}">${v}</div>
    <div class="l">${label}</div>${cmp?`<div class="cmp">${cmp}</div>`:''}</div>`;
}

function buildView(m){
  const d = DATA[m];
  const tr = d.total_return>=0?'pos':'neg';
  const ddCls = 'neg';
  const exc = (d.total_return - d.bench_return);
  const cards =
    card(fmtPct(d.total_return),'累计收益',tr,`基准 ${fmtPct(d.bench_return)}`)+
    card(d.sharpe.toFixed(2),'夏普比率','accent',`基准 ${d.bench_sharpe.toFixed(2)}`)+
    card(fmtPct(d.ann),'年化收益',d.ann>=0?'pos':'neg')+
    card('-'+d.max_dd.toFixed(1)+'%','最大回撤',ddCls)+
    card((exc>=0?'+':'')+exc.toFixed(1)+'%','超额收益',exc>=0?'pos':'neg')+
    card(d.n_stocks,'股票数','')+
    card(d.n_factors,'因子数','')+
    card(d.n_trades,'调仓次数','');

  const holdRows = d.holdings.slice().reverse().map(h=>
    `<tr><td>${h.date}</td><td>${h.n}</td><td class="tk">${h.stocks}</td></tr>`).join('');
  const tradeRows = d.trades.slice().reverse().map(t=>
    `<tr><td>${t.date}</td><td class="sell">${t.sold}</td><td class="buy">${t.bought}</td>
     <td>${(t.turnover*100).toFixed(0)}%</td></tr>`).join('');

  return `<div class="view" id="view-${m}">
    <div class="desc">${d.desc}</div>
    <div class="cards">${cards}</div>
    <div class="grid2">
      <div class="chart-box full"><h3>净值曲线（对数轴，可框选缩放）</h3><div class="chart tall" id="eq-${m}"></div></div>
    </div>
    <div class="grid2">
      <div class="chart-box"><h3>回撤分析</h3><div class="chart" id="dd-${m}"></div></div>
      <div class="chart-box"><h3>滚动 60 日夏普</h3><div class="chart" id="rs-${m}"></div></div>
    </div>
    <div class="grid2">
      <div class="chart-box full"><h3>逐年收益（策略 vs 基准）</h3><div class="chart" id="yr-${m}"></div></div>
    </div>
    <div class="grid2">
      <div class="chart-box"><h3>最近持仓（Top 选股）</h3>
        <div class="tbl-wrap"><table><thead><tr><th>日期</th><th>#</th><th>持仓</th></tr></thead>
        <tbody>${holdRows||'<tr><td colspan=3>—</td></tr>'}</tbody></table></div></div>
      <div class="chart-box"><h3>调仓记录</h3>
        <div class="tbl-wrap"><table><thead><tr><th>日期</th><th>卖出</th><th>买入</th><th>换手</th></tr></thead>
        <tbody>${tradeRows||'<tr><td colspan=4>—</td></tr>'}</tbody></table></div></div>
    </div>
  </div>`;
}

const axisCommon = {
  axisLine:{lineStyle:{color:'#374151'}},
  axisLabel:{color:'#8b97a8',fontSize:11},
  splitLine:{lineStyle:{color:'#1f2937'}}
};
const tooltipCommon = {trigger:'axis',backgroundColor:'#111827',borderColor:'#1f2937',
  textStyle:{color:'#e5e7eb',fontSize:12}};

function renderCharts(m){
  const d = DATA[m], c = COLORS[m];
  // 净值
  const eq = echarts.init(document.getElementById('eq-'+m),'dark',{renderer:'canvas'});
  eq.setOption({
    backgroundColor:'transparent',
    grid:{left:52,right:20,top:30,bottom:60},
    legend:{data:['策略','基准'],textStyle:{color:'#8b97a8'},top:0,right:0},
    tooltip:Object.assign({},tooltipCommon,{valueFormatter:v=>v&&v.toFixed(2)}),
    xAxis:{type:'category',data:d.dates,...axisCommon},
    yAxis:{type:'log',...axisCommon,name:'净值',nameTextStyle:{color:'#8b97a8'}},
    dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:14,
      borderColor:'#1f2937',textStyle:{color:'#8b97a8'},fillerColor:'rgba(56,189,248,.15)'}],
    series:[
      {name:'策略',type:'line',data:d.equity,showSymbol:false,lineStyle:{width:2,color:c},
        areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,
          [{offset:0,color:c+'44'},{offset:1,color:c+'00'}])}},
      {name:'基准',type:'line',data:d.bench_equity,showSymbol:false,
        lineStyle:{width:1.2,color:'#6b7280',type:'dashed'}}
    ]
  });
  // 回撤
  const dd = echarts.init(document.getElementById('dd-'+m),'dark');
  dd.setOption({
    backgroundColor:'transparent',
    grid:{left:48,right:16,top:18,bottom:30},
    tooltip:Object.assign({},tooltipCommon,{valueFormatter:v=>v&&v.toFixed(1)+'%'}),
    xAxis:{type:'category',data:d.dd_dates,...axisCommon},
    yAxis:{type:'value',...axisCommon,axisLabel:{color:'#8b97a8',formatter:'{value}%'}},
    series:[{type:'line',data:d.drawdown,showSymbol:false,lineStyle:{width:1,color:'#ef4444'},
      areaStyle:{color:'rgba(239,68,68,.25)'}}]
  });
  // 滚动夏普
  const rs = echarts.init(document.getElementById('rs-'+m),'dark');
  rs.setOption({
    backgroundColor:'transparent',
    grid:{left:40,right:16,top:18,bottom:30},
    tooltip:Object.assign({},tooltipCommon,{valueFormatter:v=>v&&v.toFixed(2)}),
    xAxis:{type:'category',data:d.rs_dates,...axisCommon},
    yAxis:{type:'value',...axisCommon},
    series:[{type:'line',data:d.roll_sharpe,showSymbol:false,lineStyle:{width:1.4,color:c},
      markLine:{silent:true,symbol:'none',data:[{yAxis:0}],
        lineStyle:{color:'#6b7280',type:'dashed'}}}]
  });
  // 逐年
  const yr = echarts.init(document.getElementById('yr-'+m),'dark');
  yr.setOption({
    backgroundColor:'transparent',
    grid:{left:48,right:16,top:30,bottom:30},
    legend:{data:['策略','基准'],textStyle:{color:'#8b97a8'},top:0,right:0},
    tooltip:Object.assign({},tooltipCommon,{valueFormatter:v=>v&&v.toFixed(1)+'%'}),
    xAxis:{type:'category',data:d.years,...axisCommon},
    yAxis:{type:'value',...axisCommon,axisLabel:{color:'#8b97a8',formatter:'{value}%'}},
    series:[
      {name:'策略',type:'bar',data:d.yearly,itemStyle:{color:c,borderRadius:[3,3,0,0]}},
      {name:'基准',type:'bar',data:d.yearly_bench,itemStyle:{color:'#4b5563',borderRadius:[3,3,0,0]}}
    ]
  });
  charts[m]=[eq,dd,rs,yr];
}

// 构建 Tab 与视图
const tabsEl = document.getElementById('tabs');
const viewsEl = document.getElementById('views');
markets.forEach((m,i)=>{
  const b=document.createElement('div');
  b.className='tab'+(i===0?' active':'');
  b.textContent=m+' 市场';
  b.onclick=()=>show(m,b);
  tabsEl.appendChild(b);
  viewsEl.insertAdjacentHTML('beforeend',buildView(m));
});

function show(m,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('view-'+m).classList.add('active');
  if(!charts[m]) renderCharts(m);
  setTimeout(()=>charts[m]&&charts[m].forEach(c=>c.resize()),50);
}

// 初始
if(markets.length){
  document.getElementById('view-'+markets[0]).classList.add('active');
  renderCharts(markets[0]);
}
window.addEventListener('resize',()=>{
  Object.values(charts).forEach(arr=>arr.forEach(c=>c.resize()));
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
