# -*- coding: utf-8 -*-
"""
JoinQuant A-share simulation strategy v3.

This is the non-simplified execution adapter for the local M14 A-share logic:
    - full factor set where JoinQuant data is available:
      earnings yield, cash-flow yield, sales yield, growth/PEG proxy, Amihud
    - industry and size neutralization inside the daily cross-section
    - fixed 10-stock portfolio, one stock per industry, 95% max exposure
    - 100-share affordability filter; unfilled capital stays as cash

Paste this whole file into a JoinQuant stock strategy. Set initial capital to
60000 CNY. The strategy rebalances at 14:30 every 40 trading days.
"""

import math

import numpy as np
import pandas as pd


# ============================ User Config ============================

FEISHU_WEBHOOK = ""

INITIAL_CAPITAL = 60000
MAX_EXPOSURE = 0.95
TOP_N = 10
INDUSTRY_CAP = 1
REBALANCE_DAYS = 40
TRADE_TIME = "14:30"
STRATEGY_VERSION = "jq-cn-sim-v3-top10-reb40"
PUSH_DAILY_REPORT = False

COMMISSION_RATE = 0.00025
STAMP_DUTY_RATE = 0.0005
TRANSFER_FEE_RATE = 0.00001
EFFECTIVE_COMMISSION_RATE = COMMISSION_RATE + TRANSFER_FEE_RATE

STOCK_POOL = [
    "600036.XSHG", "601318.XSHG", "600000.XSHG", "601166.XSHG", "601328.XSHG", "601398.XSHG",
    "601288.XSHG", "600016.XSHG", "600519.XSHG", "000858.XSHE", "000568.XSHE", "600809.XSHG",
    "002304.XSHE", "603288.XSHG", "600887.XSHG", "000895.XSHE", "600276.XSHG", "300760.XSHE",
    "603259.XSHG", "002594.XSHE", "300015.XSHE", "000538.XSHE", "600196.XSHG", "002821.XSHE",
    "002415.XSHE", "002475.XSHE", "000725.XSHE", "002230.XSHE", "603501.XSHG", "603986.XSHG",
    "688981.XSHG", "688111.XSHG", "300059.XSHE", "002460.XSHE", "300433.XSHE", "600588.XSHG",
    "002241.XSHE", "601012.XSHG", "300750.XSHE", "002129.XSHE", "601877.XSHG", "600438.XSHG",
    "300274.XSHE", "002459.XSHE", "600900.XSHG", "601985.XSHG", "600025.XSHG", "003816.XSHE",
    "000333.XSHE", "000651.XSHE", "600690.XSHG", "002508.XSHE", "000100.XSHE", "603833.XSHG",
    "600104.XSHG", "601238.XSHG", "000625.XSHE", "601633.XSHG", "600031.XSHG", "000157.XSHE",
    "600030.XSHG", "601688.XSHG", "600837.XSHG", "000776.XSHE", "601211.XSHG", "000002.XSHE",
    "600048.XSHG", "601668.XSHG", "601800.XSHG", "600019.XSHG", "603799.XSHG", "600111.XSHG",
    "000709.XSHE", "600028.XSHG", "601857.XSHG", "600309.XSHG", "000792.XSHE", "600346.XSHG",
    "601111.XSHG", "600009.XSHG", "601006.XSHG", "601816.XSHG", "000876.XSHE", "300498.XSHE",
    "002714.XSHE", "600050.XSHG", "000063.XSHE", "600941.XSHG", "601989.XSHG", "600760.XSHG",
    "000768.XSHE", "002013.XSHE",
]

INDUSTRY_MAP = {
    "银行": ["600036", "601318", "600000", "601166", "601328", "601398", "601288", "600016"],
    "白酒食品": ["600519", "000858", "000568", "600809", "002304", "603288", "600887", "000895"],
    "医药": ["600276", "300760", "603259", "002594", "300015", "000538", "600196", "002821"],
    "科技电子": ["002415", "002475", "000725", "002230", "603501", "603986", "688981", "688111"],
    "计算机传媒": ["300059", "002460", "300433", "600588", "002241"],
    "新能源电力设备": ["601012", "300750", "002129", "601877", "600438", "300274", "002459"],
    "电力公用": ["600900", "601985", "600025", "003816"],
    "家电消费": ["000333", "000651", "600690", "002508", "000100", "603833"],
    "汽车制造": ["600104", "601238", "000625", "601633", "600031", "000157"],
    "券商非银": ["600030", "601688", "600837", "000776", "601211"],
    "地产建筑": ["000002", "600048", "601668", "601800"],
    "钢铁有色": ["600019", "603799", "600111", "000709"],
    "化工能源": ["600028", "601857", "600309", "000792", "600346"],
    "交运": ["601111", "600009", "601006", "601816"],
    "农业": ["000876", "300498", "002714"],
    "通信": ["600050", "000063", "600941"],
    "机械军工": ["600760", "000768"],
}


def _industry_by_stock():
    out = {}
    for ind, codes in INDUSTRY_MAP.items():
        for code in codes:
            suffix = ".XSHG" if code.startswith("6") else ".XSHE"
            out[code + suffix] = ind
    return out


INDUSTRY_BY_STOCK = _industry_by_stock()


# ============================ JoinQuant Hooks ============================

def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)

    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=STAMP_DUTY_RATE,
            open_commission=EFFECTIVE_COMMISSION_RATE,
            close_commission=EFFECTIVE_COMMISSION_RATE,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )

    try:
        set_slippage(PriceRelatedSlippage(0.0005))
    except Exception:
        pass

    g.days_since_rebalance = REBALANCE_DAYS
    g.last_targets = []
    g.last_factor_names = []
    g.last_candidate_count = 0
    g.last_score_count = 0

    run_daily(rebalance, time=TRADE_TIME)
    run_daily(after_close_report, time="after_close")

    send_feishu(
        "A股聚宽模拟盘已启动\n"
        "版本: %s\n"
        "本金: 60000\n"
        "配置: 10只 / 单行业最多1只 / 40交易日再平衡 / 总仓位95%%\n"
        "执行: 14:30 自动调仓，未成交留现金"
        % STRATEGY_VERSION
    )


def rebalance(context):
    today = str(context.current_dt.date())
    g.days_since_rebalance += 1
    if g.days_since_rebalance < REBALANCE_DAYS:
        log.info("No rebalance today: %s/%s", g.days_since_rebalance, REBALANCE_DAYS)
        return

    candidates, blocked = filter_universe(STOCK_POOL)
    scores = build_scores(context, candidates)
    g.last_candidate_count = len(candidates)
    g.last_score_count = len(scores)
    if scores.empty:
        msg = "[%s] v2 无有效候选股票，跳过调仓。" % today
        log_warning(msg)
        send_feishu(msg)
        return

    targets, skipped_select = select_targets(context, scores)
    skipped = list(blocked) + skipped_select
    if not targets:
        msg = "[%s] v2 没有可买目标，保留现金。" % today
        log_warning(msg)
        send_feishu(msg)
        return

    target_weight = MAX_EXPOSURE / len(targets)
    actions = []
    current_positions = list(context.portfolio.positions.keys())

    for stock in current_positions:
        pos = context.portfolio.positions[stock]
        if pos.total_amount <= 0:
            continue
        if stock not in targets:
            if not can_sell(stock):
                skipped.append("%s 卖出失败: 停牌或跌停" % stock)
                continue
            order_target(stock, 0)
            actions.append("SELL %s -> 0" % stock)

    portfolio_value = context.portfolio.total_value
    target_value = portfolio_value * target_weight
    for stock in targets:
        pos_value = get_position_value(context, stock)
        if target_value > pos_value and not can_buy(stock):
            skipped.append("%s 买入失败: 停牌/ST/涨停" % stock)
            continue
        if target_value < pos_value and not can_sell(stock):
            skipped.append("%s 减仓失败: 停牌或跌停" % stock)
            continue
        action, reason = order_target_lot_value(context, stock, target_value)
        if action:
            actions.append(action)
        elif reason:
            skipped.append("%s %s" % (stock, reason))

    g.days_since_rebalance = 0
    g.last_targets = targets
    msg = format_rebalance_message(context, targets, target_weight, actions, skipped, scores)
    log.info(
        "v2 rebalance %s factors=%s candidates=%s scored=%s targets=%s",
        today,
        ",".join(g.last_factor_names),
        g.last_candidate_count,
        g.last_score_count,
        ",".join(targets),
    )
    send_feishu(msg)


def after_close_report(context):
    if not PUSH_DAILY_REPORT:
        return
    today = str(context.current_dt.date())
    lines = []
    for stock, pos in context.portfolio.positions.items():
        if pos.total_amount > 0:
            lines.append("%s %d股 市值%.0f" % (stock, pos.total_amount, pos.value))
    if not lines:
        lines = ["空仓"]
    send_feishu(
        "A股模拟盘盘后状态 %s\n"
        "版本: %s\n"
        "总资产: %.2f\n"
        "现金: %.2f\n"
        "持仓市值: %.2f\n"
        "调仓计数: %d/%d\n"
        "持仓:\n%s"
        % (
            today,
            STRATEGY_VERSION,
            context.portfolio.total_value,
            context.portfolio.cash,
            context.portfolio.positions_value,
            g.days_since_rebalance,
            REBALANCE_DAYS,
            "\n".join(lines[:20]),
        )
    )


# ============================ Factor Logic ============================

def filter_universe(stocks):
    current = get_current_data()
    out = []
    skipped = []
    for stock in stocks:
        try:
            cd = current[stock]
        except Exception:
            skipped.append("%s 跳过: 无当前行情" % stock)
            continue
        name = getattr(cd, "name", "")
        if cd.paused:
            skipped.append("%s 跳过: 停牌" % stock)
            continue
        if cd.is_st or ("ST" in name) or ("退" in name):
            skipped.append("%s 跳过: ST/退市风险" % stock)
            continue
        out.append(stock)
    return out, skipped


def build_scores(context, stocks):
    if not stocks:
        return pd.Series(dtype=float)

    prev_date = context.previous_date
    val = get_fundamental_frame(stocks, prev_date)
    if val is None or len(val) == 0:
        return pd.Series(dtype=float)

    price = get_price(
        stocks,
        end_date=prev_date,
        count=121,
        frequency="daily",
        fields=["close", "money"],
        skip_paused=True,
        fq="pre",
        panel=False,
    )
    close = pivot_price(price, "close")
    money = pivot_price(price, "money")
    if close.empty:
        return pd.Series(dtype=float)

    factors = {}
    pe = numeric_series(val, "pe_ratio")
    pb = numeric_series(val, "pb_ratio")
    mv = numeric_series(val, "market_cap")

    factors["earnings_yield"] = (1.0 / pe.replace(0, np.nan)).where(pe > 0)

    pcf = first_numeric_series(val, ["pcf_ratio", "pcf"])
    if pcf is not None:
        factors["cashflow_yield"] = (1.0 / pcf.replace(0, np.nan)).where(pcf > 0)

    ps = first_numeric_series(val, ["ps_ratio", "ps"])
    if ps is not None:
        factors["sales_yield"] = (1.0 / ps.replace(0, np.nan)).where(ps > 0)

    growth = first_numeric_series(
        val,
        [
            "net_profit_growth_rate",
            "inc_net_profit_year_on_year",
            "operating_revenue_growth_rate",
            "inc_revenue_year_on_year",
        ],
    )
    if growth is not None:
        factors["growth_peg_proxy"] = (growth.where(growth > 0) / pe.replace(0, np.nan)).where(pe > 0)

    if len(close) >= 21 and not money.empty:
        ret_abs = close.pct_change().abs()
        illiq = (ret_abs / money.replace(0, np.nan)).tail(20).mean()
        factors["amihud"] = -illiq

    frame = pd.DataFrame(factors).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if frame.empty:
        return pd.Series(dtype=float)

    frame = neutralize_cross_section(frame, mv)
    log.info("v2 factor columns before scoring: %s", ",".join(frame.columns))
    ranked = frame.rank(pct=True)
    score = ranked.mean(axis=1).dropna()
    g.last_factor_names = list(frame.columns)
    return score


def get_fundamental_frame(stocks, date):
    val_table = globals().get("valuation")
    ind_table = globals().get("indicator")
    if val_table is None:
        return pd.DataFrame()

    fields = [
        val_table.code,
        val_table.pe_ratio,
        val_table.pb_ratio,
        val_table.market_cap,
    ]
    for table, names in [
        (val_table, ["pcf_ratio", "ps_ratio"]),
        (
            ind_table,
            [
                "roe",
                "net_profit_growth_rate",
                "inc_net_profit_year_on_year",
                "operating_revenue_growth_rate",
                "inc_revenue_year_on_year",
            ],
        ),
    ]:
        if table is None:
            continue
        for name in names:
            if hasattr(table, name):
                fields.append(getattr(table, name))

    q = query(*fields).filter(val_table.code.in_(stocks))
    df = get_fundamentals(q, date=date)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.set_index("code")


def neutralize_cross_section(frame, market_cap):
    out = pd.DataFrame(index=frame.index)
    log_mv = np.log(market_cap.replace(0, np.nan))
    industries = pd.Series({s: INDUSTRY_BY_STOCK.get(s, "其他") for s in frame.index})
    dummies = pd.get_dummies(industries)

    for name in frame.columns:
        y = pd.to_numeric(frame[name], errors="coerce")
        valid = y.notna() & log_mv.reindex(y.index).notna()
        if valid.sum() < 12:
            out[name] = y
            continue
        x_parts = [
            pd.Series(1.0, index=y.index, name="const"),
            log_mv.reindex(y.index).rename("log_mv"),
            dummies.reindex(y.index).fillna(0.0),
        ]
        x = pd.concat(x_parts, axis=1).loc[valid]
        yy = y.loc[valid]
        try:
            beta = np.linalg.lstsq(x.values.astype(float), yy.values.astype(float), rcond=None)[0]
            resid = yy - x.dot(beta)
            s = pd.Series(np.nan, index=y.index)
            s.loc[valid] = resid
            out[name] = s
        except Exception:
            out[name] = y
    return out


def select_targets(context, scores):
    current = get_current_data()
    portfolio_value = context.portfolio.total_value
    slot_value = portfolio_value * MAX_EXPOSURE / TOP_N
    targets = []
    counts = {}
    skipped = []
    for stock in scores.sort_values(ascending=False).index:
        try:
            cd = current[stock]
        except Exception:
            skipped.append("%s 备选跳过: 无行情" % stock)
            continue
        price = getattr(cd, "last_price", None)
        if price is None or price <= 0:
            skipped.append("%s 备选跳过: 无有效价格" % stock)
            continue
        if price * 100 > slot_value * 1.15:
            skipped.append("%s 备选跳过: 100股超过单票目标资金" % stock)
            continue
        ind = INDUSTRY_BY_STOCK.get(stock, "其他")
        if counts.get(ind, 0) >= INDUSTRY_CAP:
            continue
        targets.append(stock)
        counts[ind] = counts.get(ind, 0) + 1
        if len(targets) >= TOP_N:
            break
    return targets, skipped[:30]


def pivot_price(price_df, field):
    if price_df is None or len(price_df) == 0:
        return pd.DataFrame()
    df = price_df.copy()
    time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else df.columns[0])
    code_col = "code" if "code" in df.columns else ("security" if "security" in df.columns else None)
    if code_col is None or field not in df.columns:
        return pd.DataFrame()
    return df.pivot(index=time_col, columns=code_col, values=field).sort_index()


def numeric_series(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def first_numeric_series(df, names):
    for name in names:
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce")
            if s.notna().any():
                return s
    return None


# ============================ Execution Helpers ============================

def can_buy(stock):
    try:
        cd = get_current_data()[stock]
    except Exception:
        return False
    if cd.paused or cd.is_st:
        return False
    price = getattr(cd, "last_price", None)
    high_limit = getattr(cd, "high_limit", None)
    return not (price is not None and high_limit is not None and price >= high_limit)


def can_sell(stock):
    try:
        cd = get_current_data()[stock]
    except Exception:
        return False
    if cd.paused:
        return False
    price = getattr(cd, "last_price", None)
    low_limit = getattr(cd, "low_limit", None)
    return not (price is not None and low_limit is not None and price <= low_limit)


def get_position_value(context, stock):
    try:
        pos = context.portfolio.positions[stock]
    except Exception:
        return 0.0
    if pos is None or pos.total_amount <= 0:
        return 0.0
    return float(pos.value)


def order_target_lot_value(context, stock, target_value):
    """Adjust position using 100-share lots and skip sub-lot changes.

    JoinQuant's order_target_value can emit failed orders when the rounded
    adjustment is below 100 shares. Computing the target amount explicitly keeps
    the simulated execution closer to an A-share account.
    """
    try:
        cd = get_current_data()[stock]
        price = getattr(cd, "last_price", None)
    except Exception:
        return None, "调整失败: 无当前价格"
    if price is None or price <= 0:
        return None, "调整失败: 无有效价格"

    target_amount = int(math.floor(target_value / price / 100.0) * 100)
    try:
        current_amount = int(context.portfolio.positions[stock].total_amount)
    except Exception:
        current_amount = 0
    delta = target_amount - current_amount
    if target_amount <= 0:
        return None, "调整跳过: 目标资金不足100股"
    if abs(delta) < 100:
        return None, "调整跳过: 变化不足100股"

    order_target(stock, target_amount)
    return "TARGET %s %d股" % (stock, target_amount), None


def format_rebalance_message(context, targets, target_weight, actions, skipped, scores):
    lines = []
    for i, stock in enumerate(targets, 1):
        ind = INDUSTRY_BY_STOCK.get(stock, "其他")
        lines.append("%d. %s %s %.2f%% score %.3f" % (i, stock, ind, target_weight * 100, scores.get(stock, np.nan)))
    return (
        "A股模拟盘调仓 %s\n"
        "版本: %s\n"
        "总资产: %.2f\n"
        "现金: %.2f\n"
        "候选/有效打分: %d/%d\n"
        "因子: %s\n"
        "目标仓位: %.0f%% / %d只 / 单票 %.2f%%\n\n"
        "目标持仓:\n%s\n\n"
        "订单:\n%s\n\n"
        "跳过/失败:\n%s"
    ) % (
        str(context.current_dt.date()),
        STRATEGY_VERSION,
        context.portfolio.total_value,
        context.portfolio.cash,
        g.last_candidate_count,
        g.last_score_count,
        ",".join(g.last_factor_names),
        MAX_EXPOSURE * 100,
        len(targets),
        target_weight * 100,
        "\n".join(lines) if lines else "无",
        "\n".join(actions) if actions else "无",
        "\n".join(skipped[:25]) if skipped else "无",
    )


# ============================ Feishu / Logging ============================

def send_feishu(text):
    if not FEISHU_WEBHOOK:
        return
    payload = {"msg_type": "text", "content": {"text": text[:3500]}}
    try:
        import requests
        r = requests.post(FEISHU_WEBHOOK, json=payload, timeout=5)
        if r.status_code >= 400:
            log_warning("Feishu webhook failed: %s %s", r.status_code, r.text[:200])
    except Exception as exc:
        log_warning("Feishu webhook exception: %s", exc)


def log_warning(message, *args):
    try:
        log.warning(message, *args)
    except Exception:
        log.warn(message, *args)
