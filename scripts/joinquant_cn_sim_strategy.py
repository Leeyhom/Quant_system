# -*- coding: utf-8 -*-
"""
JoinQuant A-share simulation strategy, generated from the local quant project.

Purpose:
    Run a conservative A-share paper-trading strategy directly on JoinQuant.
    The local project remains the research/backtest system; this file is the
    cloud execution adapter.

How to use:
    1. Create a JoinQuant simulated trading strategy.
    2. Paste this whole file into the strategy editor.
    3. Set initial capital to 60000 CNY.
    4. Run daily. The strategy rebalances at 14:30 every 60 trading days.

Execution policy:
    - Capital: 60000
    - Max gross exposure: 95%
    - Holdings: 6 stocks, equal target weights
    - Rebalance: every 60 trading days
    - Fee model: commission 2.5bp, min 5 CNY, stamp duty 5bp sell-side,
      transfer fee 0.1bp folded into both-side commission
    - If a stock is suspended, ST, limit-up for buy, limit-down for sell, or an
      order cannot be filled, cash is left idle. No fallback chasing.

Important:
    This is a JoinQuant-compatible approximation of the local CN strategy.  It
    uses JoinQuant's own data APIs and available fields, so factor values will
    not be byte-identical to the local akshare/RQAlpha research stack.
"""

import math
import json

import numpy as np
import pandas as pd


# ============================ User Config ============================

FEISHU_WEBHOOK = ""

INITIAL_CAPITAL = 60000
MAX_EXPOSURE = 0.95
TOP_N = 6
REBALANCE_DAYS = 60
TRADE_TIME = "14:30"
STRATEGY_VERSION = "jq-cn-sim-v1.1-diagnostic"
COMMISSION_RATE = 0.00025
STAMP_DUTY_RATE = 0.0005
TRANSFER_FEE_RATE = 0.00001
EFFECTIVE_COMMISSION_RATE = COMMISSION_RATE + TRANSFER_FEE_RATE

# Keep the pool aligned with quant/data/universe.py DEFAULT_POOL.
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


# ============================ JoinQuant Hooks ============================

def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)

    # Fee settings requested by user.
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=STAMP_DUTY_RATE,    # stamp duty, sell-side
            open_commission=EFFECTIVE_COMMISSION_RATE,
            close_commission=EFFECTIVE_COMMISSION_RATE,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )

    g.days_since_rebalance = REBALANCE_DAYS
    g.last_targets = []
    g.last_signal_date = None
    g.last_message = ""

    run_daily(rebalance, time=TRADE_TIME)
    run_daily(after_close_report, time="after_close")

    send_feishu(
        "A股聚宽模拟盘已启动\n"
        "版本: %s\n"
        "本金: 60000\n"
        "配置: 6只 / 60交易日再平衡 / 总仓位95%\n"
        "执行: 14:30 自动调仓，未成交留现金"
        % STRATEGY_VERSION
    )


def rebalance(context):
    today = str(context.current_dt.date())
    g.days_since_rebalance += 1

    if g.days_since_rebalance < REBALANCE_DAYS:
        log.info("No rebalance today: %s/%s", g.days_since_rebalance, REBALANCE_DAYS)
        return

    candidates, blocked_before_score = filter_universe(context, STOCK_POOL)
    scores = build_scores(context, candidates)
    g.last_candidate_count = len(candidates)
    g.last_score_count = len(scores)
    g.last_blocked_count = len(blocked_before_score)
    if scores.empty:
        msg = "[%s] 无有效候选股票，跳过调仓。" % today
        log_warning(msg)
        send_feishu(msg)
        return

    targets = list(scores.sort_values(ascending=False).head(TOP_N).index)
    target_weight = MAX_EXPOSURE / max(1, len(targets))

    actions = []
    skipped = list(blocked_before_score)

    current_positions = list(context.portfolio.positions.keys())

    # Sell names no longer in target.
    for stock in current_positions:
        pos = context.portfolio.positions[stock]
        if pos.total_amount <= 0:
            continue
        if stock not in targets:
            if not can_sell(stock):
                skipped.append("%s 卖出失败: 停牌或跌停" % stock)
                continue
            order = order_target_value(stock, 0)
            actions.append("SELL %s -> 0" % stock)

    # Buy/adjust target names.
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
        order = order_target_value(stock, target_value)
        actions.append("TARGET %s %.1f%%" % (stock, target_weight * 100))

    g.days_since_rebalance = 0
    g.last_targets = targets
    g.last_signal_date = today

    msg = format_rebalance_message(context, targets, target_weight, actions, skipped, scores)
    g.last_message = msg
    send_feishu(msg)


def after_close_report(context):
    """Send a compact daily status after close.

    Keep it lightweight.  On non-rebalance days, this is just account status;
    rebalance details are already sent at 14:30.
    """
    today = str(context.current_dt.date())
    pos_lines = []
    for stock, pos in context.portfolio.positions.items():
        if pos.total_amount > 0:
            pos_lines.append("%s %d股 市值%.0f" % (stock, pos.total_amount, pos.value))
    if not pos_lines:
        pos_lines = ["空仓"]

    text = (
        "A股模拟盘盘后状态 %s\n"
        "总资产: %.2f\n"
        "现金: %.2f\n"
        "持仓市值: %.2f\n"
        "调仓计数: %d/%d\n"
        "持仓:\n%s"
    ) % (
        today,
        context.portfolio.total_value,
        context.portfolio.cash,
        context.portfolio.positions_value,
        g.days_since_rebalance,
        REBALANCE_DAYS,
        "\n".join(pos_lines[:12]),
    )
    send_feishu(text)


# ============================ Factor Logic ============================

def filter_universe(context, stocks):
    current = get_current_data()
    candidates = []
    skipped = []
    for s in stocks:
        try:
            cd = current[s]
        except Exception:
            skipped.append("%s 跳过: 无当前行情" % s)
            continue
        name = getattr(cd, "name", "")
        if cd.paused:
            skipped.append("%s 跳过: 停牌" % s)
            continue
        if cd.is_st or ("ST" in name) or ("退" in name):
            skipped.append("%s 跳过: ST/退市风险" % s)
            continue
        candidates.append(s)
    return candidates, skipped


def build_scores(context, stocks):
    """Build a transparent multi-factor score using JoinQuant data.

    Factors:
        - earnings_yield: 1 / PE
        - book_to_price: 1 / PB
        - small_size: -log(market_cap)
        - reversal_20: -20d return
        - reversal_5: -5d return
        - amihud_20: -mean(abs(ret) / money)
    """
    if not stocks:
        return pd.Series(dtype=float)

    prev_date = context.previous_date

    q = query(
        valuation.code,
        valuation.pe_ratio,
        valuation.pb_ratio,
        valuation.market_cap,
    ).filter(valuation.code.in_(stocks))
    val = get_fundamentals(q, date=prev_date)
    if val is None or len(val) == 0:
        return pd.Series(dtype=float)
    val = val.set_index("code")

    price = get_price(
        stocks,
        end_date=prev_date,
        count=61,
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
    pe = pd.to_numeric(val.get("pe_ratio"), errors="coerce")
    pb = pd.to_numeric(val.get("pb_ratio"), errors="coerce")
    mv = pd.to_numeric(val.get("market_cap"), errors="coerce")

    factors["earnings_yield"] = (1.0 / pe.replace(0, np.nan)).where(pe > 0)
    factors["book_to_price"] = (1.0 / pb.replace(0, np.nan)).where(pb > 0)
    factors["small_size"] = -np.log(mv.replace(0, np.nan))

    latest = close.iloc[-1]
    if len(close) >= 21:
        factors["reversal_20"] = -(latest / close.iloc[-21] - 1.0)
    if len(close) >= 6:
        factors["reversal_5"] = -(latest / close.iloc[-6] - 1.0)
    if len(close) >= 21 and not money.empty:
        ret_abs = close.pct_change().abs()
        illiq = (ret_abs / money.replace(0, np.nan)).tail(20).mean()
        factors["amihud_20"] = -illiq

    frame = pd.DataFrame(factors)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if frame.empty:
        return pd.Series(dtype=float)

    # Cross-sectional percentile rank; higher score is better.
    ranked = frame.rank(pct=True)
    score = ranked.mean(axis=1).dropna()
    return score


def pivot_price(price_df, field):
    if price_df is None or len(price_df) == 0:
        return pd.DataFrame()
    df = price_df.copy()
    if "time" in df.columns:
        time_col = "time"
    elif "date" in df.columns:
        time_col = "date"
    else:
        time_col = df.columns[0]
    if "code" in df.columns:
        code_col = "code"
    elif "security" in df.columns:
        code_col = "security"
    else:
        # JoinQuant panel=False normally includes code; if not, fail softly.
        return pd.DataFrame()
    if field not in df.columns:
        return pd.DataFrame()
    out = df.pivot(index=time_col, columns=code_col, values=field)
    return out.sort_index()


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
    if price is not None and high_limit is not None and price >= high_limit:
        return False
    return True


def can_sell(stock):
    try:
        cd = get_current_data()[stock]
    except Exception:
        return False
    if cd.paused:
        return False
    price = getattr(cd, "last_price", None)
    low_limit = getattr(cd, "low_limit", None)
    if price is not None and low_limit is not None and price <= low_limit:
        return False
    return True


def get_position_value(context, stock):
    try:
        pos = context.portfolio.positions[stock]
    except Exception:
        return 0.0
    if pos is None or pos.total_amount <= 0:
        return 0.0
    return float(pos.value)


def format_rebalance_message(context, targets, target_weight, actions, skipped, scores):
    today = str(context.current_dt.date())
    target_lines = []
    for i, s in enumerate(targets, 1):
        target_lines.append("%d. %s 目标 %.2f%% 分数 %.3f" % (i, s, target_weight * 100, scores.get(s, np.nan)))

    msg = (
        "A股模拟盘调仓 %s\n"
        "总资产: %.2f\n"
        "现金: %.2f\n"
        "候选/有效打分/预过滤: %d/%d/%d\n"
        "目标仓位: %.0f%% / %d只 / 单票 %.2f%%\n\n"
        "目标持仓:\n%s\n\n"
        "订单:\n%s\n\n"
        "跳过/失败:\n%s"
    ) % (
        today,
        context.portfolio.total_value,
        context.portfolio.cash,
        getattr(g, "last_candidate_count", -1),
        getattr(g, "last_score_count", -1),
        getattr(g, "last_blocked_count", -1),
        MAX_EXPOSURE * 100,
        len(targets),
        target_weight * 100,
        "\n".join(target_lines) if target_lines else "无",
        "\n".join(actions) if actions else "无",
        "\n".join(skipped[:20]) if skipped else "无",
    )
    return msg


# ============================ Feishu ============================

def send_feishu(text):
    if not FEISHU_WEBHOOK:
        return
    payload = {
        "msg_type": "text",
        "content": {"text": text[:3500]},
    }
    try:
        import requests
        r = requests.post(FEISHU_WEBHOOK, json=payload, timeout=5)
        if r.status_code >= 400:
            log_warning("Feishu webhook failed: %s %s", r.status_code, r.text[:200])
    except Exception as e:
        log_warning("Feishu webhook exception: %s", e)


def log_warning(message, *args):
    try:
        log.warning(message, *args)
    except Exception:
        log.warn(message, *args)
