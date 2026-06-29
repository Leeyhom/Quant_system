# -*- coding: utf-8 -*-
"""JoinQuant A-share simulation strategy v4 (SSOT-aligned).

This strategy matches the local v4 SSOT factor definition exactly, so that
the same factor set can be validated across: local backtest → RQAlpha →
JoinQuant simulated trading.

v4 factor set (all industry+size neutralized):
  earnings_yield    = 1/pe_ratio         (value, identical to v3)
  cashflow_yield    = 1/pcf_ratio        (value, identical to v3)
  sales_yield       = 1/ps_ratio         (value, identical to v3)
  growth_peg        = 1 / peg_ratio       (growth, M20 阶段6.5: 日频PEG倒数=增速/PE,
                                            覆盖94%>季报同比68%, OOS夏普0.79>0.62,
                                            概念对齐本地 SSOT growth_mode="peg")
  amihud_illiquidity = - mean(|ret| / money, 20d)    (liquidity, identical)
  holder_concentration = - holder_change_ratio        (NEW:筹码集中度, v4 only)

Parameters:
  TOP_N = 10         (v3 base; industry cap ensures diversification)
  INDUSTRY_CAP = 1   (max stocks per industry)
  REBALANCE_DAYS = 40
  MAX_EXPOSURE = 0.95
  MIN_COMMISSION = 5
  SLIPPAGE = 0.0005

Paste this file into JoinQuant strategy editor. Set initial capital to
60000 CNY.
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
# 筹码集中度开关：True=6因子(SSOT正式版)；False=纯5因子，用于与本地 --factor-set v5yoy
# 做最干净的口径对齐验证（聚宽与本地都是季报同比、都无筹码因子）。默认 True 向后兼容。
INCLUDE_HOLDER = False
FACTOR_NAMES = [
    "earnings_yield", "cashflow_yield", "sales_yield",
    "growth_peg", "amihud",
] + (["holder_concentration"] if INCLUDE_HOLDER else [])
STRATEGY_VERSION = "jq-cn-sim-v4-SSOT"

COMMISSION_RATE = 0.00025
STAMP_DUTY_RATE = 0.0005
EFFECTIVE_COMMISSION_RATE = COMMISSION_RATE


STOCK_POOL = [
    "000001.XSHE", "000002.XSHE", "000008.XSHE", "000027.XSHE", "000039.XSHE",
    "000063.XSHE", "000100.XSHE", "000157.XSHE", "000333.XSHE", "000338.XSHE",
    "000538.XSHE", "000568.XSHE", "000625.XSHE", "000651.XSHE", "000709.XSHE",
    "000725.XSHE", "000768.XSHE", "000776.XSHE", "000786.XSHE", "000792.XSHE",
    "000858.XSHE", "000876.XSHE", "000895.XSHE", "002007.XSHE", "002027.XSHE",
    "002142.XSHE", "002230.XSHE", "002236.XSHE", "002241.XSHE", "002252.XSHE",
    "002304.XSHE", "002352.XSHE", "002371.XSHE", "002415.XSHE", "002422.XSHE",
    "002460.XSHE", "002466.XSHE", "002475.XSHE", "002493.XSHE", "002594.XSHE",
    "002714.XSHE", "002821.XSHE", "002916.XSHE", "003816.XSHE", "300015.XSHE",
    "300017.XSHE", "300033.XSHE", "300124.XSHE", "300142.XSHE", "300274.XSHE",
    "300347.XSHE", "300413.XSHE", "300433.XSHE", "300498.XSHE", "300750.XSHE",
    "300760.XSHE", "600000.XSHG", "600016.XSHG", "600019.XSHG", "600028.XSHG",
    "600030.XSHG", "600031.XSHG", "600036.XSHG", "600048.XSHG", "600050.XSHG",
    "600061.XSHG", "600104.XSHG", "600111.XSHG", "600196.XSHG", "600276.XSHG",
    "600309.XSHG", "600346.XSHG", "600438.XSHG", "600519.XSHG", "600547.XSHG",
    "600570.XSHG", "600585.XSHG", "600588.XSHG", "600606.XSHG", "600660.XSHG",
    "600690.XSHG", "600745.XSHG", "600760.XSHG", "600795.XSHG", "600809.XSHG",
    "600837.XSHG", "600873.XSHG", "600887.XSHG", "600893.XSHG", "600900.XSHG",
    "600905.XSHG", "600918.XSHG", "600989.XSHG", "601009.XSHG", "601012.XSHG",
    "601018.XSHG", "601088.XSHG", "601111.XSHG", "601138.XSHG", "601166.XSHG",
    "601211.XSHG", "601225.XSHG", "601229.XSHG", "601238.XSHG", "601288.XSHG",
    "601298.XSHG", "601318.XSHG", "601319.XSHG", "601328.XSHG", "601336.XSHG",
    "601360.XSHG", "601377.XSHG", "601398.XSHG", "601601.XSHG", "601628.XSHG",
    "601633.XSHG", "601658.XSHG", "601668.XSHG", "601669.XSHG", "601688.XSHG",
    "601727.XSHG", "601766.XSHG", "601788.XSHG", "601800.XSHG", "601816.XSHG",
    "601818.XSHG", "601857.XSHG", "601877.XSHG", "601888.XSHG", "601898.XSHG",
    "601899.XSHG", "601919.XSHG", "601939.XSHG", "601985.XSHG", "601988.XSHG",
    "601989.XSHG", "601995.XSHG", "603259.XSHG", "603288.XSHG", "603501.XSHG",
    "603893.XSHG", "603986.XSHG", "688008.XSHG", "688009.XSHG", "688036.XSHG",
    "688111.XSHG", "688169.XSHG", "688256.XSHG", "688289.XSHG", "688303.XSHG",
    "688561.XSHG", "688981.XSHG",
]


INDUSTRY_MAP = {
    "银行": ["600000", "600016", "600036", "601009", "601166", "601229", "601288", "601318",
             "601328", "601398", "601658", "601818", "601838", "601939", "601988", "000001",
             "002142"],
    "非银金融": ["600030", "600061", "600109", "600369", "600837", "600918", "600958",
               "601066", "601108", "601136", "601162", "601198", "601211", "601236",
               "601336", "601360", "601375", "601377", "601456", "601555", "601601",
               "601628", "601688", "601788", "601878", "601881", "601901", "601995"],
    "食品饮料": ["600519", "000568", "000858", "600809", "002304", "600887", "603288",
               "000895", "002714", "002867", "600305", "600872", "603711"],
    "医药生物": ["600196", "600276", "002007", "002422", "300003", "300142", "300347",
               "300760", "603259", "000538", "300015", "002022", "002252"],
    "电子": ["000725", "002241", "002475", "300433", "603501", "600584", "600745",
             "002138", "300661", "688008"],
    "电气设备新能源": ["300274", "300750", "002460", "002466", "601012", "002129",
                     "002594", "601633", "000625", "601727", "600875"],
    "电力公用事业": ["600900", "601985", "600027", "600795", "000027", "000543",
                   "600642", "600863", "000875", "600011"],
    "家电": ["000333", "000651", "600690", "002508", "603868", "002242"],
    "汽车": ["600104", "601238", "000625", "601633", "002594"],
    "机械设备军工": ["600031", "000157", "600038", "600760", "000768", "600893",
                  "002013", "601989"],
    "建筑建材": ["601668", "601669", "601800", "000786", "600585"],
    "地产": ["000002", "600048", "001979", "600606"],
    "有色钢铁化工煤炭": ["600547", "601899", "000709", "600019", "600111", "600309",
                      "600792", "000792", "601857", "600028", "601088", "601225"],
    "交通运输物流": ["601018", "601111", "601816", "601766", "601919", "002352"],
    "农业食品养殖": ["000876", "300498", "002714", "600598"],
    "通信5G": ["000063", "600050", "600941", "300628"],
    "计算机传媒互联网": ["300033", "002230", "600570", "600588", "300413", "300773"],
    "光伏风电储能": ["601012", "002129", "300274", "300750"],
}


def _industry_by_stock():
    out = {}
    for ind, codes in INDUSTRY_MAP.items():
        for c in codes:
            out[c] = ind
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

    run_daily(rebalance, time="14:30")

    _holder_note = "，含筹码集中度" if INCLUDE_HOLDER else "，纯5因子口径对齐版（去筹码）"
    send_feishu(
        "A股聚宽模拟盘 v4 (SSOT 对齐版) 已启动\n"
        "版本: {}\n"
        "本金: 60000\n"
        "配置: {} 只 / 单行业最多 {} 只 / {} 交易日再平衡 / 总仓位 {:.0f}%\n"
        "因子: {} ({}因子)\n"
        "说明: 成长因子改用季报净利润同比/PE（聚宽与本地口径完全对齐）{}".format(
            STRATEGY_VERSION,
            TOP_N, INDUSTRY_CAP, REBALANCE_DAYS, MAX_EXPOSURE * 100,
            "+".join(FACTOR_NAMES), len(FACTOR_NAMES),
            _holder_note,
        )
    )


def rebalance(context):
    today = str(context.current_dt.date())
    g.days_since_rebalance += 1
    if g.days_since_rebalance < REBALANCE_DAYS:
        return

    candidates, blocked = filter_universe(STOCK_POOL)
    scores = build_scores_v4(context, candidates)

    target_weights, skipped = select_targets(context, scores)
    targets = list(target_weights.keys())

    actions = []
    current_positions = list(context.portfolio.positions.keys())

    for stock in current_positions:
        if stock not in targets and context.portfolio.positions[stock].total_amount > 0:
            if not can_sell(stock):
                skipped.append("{} 卖出失败: 停牌/跌停".format(stock))
                continue
            order_target(stock, 0)
            actions.append("SELL {} -> 0".format(stock))

    for stock in targets:
        target_weight = target_weights[stock]
        target_value = context.portfolio.total_value * target_weight
        if target_value > get_position_value(context, stock):
            if not can_buy(stock):
                skipped.append("{} 买入失败: 停牌/ST/涨停".format(stock))
                continue
        action, reason = order_target_value_round_lot(context, stock, target_value)
        if action:
            actions.append(action)
        elif reason:
            skipped.append(reason)

    g.days_since_rebalance = 0
    g.last_targets = targets

    msg = format_rebalance_message(context, targets, target_weights, actions, skipped, scores)
    # 日志格式必须与 analyze_joinquant_v4.parse_rebalance_log 的正则严格对齐：
    #   "v4 rebalance <date> ...factors=X candidates=N scored=N targets=CODE,CODE"
    # 否则导出的 v4_rebalance_targets.csv 会缺列/为空。
    log.info(
        "v4 rebalance {} factors={} candidates={} scored={} targets={}".format(
            today,
            ",".join(g.last_factor_names),
            len(candidates),
            len(scores),
            ",".join(targets),
        )
    )
    send_feishu(msg)


def build_scores_v4(context, stocks):
    """Build v4 SSOT scores: 6 factors, equal weight on ranks.

    All factors are industry+size neutralized, then ranked pct-wise,
    then equal average.
    """
    prev_date = context.previous_date
    val = get_fundamental_frame(stocks, prev_date)
    price_data = get_price(
        stocks,
        end_date=prev_date,
        count=120,
        frequency="daily",
        fields=["close", "money", "volume"],
        skip_paused=True,
        fq="pre",
        panel=False,
    )
    close = pivot_price(price_data, "close")
    money = pivot_price(price_data, "money")

    factors = {}
    pe = numeric_series(val, "pe_ratio")
    mv = numeric_series(val, "market_cap")

    factors["earnings_yield"] = (1.0 / pe.replace(0, np.nan)).where(pe > 0)

    pcf = first_numeric_series(val, ["pcf_ratio", "pcf"])
    if pcf is not None:
        factors["cashflow_yield"] = (1.0 / pcf.replace(0, np.nan)).where(pcf > 0)

    ps = first_numeric_series(val, ["ps_ratio", "ps"])
    if ps is not None:
        factors["sales_yield"] = (1.0 / ps.replace(0, np.nan)).where(ps > 0)

    # 成长因子 = 1 / PEG（M20 阶段6.5：日频 PEG 倒数，对齐本地 SSOT growth_mode="peg"）。
    # 1/PEG = 增速/PE，与季报同比/PE 概念同一，但聚宽日频 PEG 覆盖远高于季报增速字段。
    # 负/零 PEG（增速为负无意义）置 NaN，与本地 factors.growth_peg 逐字一致。
    peg = first_numeric_series(val, ["peg_ratio", "peg"])
    if peg is not None:
        factors["growth_peg"] = (1.0 / peg.replace(0, np.nan)).where(peg > 0)
    else:
        # 回退：日频 PEG 不可得时退回季报同比/PE（覆盖低但聊胜于无）
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
            factors["growth_peg"] = (growth.where(growth > 0) / pe.replace(0, np.nan)).where(pe > 0)

    if len(close) >= 21 and not money.empty:
        ret_abs = close.pct_change().abs()
        illiq = (ret_abs / money.replace(0, np.nan)).tail(20).mean()
        factors["amihud"] = -illiq

    # 6th factor: holder_concentration (筹码集中度, v4 only)
    # -1 * shareholders_change_rate (the more concentration, the higher the score)
    # 受 INCLUDE_HOLDER 开关控制：纯5因子口径对齐时跳过。
    if INCLUDE_HOLDER:
        holder_change = first_numeric_series(
            val,
            ["shareholders_0_ratio", "shareholder_change_ratio", "chg_ratio_shareholder",
             "holder_change_ratio", "holder_num_change_ratio"],
        )
        if holder_change is not None and holder_change.notna().any():
            factors["holder_concentration"] = -holder_change

    frame = pd.DataFrame(factors).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if frame.empty:
        return pd.Series(dtype=float)

    frame = neutralize_cross_section(frame, mv)
    g.last_factor_names = list(frame.columns)

    ranked = frame.rank(pct=True)
    score = ranked.mean(axis=1).dropna()
    # parser 期望 "v4 factor columns before scoring: ..."（注意是 columns）
    log.info("v4 factor columns before scoring: {}".format(",".join(frame.columns)))
    return score


def select_targets(context, scores, top_n=TOP_N, industry_cap=INDUSTRY_CAP):
    current_positions = list(context.portfolio.positions.keys())
    portfolio_value = context.portfolio.total_value
    counts = {}
    targets = {}
    skipped = []

    for stock in scores.sort_values(ascending=False).index:
        price = get_current_price(stock)
        if price is None or price <= 0:
            skipped.append("{} 备选跳过: 无价格".format(stock))
            continue
        code6 = stock[:6] if "." in stock else stock
        ind = INDUSTRY_BY_STOCK.get(code6, "其他")
        if counts.get(ind, 0) >= industry_cap:
            continue
        slot_value = portfolio_value * MAX_EXPOSURE / top_n
        if price * 100 > slot_value * 1.15:
            skipped.append("{} 备选跳过: 100股超过单票资金".format(stock))
            continue
        targets[stock] = MAX_EXPOSURE / top_n
        counts[ind] = counts.get(ind, 0) + 1
        if len(targets) >= top_n:
            break
    return targets, skipped


# ============================ Helper Functions ============================


def filter_universe(stocks):
    current = get_current_data()
    out = []
    skipped = []
    for stock in stocks:
        try:
            cd = current[stock]
        except Exception:
            continue
        if cd.paused:
            skipped.append("{} 停牌".format(stock[:6]))
            continue
        if cd.is_st:
            skipped.append("{} ST".format(stock[:6]))
            continue
        out.append(stock)
    return out, skipped


def numeric_series(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def first_numeric_series(df, names):
    for name in names:
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce")
            if s.notna().any():
                return s
    return None


def pivot_price(price_df, field):
    if price_df is None or len(price_df) == 0:
        return pd.DataFrame()
    df = price_df.copy()
    time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
    code_col = "code" if "code" in df.columns else "security"
    if time_col is None or code_col is None or field not in df.columns:
        return pd.DataFrame()
    return df.pivot(index=time_col, columns=code_col, values=field).sort_index()


def neutralize_cross_section(frame, market_cap):
    out = pd.DataFrame(index=frame.index)
    log_mv = np.log(market_cap.replace(0, np.nan))
    industries = pd.Series(
        {s: INDUSTRY_BY_STOCK.get(s[:6] if "." in s else s, "其他") for s in frame.index}
    )
    dummies = pd.get_dummies(industries)

    for name in frame.columns:
        y = pd.to_numeric(frame[name], errors="coerce")
        valid = y.notna() & log_mv.notna()
        if valid.sum() < 12:
            out[name] = y
            continue
        x_parts = [
            pd.Series(1.0, index=y.index, name="const"),
            log_mv.loc[y.index],
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


def order_target_value_round_lot(context, stock, target_value):
    """Target value, rounded to 100-share lots (A股整手约束)"""
    try:
        price = get_current_data()[stock].last_price
    except Exception:
        return None, "调整失败 {}: 无当前价格".format(stock)
    if price is None or price <= 0:
        return None, "调整失败 {}: 价格无效".format(stock)
    target_amount = int(math.floor(target_value / price / 100.0) * 100)
    try:
        current_amount = int(context.portfolio.positions[stock].total_amount)
    except Exception:
        current_amount = 0
    if target_amount <= 0:
        return None, "调整跳过 {}: 目标资金不足100股".format(stock)
    delta = target_amount - current_amount
    if abs(delta) < 100:
        return None, "调整跳过 {}: 变化不足100股".format(stock)

    order_target(stock, target_amount)
    return "TARGET {} {}股".format(stock, target_amount), None


def get_current_price(stock):
    try:
        return float(get_current_data()[stock].last_price)
    except Exception:
        return None


def get_fundamental_frame(stocks, date):
    """Fetch valuation + indicator data from JoinQuant's daily update tables"""
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

    # 成长因子改用日频 PEG 倒数（M20 阶段6.5：覆盖 94% > 季报同比 68%，OOS 更优，
    # 且 1/PEG = 增速/PE 概念与本地 SSOT growth_mode="peg" 完全对齐）。
    if hasattr(val_table, "peg_ratio"):
        fields.append(val_table.peg_ratio)

    # First try pcf/ps from valuation
    for field_name in ["pcf_ratio", "pcf", "ps_ratio", "ps", "turnover_ratio", "pc"]:
        if hasattr(val_table, field_name):
            fields.append(getattr(val_table, field_name))

    # Try growth/holder fields from indicator
    if ind_table is not None:
        for name in [
            "net_profit_growth_rate", "inc_net_profit_year_on_year",
            "operating_revenue_growth_rate", "inc_revenue_year_on_year",
            "shareholders_0_ratio", "shareholder_change_ratio",
        ]:
            if hasattr(ind_table, name):
                fields.append(getattr(ind_table, name))

    q = query(*fields).filter(val_table.code.in_(stocks))
    df = get_fundamentals(q, date=date)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.set_index("code")


def format_rebalance_message(context, targets, weights, actions, skipped, scores):
    lines = []
    for i, stock in enumerate(targets, 1):
        w = weights.get(stock, 0.0)
        sc = scores.loc[stock] if stock in scores.index else float("nan")
        lines.append("{}. {} 持仓 {:.2f}% score={:.3f}".format(i, stock, w * 100, sc))
    return (
        "A股模拟盘调仓 (v4 SSOT) {}\n"
        "总资产: {:.2f}\n"
        "现金: {:.2f}\n"
        "因子: {}\n"
        "目标仓位: {:.0f}% / {} 只\n\n"
        "目标持仓:\n{}\n\n"
        "交易:\n{}\n\n"
        "跳过/失败:\n{}"
    ).format(
        str(context.current_dt.date()),
        context.portfolio.total_value,
        context.portfolio.cash,
        ",".join(g.last_factor_names),
        MAX_EXPOSURE * 100,
        len(targets),
        "\n".join(lines[:20]) if lines else "无",
        "\n".join(actions) if actions else "无",
        "\n".join(skipped[:20]) if skipped else "无",
    )


def send_feishu(text):
    if not FEISHU_WEBHOOK:
        return
    import json
    import requests
    payload = {"msg_type": "text", "content": {"text": text[:3500]}}
    try:
        requests.post(FEISHU_WEBHOOK, json=payload, timeout=5)
    except Exception:
        pass
