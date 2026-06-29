"""layered —— 因子分层回测（quantile/layered backtest）。

分层回测回答：因子分数从低到高的各层，未来收益是否也逐层变好？
这是检验因子排序能力和单调性的核心工具。

实现约定：
- 每个再平衡日，用上一交易日因子把股票分层，防未来函数。
- 每层内部等权。
- 对每层单独扣换手成本。
- top-bottom 仅作研究指标；A股做空约束下不代表可直接实盘。
- tradable_mask（M22）: 轻量执行约束，调仓日只从当日可交易的票里选。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.metrics import summary


def simple_tradable_mask(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    limit_up_pct: float = 0.0995,
    limit_down_pct: float = -0.0995,
) -> pd.DataFrame:
    """构造简易 A股可交易掩码（对齐聚宽轻量约束，M22）。

    判定（当日不可交易 = False）：
    1. 停牌 = 当日成交量 = 0 或价格缺失
    2. 涨停：当日涨跌幅 ≥ limit_up_pct（默认9.95%，近似10%涨停）
    3. 跌停：当日涨跌幅 ≤ limit_down_pct（默认-9.95%，近似10%跌停）

    ST/*ST 风险警示板过滤：本池是大盘龙头（默认池），ST 极少，故省略（如需
    可在 pool 层先剔除，或在本函数扩展）。

    返回: index=日期, columns=股票 的 bool 面板，True=当日可交易可进可出。
    """
    pct = close.pct_change().fillna(0.0)
    trading = (volume.fillna(0) > 0) & close.notna() & high.notna() & low.notna()
    not_limit_up = pct < limit_up_pct
    not_limit_down = pct > limit_down_pct
    return trading & not_limit_up & not_limit_down


def _layer_weights(scores: pd.Series, n_layers: int, tradable: pd.Series | None = None) -> dict[str, pd.Series]:
    """把一日因子分数分成 n_layers 层，返回每层等权权重。

    tradable：可选的当日可交易掩码（index=股票，bool）。传入则把不可交易
        （停牌/涨跌停/ST）的股票从分层池剔除——轻量执行约束（M22，对齐聚宽）。
        默认 None=不约束，向后兼容。注意只影响「当日新选入」，已不在池的票自然不持有。
    """
    valid = scores.dropna()
    if tradable is not None:
        # 只保留当日可交易的票参与分层（不可交易的票当日不被选入）。
        keep = tradable.reindex(valid.index).fillna(False)
        valid = valid[keep.astype(bool)]
    valid = valid.sort_values()
    cols = scores.index
    empty = {f"L{i}": pd.Series(0.0, index=cols) for i in range(1, n_layers + 1)}
    if len(valid) < n_layers:
        return empty

    # rank(method="first") 避免重复值导致 qcut 分箱失败。
    ranks = valid.rank(method="first")
    labels = list(range(1, n_layers + 1))
    buckets = pd.qcut(ranks, q=n_layers, labels=labels)

    weights = {}
    for layer in labels:
        w = pd.Series(0.0, index=cols)
        selected = buckets[buckets == layer].index
        if len(selected) > 0:
            w.loc[selected] = 1.0 / len(selected)
        weights[f"L{layer}"] = w
    return weights


def _layer_weights_risk_parity(
    scores: pd.Series,
    n_layers: int,
    ret_history: np.ndarray,
    stock_list: list,
    date_idx: int,
    lookback: int = 60,
) -> dict[str, pd.Series]:
    """用风险平价权重替代等权——仅对最高层(L{n_layers})做，其他层仍等权。

    参数:
        scores: 当日因子分数 Series（index=股票名）。
        n_layers: 分层数。
        ret_history: 历史收益矩阵 (n_dates × n_stocks)，用于估计协方差。
        stock_list: 股票列表（与 ret_history 的列序一致）。
        date_idx: 当前日期在 ret_history 中的索引。
        lookback: 协方差估计窗口（交易日）。

    返回与 _layer_weights 相同格式的 dict。
    """
    # 先用等权分层
    base = _layer_weights(scores, n_layers)
    top_key = f"L{n_layers}"
    top_stocks = base[top_key][base[top_key] > 0].index.tolist()

    if len(top_stocks) <= 2:
        return base  # 股票太少，风险平价无意义

    # 映射到收益矩阵的列索引
    col_map = {s: i for i, s in enumerate(stock_list)}
    top_cols = [col_map[s] for s in top_stocks if s in col_map]
    if len(top_cols) < 3:
        return base

    start_idx = max(0, date_idx - lookback)
    ret_window = ret_history[start_idx:date_idx, :][:, top_cols]
    ret_window = ret_window[~np.isnan(ret_window).any(axis=1)]
    if len(ret_window) < 20:
        return base

    cov = np.cov(ret_window, rowvar=False)
    cov = cov + np.eye(len(cov)) * 1e-6  # 正则化

    # 风险平价
    n = len(cov)
    w0 = np.ones(n) / n

    def _rc(w, C):
        sw = C @ w
        pv = np.sqrt(w @ sw)
        return w * sw / pv if pv > 1e-12 else np.ones(n) / n

    def _obj(w):
        rc = _rc(w, cov)
        return np.sum((rc - 1.0 / n) ** 2)

    try:
        from scipy.optimize import minimize
        max_w = min(0.15, max(0.05, 3.0 / n))  # 单票上限15%，不低于5%
        res = minimize(_obj, w0, method="SLSQP",
                       bounds=[(0, max_w)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                       options={"maxiter": 100, "ftol": 1e-10})
        if res.success:
            rp_w = np.maximum(res.x, 0)
            rp_w = rp_w / rp_w.sum()
            new_top = pd.Series(0.0, index=scores.index)
            for j, s in enumerate(top_stocks):
                new_top[s] = rp_w[j]
            base[top_key] = new_top
    except Exception:
        pass  # 优化失败则退回等权

    return base


def layered_backtest(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    n_layers: int = 5,
    rebalance_every: int = 20,
    cost_rate: float = 0.001,
    first_rebalance: bool = False,
    cost_fn=None,
    weight_mode: str = "equal",
    tradable_mask: pd.DataFrame | None = None,
) -> dict:
    """运行因子分层回测。

    cost_fn：可选的费用回调（默认 None=用比例成本 cost_rate×换手，向后兼容 A股口径）。
        若传入，签名为 cost_fn(weight_change: Series, prices: Series) -> fractional_cost，
        在每个再平衡日按各层权重变动+当日价格算「收益拖累比例」，替代比例成本。
        用于美股「每股费+每笔最低费」模型（见 quant/backtest/us_cost.py）。

    tradable_mask：可选「日期 × 股票」bool 面板（M22 轻量执行约束，对齐聚宽）。
        再平衡日只从当日可交易（True）的股票里分层选股，剔除停牌/涨跌停/ST。
        默认 None=不约束，向后兼容（结果与改动前完全一致）。

    返回 dict：
        returns: 各层日收益 + top_bottom
        equity: 各层净值 + top_bottom
        turnover: 各层换手率
    """
    close = close.sort_index()
    factor = factor.reindex_like(close)
    if tradable_mask is not None:
        tradable_mask = tradable_mask.reindex(index=close.index, columns=close.columns)
    ret = close.pct_change().fillna(0.0)
    ret_vals = ret.values
    stocks_list = close.columns.tolist()
    layer_names = [f"L{i}" for i in range(1, n_layers + 1)]

    current = {name: pd.Series(0.0, index=close.columns) for name in layer_names}
    turnover = pd.DataFrame(0.0, index=close.index, columns=layer_names)
    returns = pd.DataFrame(0.0, index=close.index, columns=layer_names)
    # 当 cost_fn 注入时，按再平衡日记录每层的比例费用拖累，跟 turnover 分开存。
    fee_drag = pd.DataFrame(0.0, index=close.index, columns=layer_names)

    for i, date in enumerate(close.index):
        should_rebalance = i > 0 and (
            ((i - 1) % rebalance_every == 0) if first_rebalance else (i % rebalance_every == 0)
        )
        if should_rebalance:
            # 上一交易日因子（防未来函数）；可交易掩码用「上一交易日」状态，与因子同口径。
            tradable_prev = None if tradable_mask is None else tradable_mask.iloc[i - 1]
            if weight_mode == "risk_parity":
                target = _layer_weights_risk_parity(
                    factor.iloc[i - 1], n_layers, ret_vals, stocks_list, i)
            else:
                target = _layer_weights(factor.iloc[i - 1], n_layers, tradable=tradable_prev)
            for name in layer_names:
                weight_change = (target[name] - current[name]).abs()
                turnover.loc[date, name] = weight_change.sum()
                if cost_fn is not None:
                    fee_drag.loc[date, name] = cost_fn(weight_change, close.iloc[i])
                current[name] = target[name]

        for name in layer_names:
            gross = (current[name] * ret.loc[date]).sum()
            if cost_fn is not None:
                cost = fee_drag.loc[date, name]
            else:
                cost = turnover.loc[date, name] * cost_rate
            returns.loc[date, name] = gross - cost

    top = f"L{n_layers}"
    bottom = "L1"
    returns["top_bottom"] = returns[top] - returns[bottom]
    equity = (1.0 + returns).cumprod()
    return {"returns": returns, "equity": equity, "turnover": turnover}


def layer_summary(result: dict, n_layers: int = 5) -> pd.DataFrame:
    """汇总每层绩效，并给出简单单调性判断所需表格。"""
    rows = []
    returns = result["returns"]
    equity = result["equity"]
    for name in [f"L{i}" for i in range(1, n_layers + 1)] + ["top_bottom"]:
        m = summary(equity[name], returns[name])
        rows.append({"layer": name, **m})
    return pd.DataFrame(rows)


def is_monotonic_by_return(summary_df: pd.DataFrame, n_layers: int = 5) -> bool:
    """按总收益粗略判断 L1->Ln 是否单调递增。"""
    layer_ret = summary_df[summary_df["layer"].str.startswith("L")].set_index("layer")[
        "total_return"
    ]
    vals = [layer_ret[f"L{i}"] for i in range(1, n_layers + 1)]
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def long_top_layer(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    n_layers: int = 5,
    rebalance_every: int = 20,
    cost_rate: float = 0.001,
    first_rebalance: bool = False,
    cost_fn=None,
    weight_mode: str = "equal",
    tradable_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """分层多头组合：持有因子最高层（L{n_layers}），与等权基准对比。

    为什么这样而非固定 top-N（见 docs/12）：A股做空约束下多空对冲（L5-L1）
    收益≈0，只能靠纯多头。直接复用 `layered_backtest` 的最高层，避免另写一套
    选股逻辑；分层按截面分位数取「最好的一档」，比固定 top-N 在不同池子大小下
    更稳健。

    cost_fn：可选费用回调（默认 None=比例成本，向后兼容）；传入则透传给
        layered_backtest，用于美股「每股费+每笔最低费」模型（见 us_cost.py）。

    返回:
        DataFrame，列与 run_factor_portfolio 对齐（port_ret/equity/benchmark_ret/
        benchmark/turnover），便于接入 portfolio_validation 的 OOS 框架。
    """
    res = layered_backtest(
        close, factor,
        n_layers=n_layers,
        rebalance_every=rebalance_every,
        cost_rate=cost_rate,
        first_rebalance=first_rebalance,
        cost_fn=cost_fn,
        weight_mode=weight_mode if "weight_mode" in dir() else "equal",
        tradable_mask=tradable_mask,
    )
    top = f"L{n_layers}"
    port_ret = res["returns"][top]
    turnover = res["turnover"][top]

    # 等权基准：持有所有当日有价格的股票（与 run_factor_portfolio 口径一致）
    ret = close.sort_index().pct_change().fillna(0.0)
    available = close.notna().astype(float)
    bench_w = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    bench_ret = (bench_w * ret).sum(axis=1)

    out = pd.DataFrame(index=close.index)
    out["port_ret"] = port_ret
    out["turnover"] = turnover
    out["equity"] = (1.0 + port_ret).cumprod()
    out["benchmark_ret"] = bench_ret
    out["benchmark"] = (1.0 + bench_ret).cumprod()
    return out


def score_weighted_portfolio(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    top_frac: float = 0.20,
    rebalance_every: int = 20,
    cost_rate: float = 0.001,
    first_rebalance: bool = False,
    cost_fn=None,
) -> pd.DataFrame:
    """因子分数加权组合：选因子分最高的 top_frac 股票，按分数配权重。

    为什么不用等权分层（L5）而用分数加权：
      L5 层内股票分数可能差异很大（从 p80 到 p100），等权把最优质和次优质
      一视同仁，浪费了信号。分数加权按 z-score 配权重，信号越强的股票占用
      越多资金，理论上信息利用率更高。

    参数:
        top_frac: 选因子分最高的前多少比例（默认 0.20 = 前 20%）。
        其余同 layered_backtest。

    返回:
        DataFrame，列与 long_top_layer 对齐。
    """
    close = close.sort_index()
    factor = factor.reindex_like(close)
    ret = close.pct_change().fillna(0.0)

    port_ret = pd.Series(0.0, index=close.index)
    turnover = pd.Series(0.0, index=close.index)
    current_w = pd.Series(0.0, index=close.columns)

    for i, date in enumerate(close.index):
        should_rebalance = i > 0 and (
            ((i - 1) % rebalance_every == 0) if first_rebalance else (i % rebalance_every == 0)
        )
        if should_rebalance:
            scores = factor.iloc[i - 1].dropna()
            if len(scores) > 0:
                # 选 top_frac 最高分
                n_select = max(1, int(len(scores) * top_frac))
                selected = scores.nlargest(n_select)
                # z-score 归一化：只用正的部分做权重
                z = (selected - selected.mean()) / (selected.std() + 1e-12)
                z_pos = z.clip(lower=0) + 0.01  # 保底 0.01 让所有选中票都有权重
                target_w = pd.Series(0.0, index=close.columns)
                target_w.loc[z_pos.index] = z_pos / z_pos.sum()
            else:
                target_w = pd.Series(0.0, index=close.columns)

            weight_change = (target_w - current_w).abs()
            turnover.loc[date] = weight_change.sum()

            if cost_fn is not None:
                fee_drag = cost_fn(weight_change, close.iloc[i])
                cost = fee_drag
            else:
                cost = turnover.loc[date] * cost_rate

            current_w = target_w
        else:
            cost = 0.0

        gross = (current_w * ret.loc[date]).sum()
        port_ret.loc[date] = gross - cost

    # 等权基准
    available = close.notna().astype(float)
    bench_w = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    bench_ret = (bench_w * ret).sum(axis=1)

    out = pd.DataFrame(index=close.index)
    out["port_ret"] = port_ret
    out["turnover"] = turnover
    out["equity"] = (1.0 + port_ret).cumprod()
    out["benchmark_ret"] = bench_ret
    out["benchmark"] = (1.0 + bench_ret).cumprod()
    return out


def fixed_topn_portfolio(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    top_n: int = 8,
    rebalance_every: int = 20,
    cost_rate: float = 0.001,
    first_rebalance: bool = False,
    cost_fn=None,
) -> pd.DataFrame:
    """固定持仓数的等权纯多头组合：每次只持有因子分最高的 top_n 只，等权。

    为什么需要它（区别于 long_top_layer / score_weighted_portfolio）：
      那两个都是「比例制」——分层 L5 或 top_frac，池子越大持仓越多（扩展池里
      可能几十~上百只）。但小本金实盘（如 1w 美元）下，$1/笔固定费会被持仓数
      放大：持仓越多、每只越小额，固定费占比越高。实盘必须能表达「就持有 N 只」，
      并据此评估真实费用。本函数就持有固定 N 只，便于和 us_cost 的 $1/笔模型联用。

    防未来函数：再平衡日用 factor.iloc[i-1]（今算明用），与 layered_backtest 同口径。

    参数:
        top_n: 固定持仓数（等权 1/n）。
        cost_fn: 可选费用回调（默认 None=比例成本，向后兼容）；签名
            cost_fn(weight_change: Series, prices: Series) -> fractional_cost，
            用于美股「每股费+每笔最低费」模型（见 quant/backtest/us_cost.py）。
        其余同 score_weighted_portfolio。

    返回:
        DataFrame，列与 long_top_layer / score_weighted_portfolio 对齐
        （port_ret/turnover/equity/benchmark_ret/benchmark），可直接接入 walk-forward。
    """
    close = close.sort_index()
    factor = factor.reindex_like(close)
    ret = close.pct_change().fillna(0.0)

    port_ret = pd.Series(0.0, index=close.index)
    turnover = pd.Series(0.0, index=close.index)
    current_w = pd.Series(0.0, index=close.columns)

    for i, date in enumerate(close.index):
        should_rebalance = i > 0 and (
            ((i - 1) % rebalance_every == 0) if first_rebalance else (i % rebalance_every == 0)
        )
        if should_rebalance:
            scores = factor.iloc[i - 1].dropna()  # 用昨日因子选股，防未来函数
            target_w = pd.Series(0.0, index=close.columns)
            if len(scores) > 0:
                selected = scores.nlargest(min(top_n, len(scores))).index
                target_w.loc[selected] = 1.0 / len(selected)  # 等权 1/n

            weight_change = (target_w - current_w).abs()
            turnover.loc[date] = weight_change.sum()
            if cost_fn is not None:
                cost = cost_fn(weight_change, close.iloc[i])
            else:
                cost = turnover.loc[date] * cost_rate
            current_w = target_w
        else:
            cost = 0.0

        gross = (current_w * ret.loc[date]).sum()
        port_ret.loc[date] = gross - cost

    # 等权基准：持有所有当日有价格的股票（与项目其它回测同口径）
    available = close.notna().astype(float)
    bench_w = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    bench_ret = (bench_w * ret).sum(axis=1)

    out = pd.DataFrame(index=close.index)
    out["port_ret"] = port_ret
    out["turnover"] = turnover
    out["equity"] = (1.0 + port_ret).cumprod()
    out["benchmark_ret"] = bench_ret
    out["benchmark"] = (1.0 + bench_ret).cumprod()
    return out


def vol_targeted(equity_df: pd.DataFrame, target_vol: float = 0.15, lookback: int = 60,
                 max_leverage: float = 2.0) -> pd.DataFrame:
    """波动率目标叠加：当近期波动率高时降低敞口，波动率低时放大敞口。

    为什么：2020年3月的 -36.1% 回撤来自系统性波动率飙升。在波动率高的时期
    主动降仓，可以显著压低尾部风险。目标年化波动率默认 15%（与美股长期均值接近）。

    参数:
        equity_df: 含 port_ret 列的 DataFrame（来自 long_top_layer / score_weighted_portfolio
            / fixed_topn_portfolio）。
        target_vol: 目标年化波动率（默认 15%）。
        lookback: 波动率估算回溯窗口（交易日）。
        max_leverage: 缩放上界（默认 2.0=允许 2 倍杠杆，向后兼容）。
            小资金无融资场景传 1.0——只在高波动时【降仓】、低波动时最多满仓，不加杠杆。

    防未来函数：缩放用 scale.shift(1)（昨天及更早的波动率决定今天敞口），无未来信息。

    返回:
        新 DataFrame，port_ret 和 equity 已做波动率缩放。
    """
    df = equity_df.copy()
    port_ret = df["port_ret"]

    # 滚动波动率（年化）
    rolling_vol = port_ret.rolling(lookback).std() * np.sqrt(252)
    # 缩放因子：目标波动率 / 实际波动率，上界 max_leverage、下界 0.1
    scale = (target_vol / rolling_vol.replace(0, np.nan)).clip(upper=max_leverage, lower=0.1)
    scale = scale.fillna(1.0)  # 初期无数据时用 1x

    # 缩放日收益并重建净值
    df["port_ret_scaled"] = port_ret * scale.shift(1).fillna(1.0)
    df["port_ret"] = df["port_ret_scaled"]
    df["equity"] = (1.0 + df["port_ret"]).cumprod()
    return df


def drawdown_brake(equity_df: pd.DataFrame, dd_trigger: float = 0.15,
                   reduced_exposure: float = 0.3) -> pd.DataFrame:
    """移动回撤刹车：组合净值从峰值回撤超过阈值时主动降仓，回到峰值附近再恢复满仓。

    为什么比 vol_targeted 反应快：vol-target 要等「波动率滚动窗口」升上来才降仓，对突发
    下跌反应慢一两天；本刹车直接看「当前净值距历史峰值的回撤」，回撤一旦破阈值【当根】即触发，
    且只在真回撤时降仓、上涨段保持满仓——实测在美股上夏普↑且回撤↓（不像 vol-target 牺牲收益）。

    参数:
        equity_df: 含 port_ret 列的 DataFrame（来自 fixed_topn_portfolio / long_top_layer 等）。
        dd_trigger: 回撤触发阈值（默认 0.15=从峰值回撤超 15% 就降仓）。
        reduced_exposure: 触发后保留的仓位比例（默认 0.3=降到三成仓，留七成现金）。

    防未来函数：用 scale.shift(1) 应用——今天的降仓只依据【昨天及更早】的回撤，无未来信息。

    返回:
        新 DataFrame，port_ret 和 equity 已按回撤刹车缩放。
    """
    df = equity_df.copy()
    port_ret = df["port_ret"]

    # 用「未刹车」的净值算回撤：峰值与回撤都只看历史，cummax 不含未来
    equity = (1.0 + port_ret).cumprod()
    dd = 1.0 - equity / equity.cummax()
    scale = pd.Series(1.0, index=port_ret.index)
    scale[dd > dd_trigger] = reduced_exposure  # 回撤破阈值→降仓

    df["port_ret"] = port_ret * scale.shift(1).fillna(1.0)  # shift(1) 防未来
    df["equity"] = (1.0 + df["port_ret"]).cumprod()
    return df


def long_short_portfolio(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    n_layers: int = 5,
    rebalance_every: int = 20,
    cost_rate: float = 0.001,
    first_rebalance: bool = False,
    cost_fn=None,
    borrow_rate: float = 0.02,
) -> pd.DataFrame:
    """多空组合：做多因子最高层(L{n_layers})，做空因子最低层(L1)，市值中性。

    为什么要多空（对症 M17「方向先验漂移」诊断）：
      单边多头依赖「因子方向先验正确」。当池子构成变化导致价值/成长相对强弱
      漂移时，多头可能押错方向。多空对冲掉市场 beta——只要因子排序能力真实
      （IC>0），L5 跑赢 L1 就能赚钱，不受方向漂移影响。

    借券成本模型（美股的现实约束）：
      - borrow_rate：年化借券费率，默认 2%（覆盖多数美股；易借大盘 ~0.3%、
        难借小盘 ~5-20%，取中偏保守）。
      - 每日扣 borrow_rate/252 在空头腿上。
      - 不建模卖空保证金利息（各券商差异大，且当前利率环境下可正可负）。

    返回 DataFrame（与 long_top_layer 对齐）：
      port_ret: 多空组合日收益（已扣双边交易成本 + 借券费）
      equity: 多空净值
      long_ret: 纯多头腿日收益
      short_ret: 纯空头腿日收益（做空 L1 的收益 = -L1_return）
      benchmark_ret/benchmark: 等权全持有基准（仅作参考，多空不应与多头基准比）
    """
    res = layered_backtest(
        close, factor,
        n_layers=n_layers,
        rebalance_every=rebalance_every,
        cost_rate=cost_rate,
        first_rebalance=first_rebalance,
        cost_fn=cost_fn,
        weight_mode=weight_mode if "weight_mode" in dir() else "equal",
    )
    top = f"L{n_layers}"
    bottom = "L1"

    long_ret = res["returns"][top]       # L5 多头日收益（已扣交易成本）
    l1_ret = res["returns"][bottom]      # L1 多头日收益（已扣交易成本）
    short_ret = -l1_ret                  # 做空 L1 = 反向持有
    borrow_daily = borrow_rate / 252     # 日借券费

    # 多空日收益 = 多头腿 - 空头腿（注意：做空赚钱时 l1_ret<0，-l1_ret>0）
    # 再扣借券费
    port_ret = long_ret + short_ret - borrow_daily

    # 等权基准（与 long_top_layer 口径一致）
    ret = close.sort_index().pct_change().fillna(0.0)
    available = close.notna().astype(float)
    bench_w = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    bench_ret = (bench_w * ret).sum(axis=1)

    out = pd.DataFrame(index=close.index)
    out["port_ret"] = port_ret
    out["long_ret"] = long_ret
    out["short_ret"] = short_ret
    out["turnover"] = res["turnover"][top] + res["turnover"][bottom]
    out["equity"] = (1.0 + port_ret).cumprod()
    out["benchmark_ret"] = bench_ret
    out["benchmark"] = (1.0 + bench_ret).cumprod()
    return out
