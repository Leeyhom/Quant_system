"""composite —— 多因子 IC 加权合成（M13）。

动机（见 docs/13）：M12 证明单因子即使中性化也躲不过 regime 依赖
（2025Q3 价值因子单季 IC 转负）。把多个**已中性化、OOS 同号、彼此低相关**
的因子合成一个组合因子，能分散单因子的 regime 风险——某个分量在某段行情
失灵时，其它分量顶上，合成 IC 的波动下降、ICIR 上升。

与 M9 `combine_factors`（等权 rank 平均）的区别：
- 等权：每个因子同样可信。
- IC 加权：权重正比于因子的历史预测力（|IC|），强因子说话更响。

防前视铁律（最易作弊处）：算权重的 IC **只能用切分日之前的数据**，并裁掉
末尾 horizon 天的重叠（见 docs/10）。本模块直接复用 M9 已验证的 train IC
口径与 `orient_by_ic` 定向，方向（IC 符号）与权重大小（|IC|）共享同一段
train IC，天然一致。

函数分两代（向后兼容）：
- `ic_weighted_composite`：M13 基线，单切分点 |IC| 加权。
- `weighted_composite`：M13 打磨，支持 ICIR 加权（除 stdIC 降噪）+ 多切分点
  平均带符号权重（方向不稳的因子被相消收缩）。见 docs/13 第七节。
"""
from __future__ import annotations

import pandas as pd

from quant.backtest.ic_analysis import daily_ic, ic_summary
from quant.backtest.factor_validation import orient_by_ic


def factor_correlation(
    factors: dict[str, pd.DataFrame], method: str = "spearman"
) -> pd.DataFrame:
    """因子两两相关矩阵，用于合成前去冗余（见 docs/13 第一节）。

    高相关的因子（如 1/PE、1/PB、1/PS 三个价值因子常 >0.7）合在一起只是把
    同一个赌注下多遍，分散不了 regime 风险还稀释正交信息，应只留代表。

    做法：把每个因子先做横截面 rank（与合成口径一致、去量纲），逐日展平成
    一个长向量，再算因子间相关。相关基于 rank 值，等价于 Spearman。

    返回:
        DataFrame，行列均为因子名，值为相关系数（对角线为 1）。
    """
    names = list(factors.keys())
    # 各因子横截面 rank 后展平成一维序列（对齐到相同的 (日期,股票) 索引）。
    # stack() 默认丢弃缺失格，DataFrame 按公共 (日期,股票) 索引对齐后算相关。
    series = {name: factors[name].rank(axis=1, pct=True).stack() for name in names}
    mat = pd.DataFrame(series)
    # method 仅区分是否在已 rank 的数据上再算 pearson；rank 后 pearson≈spearman。
    return mat.corr(method="pearson" if method == "spearman" else method)


def ic_weighted_composite(
    factors: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    upto_date,
    horizon: int = 20,
    method: str = "spearman",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """用 upto_date 之前（train 段）的 IC 给每个分量定向 + 定权，再加权合成。

    参数:
        factors:   {名称: 因子面板}，待合成的分量（**调用方应先各自中性化**）。
        fwd_ret:   未来收益面板，用于算各分量的 train 段 IC。
        upto_date: train/test 切分日期。**只用此日期之前的 IC**算权重（防前视）。
        horizon:   未来收益跨度，用于裁掉 train 末尾的重叠天。

    返回:
        (合成因子面板, {分量名: 该分量 train 段 mean_ic})
        合成 = Σ_k weight[k] · rank(定向后的因子k)，
        其中 weight[k] = |IC[k]| / Σ_j|IC[j]|（IC 为 0 的因子权重为 0）。

    与 build_oriented_composite 的差别只在权重：那个是等权 rank 平均，
    这个按 |IC| 加权。方向逻辑（orient_by_ic）完全一致。
    """
    if not factors:
        raise ValueError("至少传入一个因子")

    # 只取切分日期之前的未来收益算 train IC；再裁末尾 horizon 天避免探入 test。
    fwd_train = fwd_ret.loc[fwd_ret.index < upto_date]
    if len(fwd_train) > horizon:
        fwd_train = fwd_train.iloc[:-horizon]

    signs: dict[str, float] = {}
    oriented_ranks: dict[str, pd.DataFrame] = {}
    for name, fac in factors.items():
        ic = daily_ic(fac, fwd_train, method=method, min_count=5)
        mean_ic = ic_summary(ic)["mean_ic"]
        mean_ic = float(mean_ic) if pd.notna(mean_ic) else 0.0
        signs[name] = mean_ic
        # 先按 IC 符号定向（越高越好），再做横截面 rank 去量纲。
        oriented_ranks[name] = orient_by_ic(fac, mean_ic).rank(axis=1, pct=True)

    # 权重 = |train IC| 归一化。全为 0（无预测力）则退化为等权，避免除零。
    abs_ic = {name: abs(s) for name, s in signs.items()}
    total = sum(abs_ic.values())
    if total > 0:
        weights = {name: w / total for name, w in abs_ic.items()}
    else:
        n = len(factors)
        weights = {name: 1.0 / n for name in factors}

    composite = None
    for name, ranked in oriented_ranks.items():
        term = ranked * weights[name]
        composite = term if composite is None else composite + term
    return composite, signs


# ───────────────────────── ICIR / 多切分点加权（M13 增量）─────────────────────────
# M13 的朴素 |IC| 单点加权暴露了两个弱点（见 docs/13 第六节）：
#   ① 权重方向只看单一切分日前的 IC 符号，IC 在切分点附近变号就押反；
#   ② meanIC 大但飘忽的因子被高估（没惩罚波动）。
# 下面两招针对性修：
#   - ICIR 加权：权重 ∝ meanIC/stdIC，惩罚"均值大但飘"的因子，降低对单段行情的敏感。
#   - 多切分点平均：在切分日前取多个 expanding 切分点各算一版"带符号"权重再平均。
#     带符号是关键——某因子若在不同切分点 IC 变号，正负相消，其平均权重趋于 0，
#     等于自动把"方向不稳"的因子收缩掉，而不是盲目押某一段的方向。


def _train_ic_stats(
    factors: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    cut_date,
    horizon: int,
    method: str,
) -> dict[str, tuple[float, float]]:
    """算每个因子在 cut_date 之前（裁掉末尾 horizon 重叠）的 (meanIC, stdIC)。

    防前视：只用 cut_date 之前的未来收益，且再裁 horizon 天，与 docs/10 口径一致。
    """
    fwd_train = fwd_ret.loc[fwd_ret.index < cut_date]
    if len(fwd_train) > horizon:
        fwd_train = fwd_train.iloc[:-horizon]
    stats: dict[str, tuple[float, float]] = {}
    for name, fac in factors.items():
        s = ic_summary(daily_ic(fac, fwd_train, method=method, min_count=5))
        mean_ic = float(s["mean_ic"]) if pd.notna(s["mean_ic"]) else 0.0
        std_ic = float(s["std_ic"]) if pd.notna(s["std_ic"]) else 0.0
        stats[name] = (mean_ic, std_ic)
    return stats


def _signed_raw_weight(mean_ic: float, std_ic: float, scheme: str) -> float:
    """单切分点的带符号原始权重（未归一化）。

    scheme="ic":   meanIC                （M13 基线）
    scheme="icir": meanIC / stdIC        （惩罚波动；std≤0 退化为 0，视作无信息）
    符号即因子方向：正=越高越好，负=越低越好。
    """
    if scheme == "ic":
        return mean_ic
    if scheme == "icir":
        return mean_ic / std_ic if std_ic and std_ic > 0 else 0.0
    raise ValueError(f"未知 scheme: {scheme}")


def weighted_composite(
    factors: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    upto_date,
    scheme: str = "icir",
    n_cuts: int = 3,
    horizon: int = 20,
    method: str = "spearman",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """ICIR / 多切分点加权合成（M13 增量，防前视）。

    参数:
        factors:   {名称: 因子面板}（调用方应先各自中性化）。
        fwd_ret:   未来收益面板。
        upto_date: 最终切分日。所有切分点都 < upto_date，权重只用过去信息。
        scheme:    "ic"=|meanIC| 加权（基线）；"icir"=meanIC/stdIC 加权（默认，降噪）。
        n_cuts:    切分点个数。1=单点（退化为原 ic/icir 单点加权）；
                   >1=在 [50%,100%) 的 train 区间均匀取多个 expanding 切分点，
                   各算一版带符号权重再平均（方向不稳的因子被相消收缩）。
        horizon:   未来收益跨度，用于裁重叠。

    返回:
        (合成因子面板, {分量名: 最终带符号归一化权重})
        合成 = Σ_k w_k · rank(因子k)，w_k 已带方向（负权重等价于反向该因子）。

    为什么用「带符号权重 × rank」而非「定向后 rank × |权重|」：二者只差一个
    每日全体相同的常数偏移（rank(-x)=1-rank(x)），不影响横截面选股，但前者
    让"多切分点相消"在数值上自然成立。
    """
    if not factors:
        raise ValueError("至少传入一个因子")
    names = list(factors.keys())

    # 1) 确定切分点：单点就用 upto_date；多点在 train 区间后半段均匀取 expanding 切分。
    train_idx = fwd_ret.index[fwd_ret.index < upto_date]
    if n_cuts <= 1 or len(train_idx) < 2:
        cut_dates = [upto_date]
    else:
        lo = len(train_idx) // 2  # 至少用一半数据才算 IC，避免早期样本太短
        positions = [
            int(round(lo + (len(train_idx) - 1 - lo) * i / (n_cuts - 1)))
            for i in range(n_cuts)
        ]
        # 末位用 upto_date 本身，其余用对应 train 日期；去重保序。
        cut_dates = []
        for p in positions[:-1]:
            d = train_idx[p]
            if d not in cut_dates:
                cut_dates.append(d)
        cut_dates.append(upto_date)

    # 2) 各切分点算带符号原始权重，跨切分点平均（带符号→方向不稳者相消）。
    acc = {name: 0.0 for name in names}
    for cd in cut_dates:
        stats = _train_ic_stats(factors, fwd_ret, cd, horizon, method)
        for name in names:
            acc[name] += _signed_raw_weight(*stats[name], scheme)
    raw = {name: acc[name] / len(cut_dates) for name in names}

    # 3) 按 |权重| 归一化（保留符号）。全 0 则退化为等权（无方向偏好）。
    total = sum(abs(v) for v in raw.values())
    if total > 0:
        weights = {name: v / total for name, v in raw.items()}
    else:
        weights = {name: 1.0 / len(names) for name in names}

    composite = None
    for name in names:
        term = factors[name].rank(axis=1, pct=True) * weights[name]
        composite = term if composite is None else composite + term
    return composite, weights

