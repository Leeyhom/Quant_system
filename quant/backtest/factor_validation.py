"""factor_validation —— 因子层面的样本外验证 + 合成定向工具。

区别于 `portfolio_validation`（组合层 OOS 收益），本模块回答更本质的问题：
    **因子本身的预测力（IC）在样本外是否还在、是否同号？**

为什么需要它（见 docs/10）：
- IC 在某段样本里显著，可能只是这段行情的运气。
- 把 IC 序列按日期切 train/test，看 test 段是否保持同号且仍显著，
  才是「因子稳定」的直接证据。
- 合成因子需要先给每个分量定方向（IC 为负则反向），这个方向**只能用
  train 段的 IC 决定**，否则就是前视（用未来信息选方向）。

防泄漏：IC 在 t 日用的是 t→t+horizon 的未来收益。train 段最后 horizon 天
的 IC 会「看到」test 段的价格，故切分时裁掉这段重叠。
"""
from __future__ import annotations

import pandas as pd

from quant.backtest.ic_analysis import daily_ic, ic_summary
from quant.factor.factors import combine_factors


def ic_train_test(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    train_ratio: float = 0.7,
    horizon: int = 20,
    method: str = "spearman",
    min_count: int = 5,
) -> dict:
    """把因子的 daily IC 序列按日期切 train/test，分别汇总。

    返回:
        {
          "full":  ic_summary(全段),
          "train": ic_summary(train 段),
          "test":  ic_summary(test 段),
          "sign_consistent": bool,  # train 与 test 的 mean_ic 是否同号
        }

    切分逻辑：按 IC 日期序列的位置切。train 末尾裁掉 horizon 天，
    因为那些日子的 IC 用到了 test 段的未来收益，留着会泄漏。
    """
    ic = daily_ic(factor, fwd_ret, method=method, min_count=min_count)
    n = len(ic)
    summaries = {"full": ic_summary(ic)}

    if n == 0:
        summaries["train"] = ic_summary(ic)
        summaries["test"] = ic_summary(ic)
        summaries["sign_consistent"] = False
        return summaries

    cut = int(n * train_ratio)
    # train 裁掉末尾 horizon 天，避免其未来收益窗口探入 test。
    train_ic = ic.iloc[: max(cut - horizon, 0)]
    test_ic = ic.iloc[cut:]

    train_s = ic_summary(train_ic)
    test_s = ic_summary(test_ic)
    summaries["train"] = train_s
    summaries["test"] = test_s

    tr, te = train_s["mean_ic"], test_s["mean_ic"]
    summaries["sign_consistent"] = bool(
        pd.notna(tr) and pd.notna(te) and (tr * te > 0)
    )
    return summaries


def orient_by_ic(factor: pd.DataFrame, ic_mean: float) -> pd.DataFrame:
    """按 IC 符号给因子定向：IC<0 则反向（取负），使其「越高越好」成立。

    纯函数，不看未来；调用方负责传入「只用 train 段算出的」ic_mean。
    """
    if pd.notna(ic_mean) and ic_mean < 0:
        return -factor
    return factor


def build_oriented_composite(
    factors: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    upto_date,
    horizon: int = 20,
    method: str = "spearman",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """用 upto_date 之前（train 段）的 IC 给每个分量定向，再 rank 合成。

    参数:
        factors:   {名称: 因子面板}，多个待合成的分量。
        fwd_ret:   未来收益面板，用于算各分量的 train 段 IC。
        upto_date: train/test 的切分日期。**只用此日期之前的 IC 定向**，
                   避免用未来信息选方向（前视）。
        horizon:   未来收益跨度，用于裁掉 train 末尾的重叠天。

    返回:
        (合成因子面板, {分量名: 该分量 train 段 mean_ic})
    其中合成 = 各分量按 IC 定向后做横截面 rank 再平均（combine_factors）。
    """
    # 只取切分日期之前的未来收益来算 train IC（其未来窗口可能仍探入，
    # 故再按位置裁掉末尾 horizon 天，与 ic_train_test 口径一致）。
    fwd_train = fwd_ret.loc[fwd_ret.index < upto_date]
    if len(fwd_train) > horizon:
        fwd_train = fwd_train.iloc[:-horizon]

    oriented = []
    signs: dict[str, float] = {}
    for name, fac in factors.items():
        ic = daily_ic(fac, fwd_train, method=method, min_count=5)
        mean_ic = ic_summary(ic)["mean_ic"]
        signs[name] = float(mean_ic) if pd.notna(mean_ic) else 0.0
        oriented.append(orient_by_ic(fac, mean_ic))

    composite = combine_factors(*oriented)
    return composite, signs
