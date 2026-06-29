"""engine —— 最小回测引擎：信号 + 行情 -> 每日净值序列。

模型（呼应 docs/02 第三节）：
- 每日策略收益 = 当日持仓(0/1) × 当日价格收益率。
- 每当仓位变化（买入或卖出），按成交比例扣「手续费 + 滑点」成本。
- 净值从 1 起步累乘，得到 1 元本金的增长轨迹。

简化与边界（M2 故意保持最小，M3 再补）：
- 全仓进出（仓位只有 0 或 1），不做分批/加减仓。
- 暂不模拟涨跌停导致的「无法成交」。
- T+1 由策略层的 shift(1) 保证（今买明卖），引擎不再重复处理。
"""
from __future__ import annotations

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    price_col: str = "close",
    fee_rate: float = 0.00025,   # 单边手续费率，万2.5（券商佣金量级）
    slippage: float = 0.0005,    # 单边滑点，0.05%（真实成交价的偏差）
) -> pd.DataFrame:
    """运行回测，返回逐日明细（含策略净值与基准净值）。

    参数:
        df: 行情，含 price_col 列，按日期升序。
        signal: 每日目标仓位(0/1)，与 df 等长（应已 shift 防未来函数）。
        fee_rate/slippage: 单边手续费率与滑点率，每次买或卖各扣一次。

    返回:
        DataFrame，逐日含列：
          ret           当日价格收益率
          position      当日持仓(0/1)
          cost          当日因交易产生的成本（占净值比例）
          strat_ret     扣成本后的策略当日收益率
          equity        策略净值（从 1 起）
          benchmark     基准净值：买入持有(Buy&Hold)
    """
    out = pd.DataFrame(index=df.index)
    if "date" in df.columns:
        out["date"] = df["date"].values

    # 价格日收益率
    out["ret"] = df[price_col].pct_change().fillna(0.0)
    out["position"] = signal.values

    # 仓位变化量 -> 交易额比例 -> 成本。买入(+1)和卖出(-1)各扣一次单边成本。
    turnover = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = turnover * (fee_rate + slippage)

    # 策略当日收益 = 持仓 × 价格收益 − 成本
    out["strat_ret"] = out["position"] * out["ret"] - out["cost"]

    # 累乘成净值
    out["equity"] = (1.0 + out["strat_ret"]).cumprod()
    out["benchmark"] = (1.0 + out["ret"]).cumprod()
    return out
