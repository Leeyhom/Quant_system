"""walkforward_demo —— B：样本内/样本外检验双均线是否过拟合。

运行方式：
    conda activate quant
    python scripts/walkforward_demo.py            # 默认 600519
    python scripts/walkforward_demo.py 000001

逻辑：前 70% 数据选最优参数，后 30% 用该参数验证，对比两段表现。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.config import RAW_DATA_DIR
from quant.data.akshare_loader import fetch_daily, save_parquet, load_parquet
from quant.backtest.split import walk_forward_dual_ma

SHORT_WINDOWS = [3, 5, 8, 10, 15, 20]
LONG_WINDOWS = [20, 30, 40, 50, 60]


def _load_or_fetch(symbol: str):
    path = RAW_DATA_DIR / f"{symbol}.parquet"
    if path.exists():
        return load_parquet(symbol)
    df = fetch_daily(symbol, start="20240101", end="20251231")
    save_parquet(df, symbol)
    return df


def _fmt(m: dict) -> str:
    return (
        f"总收益 {m['total_return']:+.2%} | 夏普 {m['sharpe']:.2f} | "
        f"最大回撤 {m['max_drawdown']:.2%}"
    )


def main(symbol: str = "600519") -> None:
    print(f"读取/拉取 {symbol} 行情 ...")
    df = _load_or_fetch(symbol)

    res = walk_forward_dual_ma(df, SHORT_WINDOWS, LONG_WINDOWS, train_ratio=0.7)
    print(f"\n样本内({res['n_train']}天) 选出最优参数: MA{res['best_short']}/{res['best_long']}")
    print(f"  样本内 train : {_fmt(res['train'])}")
    print(f"  样本外 test  : {_fmt(res['test'])}   <- 用同参数、没见过的数据")

    # 简单判读
    train_s, test_s = res["train"]["sharpe"], res["test"]["sharpe"]
    if test_s >= train_s * 0.5 and test_s > 0:
        verdict = "样本外表现尚可，过拟合迹象较弱"
    elif test_s <= 0 < train_s:
        verdict = "样本内赚、样本外亏 —— 典型过拟合信号 ⚠️"
    else:
        verdict = "样本外明显弱于样本内，需警惕过拟合"
    print(f"\n判读：{verdict}")
    print("完成 ✅")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "600519")
