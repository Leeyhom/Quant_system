"""fetch_demo —— M1 端到端 demo：拉取 → 存盘 → 读回 → 画图。

运行方式（先激活环境）：
    conda activate quant
    python scripts/fetch_demo.py

预期结果：
    1) 终端打印贵州茅台(600519)最近若干天日线
    2) data/raw/600519.parquet 文件生成
    3) data/raw/600519_close.png 收盘价走势图保存
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能 import 到 quant 包（把项目根目录加入搜索路径）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # 无界面后端：直接存图，不依赖弹窗
import matplotlib.pyplot as plt

from quant.data.akshare_loader import fetch_daily, save_parquet, load_parquet
from quant.config import RAW_DATA_DIR

SYMBOL = "000001"  # 贵州茅台


def main() -> None:
    print(f"[1/4] 拉取 {SYMBOL} 日线行情 ...")
    df = fetch_daily(SYMBOL, start="20240101", end="20251231")
    print(f"      共 {len(df)} 个交易日，列：{list(df.columns)}")

    print(f"[2/4] 存为 Parquet ...")
    save_parquet(df, SYMBOL)

    print(f"[3/4] 从本地读回校验 ...")
    df2 = load_parquet(SYMBOL)
    assert len(df) == len(df2), "读回行数与写入不一致！"
    print(df2.tail())

    print(f"[4/4] 画收盘价走势图 ...")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df2["date"], df2["close"])
    ax.set_title(f"{SYMBOL} close price")
    ax.set_xlabel("date")
    ax.set_ylabel("close")
    fig.tight_layout()
    out_png = RAW_DATA_DIR / f"{SYMBOL}_close.png"
    fig.savefig(out_png, dpi=120)
    print(f"      图已保存：{out_png}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
