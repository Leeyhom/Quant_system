"""全局配置：集中管理路径与常量，避免代码里散落硬编码字符串。

为什么这么做：
- 路径只在一处定义，将来改目录结构只改这里。
- 用 pathlib.Path 而非字符串拼接，跨平台更稳。
"""
from pathlib import Path

# 项目根目录 = 本文件(quant/config.py)的上上级目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"        # 原始行情（直接落地的数据）

# 确保目录存在（首次运行自动创建，省去手动 mkdir）
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────── 数据时间窗口（M14） ─────────────────────────
# 为什么从 2018 起：M13 用 2 年(2024~2025)只覆盖 1~2 个 regime，撞上 2025Q3
# 行业回调一次就被带偏，且滚动只有 4 个窗口、统计不可信。拉到 2018 能覆盖
# 2018 熊 / 2019~2021 结构牛 / 2022 熊 / 2023 震荡 / 2024~2025 等多个 regime，
# 让「哪些因子真稳」的判断不被单次行情主导，滚动窗口也增到 20+。
# 实测新浪行情与东财估值接口都能回溯到 2018-01。
HISTORY_START = "20180101"
HISTORY_END = "20251231"
