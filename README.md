# Quant System

面向学习、研究和模拟盘迭代的多市场量化交易系统。项目从 A股 起步，逐步扩展到港股、美股；核心目标不是追一个漂亮回测曲线，而是建立一套可复现、可审计、可被人和 LLM 共同改进的研究闭环。

> 重要提醒：本项目不是投资建议，所有收益数字都来自历史回测或模拟盘导出。历史表现不代表未来收益。当前最重要的工程任务是继续消除过拟合、数据不一致和执行差异。

## 当前状态

项目已完成从数据管道到跨市场验证的完整迭代。当前主攻方向：**CN800扩展池（800只）+ 多因子截面选股**。

### 当前最优基线：CN800 v5

聚宽验证（2019-2025, 6万本金）：
- v5 在 v2~v8 七个迭代版本中回测表现最好
- 800只候选池（沪深300+中证500），申万31行业分类，日均持仓~10只

v5 因子体系（7因子）：
- value_blend(1.0) + growth_peg(1.0) + amihud(1.0) + quality_roe(0.5) + low_vol_60(0.5) + residual_momentum(0.5) + roe_stability(0.3)
- 自适应动量: 市场regime检测 + 牛市raw_momentum_120加分
- 波动率目标降仓: 25%目标, 50%底仓
- 动态行业上限: max(2, 池内行业股票数/10)
- 资金效率: 放宽slot约束(1.50x)

### 本地 Walk-Forward 验证

本地 honest walk-forward（train=480/test=120/step=120, 11窗口, v3因子集）验证了池子扩展的收益：
- CN800池(702只有效): +182.8% 收益, 年化 +22.0%, **Sharpe 0.98**, 回撤 34.8%
- 旧池(91只有效): +60.3% 收益, 年化 +9.4%, Sharpe 0.48, 回撤 46.2%
- **不改因子，仅将股票池从152→800只，收益+122pct，Sharpe翻倍，回撤-11pct**

> 注意：本地WF用的是v3时代的5因子（无residual_momentum/roe_stability），用于验证池子扩大的效果。聚宽v2~v8迭代在此基础上逐步叠加因子和组合优化。

### 策略文件

```text
scripts/joinquant_cn800_strategy_v5.py    ← 当前生产基线（聚宽回测最优）
scripts/joinquant_cn800_strategy_v4.py    ← v4（自适应动量+资金效率，轻量备选）
scripts/joinquant_cn800_strategy.py       ← v3（残差动量+动态cap）
scripts/joinquant_cn800_strategy_v6~v8.py ← 实验版本（完整迭代记录）
scripts/cn800_walkforward.py              ← 本地 walk-forward 验证
scripts/cn800_v4_engine.py                ← 本地执行引擎（实盘对接用）
```

### 版本演进简史

| 版本 | 核心改动 | 聚宽相对表现 | 结论 |
|---|---|---|---|
| v2 | 800池+SW31行业 | 基线 | 池子扩大奠定基础 |
| v3 | +残差动量+动态cap | 改善 | 动量信号有效 |
| v4 | +自适应动量+资金效率 | 改善 | 市场状态感知有价值 |
| **v5** | **+vol目标降仓+ROE稳定性** | **最优** | **✅ 当前生产基线** |
| v6 | 回撤刹车 | 变差 | 择时降仓不适配集中组合 |
| v7 | 价值×动量交互 | 持平 | 交互项与父因子共线 |
| v8 | 短期反转因子 | 变差 | 因子叠加边际衰减 |

### 关键经验

经过v2→v8七轮迭代的核心教训：
1. **池子扩大是最强杠杆**：152→800只带来的改善超过所有后续因子叠加之和
2. **结构性修复 > 因子叠加**：v3(残差动量)、v4(自适应动量)改变了信号利用方式，比v7/v8的纯因子叠加有效
3. **择时降仓极度危险**：v6回撤刹车锁死仓位2年+，收益腰斩。10只集中组合的正常波动就是15-25%，降仓阈值无法避开
4. **因子边际效用递减**：6→7因子的增量远小于2→3或3→4
5. **不是每个版本都需要超越前一个**：v5→v8的"持平或更差"同样是珍贵的负样本

## 核心结论

过去几个阶段反复验证出的经验：

- 信息量优先于模型复杂度。加权方法不能凭空创造信号。
- A股量价因子在样本外容易变号，基本面/价值/质量/现金流更稳。
- 行业/市值中性化能提高因子统计稳定性，但不保证每段样本都提高组合夏普。
- 等权合成在方向先验稳定时，常常比 IC 加权更稳，因为 A股因子方向会随 regime 漂移。
- 本地验证池必须和实盘/聚宽池一致，否则横截面排名会变成另一个问题。
- 任何“全样本最优参数”都应默认视为过拟合嫌疑，必须用 walk-forward 和真实导出复盘二次确认。

## 项目结构

```text
quant/
  config.py                    # 路径、历史窗口等全局配置
  data/                        # 数据层：A股/港股/美股 loader、估值、行业、股票池
  factor/                      # 因子层：价格量价、基本面、中性化、多因子合成
  strategy/                    # 策略层：单票策略、A股因子规格
  backtest/                    # 回测层：组合、分层、walk-forward、费用、指标

scripts/
  joinquant_cn_sim_strategy_v8.py      # v7 失败后的稳健恢复基线
  joinquant_cn_sim_strategy_v9.py      # 当前最强聚宽历史回测基线
  joinquant_cn_sim_strategy_v10.py     # 冷启动/模拟盘启动保护候选
  analyze_joinquant_exports.py         # 聚宽交易/持仓/日志导出复盘
  joinquant_v9_2025_attribution.py     # v9 真实导出的 2025 年归因
  joinquant_v9_path_sensitivity.py     # v9 热路径/冷启动/仓位 ramp 对照
  refetch_joinquant_pool.py            # 拉取聚宽策略池到本地缓存
  joinquant_v6_validation.py           # v6 本地验证
  joinquant_v7_validation.py           # v7 失败前本地验证，用于反思过拟合
  cn_walkforward_honest.py             # A股 walk-forward 验证
  *_demo.py                            # 各阶段研究/演示脚本

docs/
  01~17                                # 从基础概念到跨市场研究的阶段文档
  18_joinquant_a_share_sim.md          # 聚宽模拟盘起点
  19_joinquant_v6_alpha.md             # v6 收益改进
  20_joinquant_v7_score_tilt.md        # v7 设计与本地验证
  21_joinquant_v7_failure_v8_recovery.md # v7 失败复盘与 v8 恢复
  22_platform_backtest_and_v9.md        # 平台化回测与 v9 候选策略
  23_v9_2025_cold_start_v10.md          # v9 2025 冷启动复盘与 v10
  AUDIT_专业量化审计报告.md             # 审计视角的风险提示

jointquant/
  v6/, v7/, v8/, v9/                   # 聚宽版本复盘摘要和验证文件
  version_metrics/                     # 各版本指标对比

data/
  raw/                                 # 本地行情/估值缓存，体积大，默认不入库
```

## 快速开始

推荐环境：

```bash
conda create -n quant python=3.11
conda activate quant
pip install -r requirements.txt
```

联网拉数据时，本地代理可能会拦截，需要绕过：

```bash
NO_PROXY='*' PYTHONPATH=. python scripts/refetch_history.py --limit 10
```

运行基础演示：

```bash
PYTHONPATH=. python scripts/fetch_demo.py
PYTHONPATH=. python scripts/backtest_demo.py
PYTHONPATH=. python scripts/multifactor_demo.py
```

运行 A股 walk-forward/验证脚本：

```bash
PYTHONPATH=. python scripts/cn_walkforward_honest.py
PYTHONPATH=. python scripts/joinquant_v6_validation.py
PYTHONPATH=. python scripts/joinquant_v7_validation.py
```

复盘聚宽导出：

```bash
PYTHONPATH=. python scripts/analyze_joinquant_exports.py jointquant/v7 v7
```

补齐聚宽策略池本地缓存：

```bash
NO_PROXY='*' PYTHONPATH=. python scripts/refetch_joinquant_pool.py
```

## 数据源

A股：

- 行情：AkShare 新浪接口 `stock_zh_a_daily`
- 估值：`stock_value_em`
- 默认历史窗口：`2018-01-01 ~ 2025-12-31`

注意：

- 不使用默认的东方财富 `stock_zh_a_hist` 作为 A股行情主接口，因为它容易拒绝 Python 客户端。
- `data/raw/` 是本地缓存，不提交到 GitHub。
- 若扩展历史窗口，需要强制重拉缓存，不能只改配置。

美股/港股：

- 已有 loader 和基本面尝试，但当前主线仍是 A股聚宽模拟盘策略。
- 跨市场结果更多用于方法论验证，不应直接混同为同一实盘策略。

## 防未来函数约定

项目的基本规则：

- 今日计算的信号必须明日或之后使用。
- 聚宽策略使用 `context.previous_date` 取基本面和历史价格。
- 中性化只在单日截面内完成。
- 季报基本面必须按公告日或保守披露日对齐。
- 参数选择必须区分训练期和测试期，不能只看全样本最优。

如果新增策略，请在 PR 或文档中说明：

- 信号在哪一天可见；
- 调仓在哪一天执行；
- 是否使用了未来价格、未来财报、未来成分股；
- 费用、滑点、整手约束是否被纳入。

## 当前研究路线

短期优先级：

1. 在聚宽跑 v10 的全周期和 2025 单年冷启动回测，确认启动 ramp 是否修复 v9 的 2025 冷启动问题。
2. 修复 AkShare/py_mini_racer 行情拉取问题，补齐聚宽 152 只策略池的本地数据。
3. 用一致股票池重做 walk-forward，而不是继续在 80~89 只缓存池上调参。
4. 用 `scripts/export_joinquant_v9_targets.py` 导出 target book，在 RQAlpha 等平台重放执行。

中期方向：

- 扩大 A股可交易池并处理动态成分/幸存者偏差。
- 增加更正交的基本面信号，例如质量、成长、现金流、盈利修正。
- 构建严格的 walk-forward scoreboard：收益、Sharpe、最大回撤、信息比率、换手、费用、年度稳定性同时排名。
- 将 LLM/agent 用于复盘、异常解释和研究辅助，而不是直接替代信号生成。

## 给贡献者

请先阅读：

- [AGENTS.md](AGENTS.md)：项目工作约定，适合 LLM/代码代理读取。
- [docs/21_joinquant_v7_failure_v8_recovery.md](docs/21_joinquant_v7_failure_v8_recovery.md)：最新策略失败复盘。
- [docs/22_platform_backtest_and_v9.md](docs/22_platform_backtest_and_v9.md)：v9 候选与跨平台回测路径。
- [docs/23_v9_2025_cold_start_v10.md](docs/23_v9_2025_cold_start_v10.md)：v9 2025 冷启动复盘与 v10。
- [docs/AUDIT_专业量化审计报告.md](docs/AUDIT_专业量化审计报告.md)：审计视角风险。
- [CONTRIBUTING.md](CONTRIBUTING.md)：贡献流程和 PR 要求。

提交策略改动时，请同时提交：

- 设计解释；
- 本地验证结果；
- 是否存在未来函数风险的说明；
- 与基准版本的对比；
- 如果是聚宽策略，还需要导出交易、持仓、日志并运行 `analyze_joinquant_exports.py`。

## 给 LLM / Agent

如果你是 LLM 或代码代理，请按这个顺序理解项目：

1. 读 `README.md` 获取当前真实状态。
2. 读 `AGENTS.md` 获取工作约定。
3. 读 `docs/21_joinquant_v7_failure_v8_recovery.md` 理解 v7 失败教训。
4. 读 `docs/22_platform_backtest_and_v9.md` 理解 v9 候选和平台化回测路径。
5. 读 `docs/23_v9_2025_cold_start_v10.md` 理解 v9 2025 冷启动问题和 v10。
6. 读 `scripts/joinquant_cn_sim_strategy_v9.py` 和 `scripts/joinquant_cn_sim_strategy_v10.py` 获取当前聚宽策略。
7. 读 `scripts/analyze_joinquant_exports.py` 理解如何复盘真实导出。

不要默认相信全样本最优结果。任何新策略都要问：数据池是否一致？是否样本外？是否有未来函数？是否扣除了费用和整手约束？是否只是某一年贡献了大部分收益？

## 安全与隐私

- 不提交 `.env`、API key、飞书 webhook、券商密钥。
- JoinQuant 策略文件里的 `FEISHU_WEBHOOK` 默认留空，使用者自行在本地填写。
- `data/`、`results/`、压缩包、原始交易导出默认不入库。

## License

暂未指定开源许可证。协作者在复用、分发或商用前请先和维护者确认。
