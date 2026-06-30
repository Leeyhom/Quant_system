# 22 · 平台化回测与 v9 候选策略

> 日期：2026-06-30  
> 目标：把策略从“单一聚宽脚本”推进到“本地可验证、聚宽可执行、第三方平台可复盘”的研究闭环。

> 更新：v9 聚宽真实导出已完成，2019-2025 全周期表现确认为当前最好；但 2025 单年冷启动暴露路径依赖问题。后续结论见 [23 · v9 2025 冷启动复盘与 v10](23_v9_2025_cold_start_v10.md)。

## 一、本轮外部项目观察

这次重点看的是 GitHub 上高星、工程化成熟的量化项目。它们给我们的启发不是“复制某个因子”，而是“把研究、执行、风控、复盘拆开”。

| 项目 | GitHub 页面观察 | 可借鉴思想 | 本项目落地 |
|---|---:|---|---|
| [microsoft/qlib](https://github.com/microsoft/qlib) | 45k+ stars；定位为 AI-oriented quant platform，覆盖 data/model/backtest/order execution 全链路 | 研究工作流要模块化，数据、模型、组合、执行、分析可拆可换 | 保持 `quant/data`、`factor`、`backtest`、`strategy` 分层；新增平台回测文档 |
| [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | 50k+ stars；强调 dry-run、backtesting、hyperopt、lookahead-analysis | 任何策略上线前都要 dry-run 和反前视检查；工具链本身要暴露偏差检测 | v9 不直接上实盘，先本地/聚宽/RQAlpha 三方复核 |
| [mementum/backtrader](https://github.com/mementum/backtrader) | 22k+ stars；完整 broker simulation、commission、sizers、analyzers | 回测引擎应把策略信号、仓位 sizing、费用、撮合拆开 | 新增 `export_joinquant_v9_targets.py`，先导出 target book，再由外部引擎审计执行 |
| [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | 20k+ stars；event-driven、modular、CLI 支持 backtest/optimize/live | 事件驱动平台适合做最终执行一致性压力测试 | 短期用 CSV target replay；长期再写 Lean 原生策略 |
| [ricequant/rqalpha](https://github.com/ricequant/rqalpha) | 6k+ stars；A股友好，可扩展/可替换，支持回测、模拟、实盘和 Mod | RQAlpha 很适合做 A股执行审计，尤其是费用/整手/撮合差异 | 复用已有 `rqalpha_cn_replay_strategy.py`，用 v9 target CSV 重放 |

核心结论：外部成熟项目都不是把“收益曲线”作为唯一中心，而是把策略拆成可审计组件。我们的下一步也应该是 target book 驱动的跨平台复盘，而不是继续堆参数。

## 二、当前 v9 决策

v9 不新增新因子，不引入 LLM 选股，不启用分数倾斜。v7 已经证明：在池子未对齐时，用本地全样本调出来的进攻参数很容易在聚宽真实环境里失效。

本轮本地验证只保留一个最小改动：

```text
MAX_EXPOSURE: 95% -> 98%
```

含义：v8 在 6 万本金、100 股整手、最低佣金约束下，实际平均股票仓位只有约 87%。提高目标仓位到 98% 并不是加杠杆，而是减少整手约束导致的现金拖累。

保留但默认关闭的研究开关：

```text
HOLD_MULTIPLIER = 1.5      # 老持仓仍在 TOP_N*1.5 高分带内则优先保留
USE_VOL_BUDGET = False     # 不启用低波风险预算
```

原因：当前 80/152 缓存池验证显示，`98%仓位 + 持仓缓冲` 同时提升收益、Sharpe、回撤和换手；低波预算仍然降低收益与 Sharpe。

## 三、本地验证结果

脚本：

```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python scripts/joinquant_v9_validation.py
```

输出：

```text
jointquant/v9/v9_validation.csv
jointquant/v9/v9_validation.md
```

当前缓存覆盖：

```text
JoinQuant pool cache coverage: 80/152
```

结果摘要：

| config | total_return | annualized | Sharpe | max_drawdown | avg_cash | avg_exposure |
|---|---:|---:|---:|---:|---:|---:|
| v9_buffer_less_cash_98 | 334.95% | 24.36% | 1.097 | 19.02% | 8.81% | 91.19% |
| v8_less_cash_98 | 311.45% | 23.34% | 1.071 | 18.87% | 9.46% | 90.54% |
| v8_baseline_equal | 279.05% | 21.85% | 1.056 | 19.92% | 12.52% | 87.48% |
| v9_buffer_equal | 301.84% | 22.91% | 1.099 | 20.18% | 12.69% | 87.31% |
| v9_buffer_vol_budget | 260.03% | 20.93% | 1.042 | 20.29% | 12.55% | 87.45% |

结论：

- `v9_buffer_less_cash_98` 是当前候选：收益相对 v8 baseline 提升约 +55.90pct，Sharpe 提升约 +0.041，最大回撤下降约 -0.91pct，平均换手也降低。
- 单独 `98%` 仓位也有效，但换手更高；加入持仓缓冲后收益/换手结构更均衡。
- 有界低波预算没有改善回撤，且明显降低收益/Sharpe，当前不启用。
- 由于本地只有 80/152 缓存，这还不是最终结论。v9 必须跑聚宽真实导出后才能替代 v8。

## 四、平台回测路径

### 1. 本地框架

本地用于策略研究和快速归因：

```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python scripts/joinquant_v9_validation.py
```

本地只回答：

```text
同一缓存池、同一费用/整手模型下，候选参数是否明显优于 baseline？
```

它不能替代聚宽真实回测，因为当前 AkShare 环境未补齐 152 池。

### 2. 聚宽 JoinQuant

策略文件：

```text
scripts/joinquant_cn_sim_strategy_v9.py
```

聚宽设置：

```text
本金：60000
频率：每天
区间：2019-01-01 ~ 2025-12-31
基准：沪深300
```

跑完后导出：

```text
transaction.csv
position.csv
log.txt
```

复盘：

```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python scripts/analyze_joinquant_exports.py jointquant/v9 v9
```

v9 替代 v8 的条件：

```text
总收益、Alpha、Sharpe/信息比率至少不劣于 v8；
最大回撤不能显著放大；
年度路径不能只靠 2025 单年贡献；
单票贡献不能出现 v7 那种集中伤害。
```

### 3. RQAlpha

先导出 v9 target book：

```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python scripts/export_joinquant_v9_targets.py --out data/rqalpha_bridge/cn_target_weights.csv
```

再用现有 RQAlpha replay 策略：

```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python scripts/rqalpha_run_cn_bridge.py
```

定位：RQAlpha 不负责重新算因子，只负责重放目标权重，审计撮合、费用、整手和现金拖累。

当前本机状态：RQAlpha 已安装，但缺少 `~/.rqalpha/bundle` 数据包。需要先运行 `rqalpha download-bundle`，或用 `--bundle /path/to/bundle` 指向已有数据包。

### 4. Backtrader

Backtrader 适合做 pandas/CSV target replay。当前项目还没有完整 Backtrader strategy 文件，建议下一步用 `export_joinquant_v9_targets.py` 的 CSV 作为输入，实现一个最小 replay broker：

```text
每日读取目标权重 -> 按 close/open 成交假设调仓 -> 使用 Backtrader commission/sizer/analyzer 输出指标
```

这比直接重写因子更稳，因为先审计执行差异。

### 5. QuantConnect LEAN

LEAN 更适合做中长期跨市场执行压力测试。短期不建议直接迁移因子，因为 A股数据、涨跌停、T+1、整手、手续费都需要单独适配。

推荐路径：

```text
CSV target book -> LEAN custom data -> OnData 按目标权重调仓 -> 对比本地/RQAlpha/聚宽
```

### 6. Qlib

Qlib 对机器学习因子、数据工作流、模型滚动训练很强，但当前项目还没准备好直接接入 Qlib：

```text
需要先把 A股日频行情/估值/行业映射转换成 Qlib provider 格式；
再把当前 v8/v9 规则作为 baseline；
最后才适合尝试 ML 模型。
```

Qlib 当前对我们的最大启发是工程结构：完整 pipeline，而不是立刻上机器学习。

## 五、数据阻塞

本轮尝试补齐聚宽 152 池时，行情接口在当前环境报错：

```text
AttributeError: dlsym(..., mr_eval_context): symbol not found
```

这是 AkShare/py_mini_racer 动态库兼容问题，不是策略逻辑问题。估值接口可以部分继续，但行情无法补齐。

P0 修复方向：

```text
1. 修复 py_mini_racer / akshare 环境，或换一个不依赖该 JS runtime 的行情接口；
2. 补齐 v9 的 152 只聚宽池本地缓存；
3. 重跑 joinquant_v9_validation.py；
4. 再跑聚宽真实导出复盘。
```

## 六、下一步

短期：

1. 在聚宽跑 `joinquant_cn_sim_strategy_v9.py`。
2. 把 v9 的 transaction/position/log 放入 `jointquant/v9/`。
3. 用 `analyze_joinquant_exports.py` 生成真实导出归因。
4. 用 `export_joinquant_v9_targets.py` 生成 target book，并在 RQAlpha 重放。

中期：

1. 修复 152 池本地数据拉取。
2. 给 Backtrader 写 target replay adapter。
3. 给 LEAN 写 custom target book adapter。
4. 建立统一 scoreboard：local / JoinQuant / RQAlpha / Backtrader / LEAN 同表比较。

本轮最重要的克制：v9 只做减少现金拖累和降低无意义换手，不碰因子、不做分数倾斜、不引入模型。先把执行闭环打稳，再谈更高 alpha。
