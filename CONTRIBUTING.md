# Contributing

欢迎一起改进这个量化研究项目。这个仓库的优先级是：可复现、可解释、可审计，然后才是收益。

## 开发环境

```bash
conda create -n quant python=3.11
conda activate quant
pip install -r requirements.txt
```

联网拉取数据时：

```bash
NO_PROXY='*' PYTHONPATH=. python scripts/refetch_history.py --limit 10
```

## 分支和提交

建议分支命名：

```text
research/<topic>
fix/<topic>
docs/<topic>
joinquant/<version>
```

提交前至少运行：

```bash
python -m py_compile quant/**/*.py scripts/*.py
```

如果 shell 不支持 `**`，可以用：

```bash
python -m compileall quant scripts
```

## 新策略 PR 必须说明

请在 PR 描述或对应 `docs/` 文档中说明：

- 策略目标：收益增强、降回撤、降换手、扩数据、修执行差异等。
- 数据范围：市场、股票池、起止日期、数据源。
- 信号可见时间：是否使用公告日/上一交易日，是否存在未来函数风险。
- 组合规则：选股数量、行业上限、仓位、再平衡频率、费用/滑点/整手约束。
- 验证方式：全样本、样本外、walk-forward、聚宽导出复盘。
- 对比基准：至少和当前主线 `joinquant_cn800_strategy_v5.py` 或明确的历史 baseline 比较。

## 不接受的改动

- 只展示收益提升、不解释风险和数据口径的策略。
- 使用未来价格、未来财报、未来成分股但未说明。
- 把 `.env`、API key、飞书 webhook、券商密钥提交到仓库。
- 把 `data/raw/` 大体积缓存提交到仓库。
- 在未对齐股票池的情况下，用本地最优参数直接覆盖聚宽策略。

## 文档约定

重要研究结论要写入 `docs/`：

- 原理先行：先解释为什么，再写代码。
- 诚实记录负结果：失败版本同样有价值。
- 指标不要只写收益，至少包含 Sharpe、最大回撤、胜率、换手或费用。

## 当前主线

截至当前版本，A股聚宽/模拟盘主线是：

```text
scripts/joinquant_cn800_strategy_v5.py
```

观察计划：

```text
docs/24_cn800_v5_paper_trading_plan.md
```

CN800 v5 已接入聚宽模拟盘。下一阶段先观察至少一个自然月，回收交易、持仓、日志和净值，再做归因与下一轮迭代。观察期内不因为短期涨跌直接修改 alpha。

历史策略教训：

```text
docs/21_joinquant_v7_failure_v8_recovery.md
docs/22_platform_backtest_and_v9.md
docs/23_v9_2025_cold_start_v10.md
```

下一步高优先级工作：

1. 保持 CN800 v5 聚宽模拟盘运行并记录开始日期、初始资金、基准和参数。
2. 一个月后将模拟盘导出放入 `jointquant/cn800_v5_paper/`，生成可追踪复盘摘要。
3. 修复 CN800 本地验证链条中的动态成分、幸存者偏差、日志解析和消融表问题。
4. 用 RQAlpha/Backtrader/LEAN 等第三方平台重放 target book，审计执行差异。
