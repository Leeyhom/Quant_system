# JoinQuant v5 升级说明

## v4 失败原因

v4 在聚宽里不是因为因子缺失或交易失败而变差：

- 日志显示因子字段完整：`earnings_yield,cashflow_yield,sales_yield,growth_peg_proxy,amihud`
- 每次调仓候选/打分数量基本完整：84~92 只
- 平均持仓 6.02 只，平均仓位 90.75%，执行层正常

真正问题是 v4 把组合从 v3 的 10 只压缩到 6 只，并把行业上限从 1 放到 2，导致价值因子的“便宜股”风险被放大。聚宽实盘口径里，2022-2024 的回撤主要来自新能源、面板、周期价值陷阱。

最差已实现盈亏集中在：

| code | name | realized_pnl |
| --- | --- | ---: |
| 600438.XSHG | 通威股份 | -6869.40 |
| 000725.XSHE | 京东方A | -3993.48 |
| 002129.XSHE | TCL中环 | -3824.00 |
| 002460.XSHE | 赣锋锂业 | -2998.00 |
| 002821.XSHE | 凯莱英 | -2596.00 |

结论：v4 的方向不是继续集中化，而是回到 v3 的分散结构，再加一个防价值陷阱机制。

## v5 设计

v5 采用：

- `TOP_N = 10`
- `REBALANCE_DAYS = 60`
- `INDUSTRY_CAP = 1`
- 因子：v3 原五因子 + `quality_roe = PB / PE`
- `quality_roe` 只给半权重：`QUALITY_WEIGHT = 0.5`
- 120 日动量过滤：`MOMENTUM_120_MIN = -0.10`

它不是硬删除 v4 亏损股，而是用“120 日动量不能低于 -10%”过滤掉明显处于中期下跌趋势的候选，避免单纯价值因子买入持续走弱的价值陷阱。

## 本地验证口径

- 窗口：2019-01-01 ~ 2025-12-31
- 因子历史：2018-01-01 起
- 约束：6 万本金、95% 总仓位、100 股整数手、手续费+滑点、未成交留现金

| version | total_return | annualized_return | sharpe | max_drawdown | avg_cash |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 285.34% | 22.15% | 1.04 | 19.55% | 10.01% |
| v4 聚宽实测 | 138.09% | 13.61% | 0.516 | 26.11% | - |
| v5 本地验证 | 314.08% | 23.46% | 1.17 | 14.19% | 10.90% |

## 为什么没有选搜索榜首

搜索榜首是 `lowvol_half / top10 / rebalance50 / industry_cap2`，本地 Sharpe 约 1.20。但它仍然使用行业上限 2，和 v4 的失败机制有相同隐患。

v5 选择 `quality_half / top10 / rebalance60 / industry_cap1 / mom120_gt_neg10`，牺牲一点本地榜单排名，换取更好的逻辑一致性：

- 不再集中到 6 只
- 不放宽行业到 2 只
- 加质量因子降低单纯便宜股暴露
- 加趋势过滤降低价值陷阱暴露

## 文件

- 聚宽 v5 策略：`scripts/joinquant_cn_sim_strategy_v5.py`
- v4 深度审计：`jointquant/v4/v4_deep_analysis.md`
- v5 搜索结果：`jointquant/v4/joinquant_v5_risk_filter_search.csv`
- v5 搜索脚本：`scripts/joinquant_v5_risk_filter_search.py`
