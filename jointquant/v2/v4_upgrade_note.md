# JoinQuant v4 升级说明

## 对齐口径

- 本地验证窗口：2019-01-01 ~ 2025-12-31
- 因子历史窗口：2018-01-01 起，用于 2019 初始买入时的滚动因子计算
- 约束：6 万本金、95% 总仓位上限、100 股整数手、A股手续费+滑点、未买成留现金
- 数据池：本地可用的 89 只大中盘股票

## 为什么要从 2019 重新验证

聚宽 v3 截图使用的是 2019-01-01 ~ 2025-12-31。此前本地验证表混有 2018 起跑结果，而 2018 是策略很敏感的下跌起点，会明显改变首批持仓、回撤和复利路径。现在本地验证已按 2019 起跑重新对齐。

## v3 本地对齐结果

| version | factors | top_n | rebalance | industry_cap | total_return | annualized_return | sharpe | max_drawdown | avg_cash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | earnings_yield,cashflow_yield,sales_yield,growth_peg,amihud | 10 | 40 | 1 | 285.34% | 22.15% | 1.04 | 19.55% | 10.01% |

## v4 选择

搜索结果显示，最高 Sharpe 的可复刻版本是：

| version | factors | top_n | rebalance | industry_cap | weight | total_return | annualized_return | sharpe | max_drawdown | avg_cash |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| v4 | earnings_yield,cashflow_yield,sales_yield,growth_peg,amihud | 6 | 30 | 2 | equal | 394.81% | 26.77% | 1.19 | 18.25% | 7.24% |

这不是换成更复杂的模型，而是把 v3 的信号利用得更集中、更频繁：

- `top_n` 从 10 降到 6：提高单票 alpha 暴露，减少中低分股票稀释。
- `rebalance` 从 40 交易日降到 30 交易日：更快更新基本面/量价排序。
- `industry_cap` 从 1 放宽到 2：允许强行业内拿第二只票，避免过度中性化牺牲收益。
- 因子仍使用 v3 的完整可复刻因子，避免为了本地收益引入聚宽不可稳定取到的数据。

## 没有选择的更激进候选

`quality_growth_momentum / top6 / rebalance30 / industry_cap2` 的本地总收益更高：

- 总收益 453.19%
- 年化 28.88%
- Sharpe 1.09
- 最大回撤 28.58%

我没有把它作为主推 v4，因为它牺牲了太多回撤，且更多依赖 2019~2021 的成长/动量环境。对模拟盘/实盘而言，Sharpe 和回撤稳定性比单纯总收益更重要。

## 文件

- 聚宽 v4 策略：`scripts/joinquant_cn_sim_strategy_v4.py`
- 本地搜索脚本：`scripts/joinquant_v4_aligned_search.py`
- 完整搜索结果：`jointquant/v2/joinquant_v4_aligned_search.csv`
