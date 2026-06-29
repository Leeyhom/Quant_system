# 20 · 聚宽 A股 v7：从因子改进到组合施工

> 接 `docs/19_joinquant_v6_alpha.md`。v6 已经显著跑赢 v3/v4，但聚宽真实日志显示：
> holder 因子没有覆盖，实际运行的是 5 因子版。因此 v7 不再继续假设 holder 可用，
> 而是在真实可执行的 5 因子地基上改组合施工。

## 一、v6 真实复盘

聚宽导出文件位于：

```text
jointquant/v6/transaction.csv
jointquant/v6/position.csv
jointquant/v6/log.txt
```

复盘脚本：

```bash
PYTHONPATH=. python scripts/analyze_joinquant_exports.py jointquant/v6 v6
```

核心结果：

| 指标 | v6 聚宽真实导出 |
|---|---:|
| 总收益 | +188.42% |
| 年化收益 | +17.01% |
| Raw daily Sharpe | 0.96 |
| 最大回撤 | 22.28% |
| 最大回撤区间 | 2021-09-13 ~ 2022-10-31 |
| 平均股票仓位 | 86.04% |
| 平均现金 | 13.96% |
| 平均持仓数 | 9.82 |
| 平均最大单票权重 | 10.58% |
| 卖出胜率 | 61.20% |

最重要的日志证据：

```text
Factor columns: value_blend,growth_peg,amihud,quality_roe,low_vol_60 holder_coverage=0
```

所以 v6 的聚宽实盘口径并不是“含筹码集中度”的 6 因子，而是：

```text
value_blend + growth_peg + amihud + quality_roe + low_vol_60
```

这意味着 v7 的第一条纪律是：明确关闭 holder，保证说明、验证、实盘三者一致。

## 二、v7 改什么

v7 没有继续堆新因子。原因很简单：v6 的问题已经不是“完全没有 alpha”，而是 alpha 利用效率还不够。

改动：

- `INCLUDE_HOLDER = False`：聚宽覆盖为 0，直接关闭。
- `REBALANCE_DAYS = 50`：比 v6 的 60 天更快更新，但不走高频。
- `USE_MOMENTUM_FILTER = False`：本地验证里个股 120 日动量过滤牺牲了收益和 Sharpe。
- `TOP_N = 10 / INDUSTRY_CAP = 2 / MAX_EXPOSURE = 95%`：保持 v6 已验证的分散度和行业约束。
- `SCORE_TILT = 0.65`：从等权改成轻度分数倾斜，高分股略多、低分股略少。
- 单票上限 `14.5%`：避免分数倾斜变成集中押注。

分数倾斜的直觉：

```text
等权：第1名和第10名都是 9.5%
轻倾斜：第1名约 12%~13%，第10名约 6%~7%，总仓位仍约 95%
```

这比直接改成 top8 更温和。top8 的本地收益更高，但回撤和集中度也更高，不作为默认上线版。

## 三、本地验证

脚本：

```bash
PYTHONPATH=. python scripts/joinquant_v7_validation.py
```

口径：

- 2019-01-01 ~ 2025-12-31
- 本地 cached `DEFAULT_POOL`，89 只有效票
- 6 万本金
- 100 股整数手
- 最低 5 元佣金、印花税、过户费、滑点
- holder 关闭，对齐 v6 聚宽真实日志

对照：

| 版本 | 参数 | 总收益 | 年化 | Sharpe | 最大回撤 |
|---|---|---:|---:|---:|---:|
| v6 聚宽真实近似基线 | q05_lv05 / top10 / 60d / cap2 / mom120过滤 / fixed95 / equal | +257.35% | +20.79% | 0.99 | 18.06% |
| v7 默认 | q05_lv05 / top10 / 50d / cap2 / no filter / fixed95 / tilt065 | +358.44% | +25.34% | 1.20 | 20.23% |
| 高收益实验 | q00_lv00 / top8 / 50d / cap2 / no filter / fixed98 / equal | +514.20% | +30.90% | 1.19 | 24.19% |

v7 默认相对基线：

- 总收益提高约 +101pct。
- 年化提高约 +4.5pct。
- Sharpe 从 0.99 提到 1.20。
- 最大回撤增加约 +2.2pct，但仍低于 v6 聚宽真实导出的 22.28%。

## 四、为什么不用最高收益实验

`q00_lv00 / top8 / fixed98` 的本地收益非常诱人，但它做了三件更激进的事：

- 去掉 quality/lowvol 风险缓冲。
- 持仓从 10 只压到 8 只。
- 仓位从 95% 提到 98%。

这类版本适合作为研究分支，不适合作为下一版默认实盘候选。它的收益更高，但更依赖样本期里进攻暴露的好运。v7 默认要追求的是“收益、Sharpe、alpha 指标同步抬升”，而不是只把收益曲线抬陡。

## 五、产物

聚宽自包含策略：

```text
scripts/joinquant_cn_sim_strategy_v7.py
```

验证与复盘输出：

```text
scripts/joinquant_v7_validation.py
scripts/analyze_joinquant_exports.py
jointquant/v6/v6_deep_analysis.md
jointquant/v7/v7_validation.csv
jointquant/v7/v7_validation.md
```

下一步：把 `scripts/joinquant_cn_sim_strategy_v7.py` 放进聚宽跑 2019-01-01 ~ 2025-12-31，导出交易、持仓、日志后，用同一个 `analyze_joinquant_exports.py` 再做真实聚宽归因。
