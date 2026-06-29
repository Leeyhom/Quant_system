# JoinQuant v8 Validation Note

v8 是 v7 失败后的稳健恢复版，不是新的 in-sample 收益搜索冠军。

它对应 `jointquant/v7/v7_validation.md` 里的 `baseline_like_v6_jq`：

| factor_name | top_n | rebalance_days | industry_cap | filter_mode | exposure_mode | weight_mode | total_return | annualized_return | sharpe | max_drawdown |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| q05_lv05 | 10 | 60 | 2 | mom120_gt_neg10 | fixed95 | equal | +257.35% | 20.79% | 0.99 | 18.06% |

为什么不用 v7 默认：

- v7 聚宽真实总收益只有 +111.13%，低于 v6 的 +188.42%。
- v7 最大回撤 34.01%，远高于 v6 的 22.28%。
- v7 的分数倾斜在小资金整手约束下把平均最大单票权重从 10.58% 抬到 12.16%。
- v7 取消动量过滤后，2021-2022 弱势价值陷阱暴露明显增加。

v8 参数：

```text
TOP_N = 10
INDUSTRY_CAP = 2
REBALANCE_DAYS = 60
MAX_EXPOSURE = 95%
MOMENTUM_120_MIN = -10%
QUALITY_WEIGHT = 0.5
LOWVOL_WEIGHT = 0.5
INCLUDE_HOLDER = False
WEIGHT_MODE = equal
```

下一步必须先补齐聚宽 152 只股票池的本地缓存，再谈 v9 收益增强。
