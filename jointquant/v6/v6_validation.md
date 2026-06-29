# JoinQuant v6 Validation

- Window: 20190101 ~ 20251231
- Pool: DEFAULT_POOL cached A-share large/mid caps (89 names)
- Cost: 60k CNY, 100-share lots, min commission, stamp duty, transfer fee, slippage

## Baselines

| kind     | factor_name     |   top_n |   rebalance_days |   industry_cap | filter_mode     |   hold_multiplier |   total_return |   annualized_return |   sharpe |   max_drawdown |   avg_cash |   avg_turnover |   avg_holdings |
|:---------|:----------------|--------:|-----------------:|---------------:|:----------------|------------------:|---------------:|--------------------:|---------:|---------------:|-----------:|---------------:|---------------:|
| baseline | v5_quality_half |      10 |               60 |              1 | mom120_gt_neg10 |                 1 |        3.1408  |            0.234603 |  1.16864 |       0.141908 |   0.10904  |       0.833844 |             10 |
| baseline | v4_ssot_5f      |      10 |               40 |              1 | none            |                 1 |        2.85343 |            0.221503 |  1.03866 |       0.195531 |   0.100096 |       0.516414 |             10 |

## Top 25 By Sharpe

| kind     | factor_name              |   top_n |   rebalance_days |   industry_cap | filter_mode        |   hold_multiplier |   total_return |   annualized_return |   sharpe |   max_drawdown |   avg_cash |   avg_turnover |   avg_holdings |
|:---------|:-------------------------|--------:|-----------------:|---------------:|:-------------------|------------------:|---------------:|--------------------:|---------:|---------------:|-----------:|---------------:|---------------:|
| search   | v6_value_blend           |      10 |               60 |              2 | mom120_gt_neg10    |               1   |        3.38219 |            0.245023 |  1.28838 |       0.152307 |  0.123939  |       0.911972 |        9.96552 |
| search   | v6_lowvol_half           |      10 |               50 |              2 | none               |               1   |        3.22343 |            0.238227 |  1.20021 |       0.218459 |  0.0991532 |       0.579627 |       10       |
| search   | v6_value_blend_no_holder |      10 |               50 |              2 | none               |               1   |        3.65693 |            0.256303 |  1.18866 |       0.20483  |  0.110358  |       0.655792 |       10       |
| search   | v6_value_blend_no_holder |      12 |               50 |              2 | none               |               1.5 |        3.58947 |            0.253586 |  1.17511 |       0.201982 |  0.129763  |       0.413668 |       12       |
| baseline | v5_quality_half          |      10 |               60 |              1 | mom120_gt_neg10    |               1   |        3.1408  |            0.234603 |  1.16864 |       0.141908 |  0.10904   |       0.833844 |       10       |
| search   | v6_value_blend           |      12 |               60 |              2 | mom120_gt_neg10    |               1   |        2.67196 |            0.212794 |  1.1623  |       0.158448 |  0.143217  |       0.886253 |       11.8966  |
| search   | v6_value_blend           |      12 |               60 |              2 | mom120_gt_neg10    |               1.5 |        2.67405 |            0.212896 |  1.16192 |       0.176779 |  0.145802  |       0.789305 |       11.8966  |
| search   | v6_lowvol_half           |       8 |               50 |              2 | not_deep_downtrend |               1   |        3.43806 |            0.247364 |  1.15089 |       0.236371 |  0.0848175 |       0.668737 |        8       |
| search   | v6_value_blend           |      10 |               60 |              2 | mom120_gt_neg10    |               1.5 |        2.7357  |            0.215894 |  1.14553 |       0.263886 |  0.130862  |       0.83956  |       10       |
| search   | v6_value_blend           |      10 |               60 |              2 | not_deep_downtrend |               1   |        2.66738 |            0.21257  |  1.14129 |       0.18172  |  0.133791  |       0.87973  |       10       |
| search   | v6_lowvol_half           |      10 |               50 |              2 | not_deep_downtrend |               1   |        2.97344 |            0.227072 |  1.1387  |       0.217893 |  0.100166  |       0.623851 |       10       |
| search   | v6_value_blend           |      10 |               60 |              1 | mom120_gt_neg10    |               1.5 |        2.35396 |            0.196608 |  1.12955 |       0.150753 |  0.134189  |       0.828042 |       10       |
| search   | v6_quality_holder        |      10 |               60 |              1 | mom120_gt_neg10    |               1.5 |        2.35632 |            0.196733 |  1.12522 |       0.185079 |  0.118807  |       0.702525 |        9.96552 |
| search   | v6_quality_lowvol        |      10 |               50 |              2 | none               |               1   |        2.98527 |            0.227613 |  1.11986 |       0.20814  |  0.108365  |       0.595367 |       10       |
| search   | v6_value_blend           |      12 |               60 |              1 | mom120_gt_neg10    |               1.5 |        2.27686 |            0.192488 |  1.11867 |       0.158097 |  0.145379  |       0.770715 |       11.7241  |
| search   | v6_lowvol_half           |       8 |               50 |              2 | none               |               1   |        3.21259 |            0.237755 |  1.11626 |       0.242216 |  0.0868952 |       0.650205 |        8       |
| search   | v6_value_blend           |      12 |               60 |              2 | not_deep_downtrend |               1   |        2.44153 |            0.201192 |  1.11299 |       0.191194 |  0.147128  |       0.883593 |       11.8966  |
| search   | v6_value_blend           |      12 |               60 |              1 | not_deep_downtrend |               1.5 |        2.20661 |            0.188661 |  1.11254 |       0.135295 |  0.144892  |       0.730138 |       11.8276  |
| search   | v6_value_blend           |      12 |               50 |              2 | none               |               1.5 |        2.54705 |            0.206584 |  1.10276 |       0.172989 |  0.13953   |       0.581888 |       11.9706  |
| search   | v6_value_blend_no_holder |      12 |               50 |              2 | none               |               1   |        2.91658 |            0.224451 |  1.1017  |       0.193034 |  0.136739  |       0.6273   |       11.9706  |
| search   | v6_quality_lowvol        |      12 |               50 |              2 | none               |               1.5 |        2.68193 |            0.213282 |  1.09611 |       0.19192  |  0.125952  |       0.348126 |       12       |
| search   | v6_value_blend           |       8 |               60 |              2 | mom120_gt_neg10    |               1.5 |        2.94941 |            0.225968 |  1.09487 |       0.206218 |  0.106002  |       0.883759 |        8       |
| search   | v6_value_blend           |       8 |               60 |              1 | mom120_gt_neg10    |               1.5 |        2.85905 |            0.221767 |  1.09015 |       0.199802 |  0.106585  |       0.899464 |        8       |
| search   | v6_lowvol_half           |      12 |               50 |              2 | none               |               1   |        2.41746 |            0.199942 |  1.08937 |       0.193089 |  0.122063  |       0.504875 |       12       |
| search   | v6_value_blend_no_holder |       8 |               50 |              2 | none               |               1   |        3.45533 |            0.248083 |  1.08797 |       0.230958 |  0.0990649 |       0.687332 |        8       |

## Suggested v6 Default

| kind   | factor_name    |   top_n |   rebalance_days |   industry_cap | filter_mode     |   hold_multiplier |   total_return |   annualized_return |   sharpe |   max_drawdown |   avg_cash |   avg_turnover |   avg_holdings |
|:-------|:---------------|--------:|-----------------:|---------------:|:----------------|------------------:|---------------:|--------------------:|---------:|---------------:|-----------:|---------------:|---------------:|
| search | v6_value_blend |      10 |               60 |              2 | mom120_gt_neg10 |                 1 |        3.38219 |            0.245023 |  1.28838 |       0.152307 |   0.123939 |       0.911972 |        9.96552 |