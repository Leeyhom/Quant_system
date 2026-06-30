# v9 2025 Path Sensitivity

- Cached JoinQuant pool coverage: 80/152
- Purpose: mechanism check for warm-path vs cold-start differences.
- Caveat: local cached pool may differ from the real JoinQuant 152-stock run.

## Metrics

| case                          |   total_return |   annualized_return |   sharpe |   max_drawdown |   avg_exposure |   avg_cash |   avg_holdings |   start_value |   end_value |
|:------------------------------|---------------:|--------------------:|---------:|---------------:|---------------:|-----------:|---------------:|--------------:|------------:|
| warm_full_v9_total            |      3.34952   |           0.243641  | 1.09695  |       0.190161 |       0.914327 |  0.0856733 |        9.96469 |       59908.7 |    260574   |
| warm_full_v9_slice_2025       |      0.121951  |           0.126192  | 0.802158 |       0.126339 |       0.901048 |  0.098952  |        9.7541  |      232250   |    260574   |
| warm_full_v10_ramp_total      |      2.96862   |           0.226851  | 1.08074  |       0.205979 |       0.896845 |  0.103155  |        9.96469 |       59917   |    237788   |
| warm_full_v10_ramp_slice_2025 |      0.119317  |           0.123461  | 0.788003 |       0.129271 |       0.901407 |  0.0985931 |        9.7541  |      212440   |    237788   |
| cold_2025_v9_immediate        |      0.0803031 |           0.083398  | 0.602419 |       0.146835 |       0.834895 |  0.165105  |        9.75309 |       59911.7 |     64722.8 |
| cold_2025_v9_fixed95          |      0.0781245 |           0.0811324 | 0.594736 |       0.145082 |       0.801902 |  0.198098  |        9.50617 |       59912.5 |     64593.1 |
| cold_2025_v10_ramp_80_90_98   |      0.101243  |           0.105183  | 0.832563 |       0.115327 |       0.731609 |  0.268391  |        9.50617 |       59922.6 |     65989.3 |
| cold_2025_v10_ramp_70_85_98   |      0.121092  |           0.125848  | 0.990624 |       0.112878 |       0.730967 |  0.269033  |        9.75309 |       59923.8 |     67180.1 |

## Rebalance Phase

- Warm full-run v9 rebalances inside 2025: 2025-03-13, 2025-06-12, 2025-09-04, 2025-12-05
- Cold 2025 v9 rebalances: 2025-01-02, 2025-04-07, 2025-07-04, 2025-09-26, 2025-12-29
- First cold-start targets overlap with first warm-2025 rebalance targets: 5/10
- Overlap: 000725, 002241, 002475, 600019, 600887

## Cold-Start Interpretation

The same factor can produce a different 2025 result because v9 is stateful:
carried positions, the 60-trading-day rebalance phase, and the 1.5x holding buffer all depend on when the strategy starts.
The ramp variants are not new alpha. They are live-start risk controls that reduce the capital committed before the first few rebalances have confirmed the path.

## Outputs

- v9_2025_path_sensitivity.csv
- v9_2025_path_sensitivity_rebalances.csv