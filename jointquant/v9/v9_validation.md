# JoinQuant v9 Local Validation

- Window: 20190101 ~ 20251231
- Cached JoinQuant pool coverage: 80/152
- Cost: 60k CNY, 100-share lots, min commission, stamp duty, transfer fee, slippage
- Factor: same as v8, no holder, no full-sample IC direction.
- Limitation: this is not final until the full 152-stock JoinQuant pool is cached.

## Config Comparison

| config                     |   top_n |   rebalance_days |   total_return |   annualized_return |   sharpe |   max_drawdown |   avg_turnover |   avg_cash |   avg_exposure |   avg_holdings |   avg_skipped_slots |
|:---------------------------|--------:|-----------------:|---------------:|--------------------:|---------:|---------------:|---------------:|-----------:|---------------:|---------------:|--------------------:|
| v9_buffer_equal            |      10 |               60 |        3.01841 |            0.229122 | 1.09909  |       0.201845 |       0.724738 |  0.126862  |       0.873138 |        9.96552 |                   0 |
| v9_buffer_less_cash_98     |      10 |               60 |        3.34952 |            0.243641 | 1.09695  |       0.190161 |       0.756428 |  0.0880516 |       0.911948 |        9.96552 |                   0 |
| v8_less_cash_98            |      10 |               60 |        3.11447 |            0.233436 | 1.07082  |       0.188746 |       0.831165 |  0.0945879 |       0.905412 |        9.96552 |                   0 |
| v8_baseline_equal          |      10 |               60 |        2.79051 |            0.218524 | 1.05642  |       0.199224 |       0.789141 |  0.125229  |       0.874771 |       10       |                   0 |
| v9_buffer_vol_soft_brake   |      10 |               60 |        2.58101 |            0.208291 | 1.04594  |       0.203707 |       0.727893 |  0.125613  |       0.874387 |       10       |                   0 |
| v9_buffer_vol_budget       |      10 |               60 |        2.60027 |            0.209253 | 1.04167  |       0.202855 |       0.726704 |  0.125545  |       0.874455 |       10       |                   0 |
| v9_buffer_vol_less_cash_98 |      10 |               60 |        2.73633 |            0.215924 | 1.02692  |       0.205895 |       0.745197 |  0.101025  |       0.898975 |       10       |                   0 |
| v8_top12                   |      12 |               60 |        2.25146 |            0.191113 | 0.987278 |       0.211071 |       0.7589   |  0.140222  |       0.859778 |       11.9655  |                   0 |
| v8_top8                    |       8 |               60 |        2.80017 |            0.218984 | 0.987188 |       0.195303 |       0.825494 |  0.107365  |       0.892635 |        8       |                   0 |
| v8_rebalance50             |      10 |               50 |        1.95781 |            0.174507 | 0.885779 |       0.248184 |       0.726616 |  0.13252   |       0.86748  |       10       |                   0 |

## v9 Decision

Best Sharpe in this run: `v9_buffer_equal`.
Production candidate: `v9_buffer_less_cash_98`.

- Total return delta vs v8 baseline: +55.90%
- Sharpe delta vs v8 baseline: +0.041
- Max drawdown delta vs v8 baseline: -0.91%

Decision for the JoinQuant v9 file: use `v9_buffer_less_cash_98`
(same factor score as v8, target exposure 98%, keep existing holdings that
remain inside the top 1.5 * TOP_N score band). The bounded volatility budget
remains a research toggle but is disabled by default because it underperformed
in this cached-pool run.

This still needs full 152-stock cache validation and a JoinQuant export replay
before replacing v8 as the production baseline.