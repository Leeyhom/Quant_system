# LLM Context

This file is a compact orientation document for LLMs and coding agents working on this repository.

## Current Truth

The strongest confirmed A-share JoinQuant historical baseline is:

```text
scripts/joinquant_cn_sim_strategy_v9.py
```

v9 keeps the v8 factor score, raises target exposure from 95% to 98%, and keeps existing holdings that remain inside the top 1.5 * TOP_N score band. Real JoinQuant export replay for 2019-01-01 ~ 2025-12-31 shows +192.29% strategy return, better than v6/v7, with improved alpha and related metrics.

Do not treat `joinquant_cn_sim_strategy_v7.py` as the best strategy. It is an important failed experiment.

The latest paper/live cold-start candidate is:

```text
scripts/joinquant_cn_sim_strategy_v10.py
```

v10 does not change the v9 alpha. It only adds startup exposure ramping for new cash accounts:

```text
70% -> 85% -> 98%
```

Why: v9 can be negative when the JoinQuant backtest starts only on 2025-01-01. That is mainly path dependence: carried positions, rebalance phase, holding buffer state, and 100-share lot constraints differ from the full 2019-2025 run. This is also relevant to paper/live simulation because those accounts start from cash.

## Most Important Recent Evidence

Read this first:

```text
docs/21_joinquant_v7_failure_v8_recovery.md
docs/22_platform_backtest_and_v9.md
docs/23_v9_2025_cold_start_v10.md
```

Key lesson:

```text
Before optimizing parameters, distinguish full-window historical performance from cold-start paper/live deployment.
```

v7 failed because local validation used a smaller local pool while JoinQuant ranked a larger live candidate pool. The strategy also removed the 120-day momentum filter and added score-tilted sizing, which amplified weak value traps in 2021-2022.

v9 succeeded on the full JoinQuant run but exposed cold-start sensitivity in 2025-only testing. The next validation step is to run v10 in JoinQuant on both 2019-2025 and 2025-only windows.

## Safe Work Order

For strategy work:

1. Inspect current code and docs.
2. Identify the exact data universe and date window.
3. Check future-function risk.
4. Run local validation.
5. If JoinQuant exports exist, run:

```bash
PYTHONPATH=. python scripts/analyze_joinquant_exports.py jointquant/<version> <version>
```

6. Write the conclusion to `docs/`.

## Files To Read

```text
README.md
AGENTS.md
docs/21_joinquant_v7_failure_v8_recovery.md
docs/23_v9_2025_cold_start_v10.md
scripts/joinquant_cn_sim_strategy_v8.py
scripts/joinquant_cn_sim_strategy_v9.py
scripts/joinquant_cn_sim_strategy_v10.py
scripts/analyze_joinquant_exports.py
scripts/joinquant_v9_2025_attribution.py
scripts/joinquant_v9_path_sensitivity.py
scripts/refetch_joinquant_pool.py
scripts/export_joinquant_v9_targets.py
```

## Common Mistakes

- Optimizing on full-sample return and calling it alpha.
- Ignoring that A-share local `DEFAULT_POOL` and JoinQuant `STOCK_POOL` may differ.
- Adding a new factor without checking coverage and timestamp availability.
- Reporting strategy return without benchmark, drawdown, fees and turnover.
- Forgetting A-share 100-share lot constraints in small-capital simulation.
- Committing `data/raw/`, `.env`, webhook URLs, or broker credentials.

## Current Next Step

The next scientifically useful improvement is to run v10 in JoinQuant:

```text
scripts/joinquant_cn_sim_strategy_v10.py
```

Run both 2019-01-01 ~ 2025-12-31 and 2025-01-01 ~ 2025-12-31. Export transaction/position/log into `jointquant/v10/`, then run `analyze_joinquant_exports.py jointquant/v10 v10`.

Separately fix the AkShare/py_mini_racer data issue, rerun validation on the exact 152-stock JoinQuant pool, and keep using real JoinQuant exports as the arbiter for strategy versions.
