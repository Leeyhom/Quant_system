# LLM Context

This file is a compact orientation document for LLMs and coding agents working on this repository.

## Current Truth

The current A-share JoinQuant baseline is:

```text
scripts/joinquant_cn_sim_strategy_v8.py
```

v8 is a robust recovery version after v7 failed in real JoinQuant backtest exports. It intentionally avoids score-tilted sizing and keeps the v6-like equal-weight, momentum-filtered structure.

Do not treat `joinquant_cn_sim_strategy_v7.py` as the best strategy. It is an important failed experiment.

## Most Important Recent Evidence

Read this first:

```text
docs/21_joinquant_v7_failure_v8_recovery.md
```

Key lesson:

```text
Before optimizing parameters, make local validation use the same stock pool as the JoinQuant strategy.
```

v7 failed because local validation used a smaller local pool while JoinQuant ranked a larger live candidate pool. The strategy also removed the 120-day momentum filter and added score-tilted sizing, which amplified weak value traps in 2021-2022.

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
scripts/joinquant_cn_sim_strategy_v8.py
scripts/analyze_joinquant_exports.py
scripts/refetch_joinquant_pool.py
```

## Common Mistakes

- Optimizing on full-sample return and calling it alpha.
- Ignoring that A-share local `DEFAULT_POOL` and JoinQuant `STOCK_POOL` may differ.
- Adding a new factor without checking coverage and timestamp availability.
- Reporting strategy return without benchmark, drawdown, fees and turnover.
- Forgetting A-share 100-share lot constraints in small-capital simulation.
- Committing `data/raw/`, `.env`, webhook URLs, or broker credentials.

## Current Next Step

The next scientifically useful improvement is not another parameter sweep. It is:

```bash
NO_PROXY='*' PYTHONPATH=. python scripts/refetch_joinquant_pool.py
```

Then rerun validation on the exact JoinQuant stock pool and only then design v9.
