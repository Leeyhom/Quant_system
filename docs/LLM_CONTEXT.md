# LLM Context

This file is the compact orientation document for LLMs and coding agents working on this repository.

## Current Truth

The current A-share JoinQuant paper-trading line is:

```text
scripts/joinquant_cn800_strategy_v5.py
```

Status:

- CN800 v5 is connected to JoinQuant paper trading.
- The next milestone is not a new alpha version. It is to observe v5 for at least one natural month, then import the paper-trading exports into the repository.
- Put future paper-trading exports under `jointquant/cn800_v5_paper/` and keep raw `transaction.csv`, `position.csv`, `log.txt` out of git unless explicitly requested.
- Generate tracked summaries from those exports, then use the summaries for the next strategy iteration.

Confirmed historical context:

- `scripts/joinquant_cn_sim_strategy_v9.py` is the strongest pre-CN800 historical JoinQuant baseline.
- `scripts/joinquant_cn_sim_strategy_v10.py` is a cold-start/ramp lesson for cash accounts, not the current production line.
- `scripts/joinquant_cn800_strategy_v4.py` was a strong CN800 candidate.
- `scripts/joinquant_cn800_strategy_v5.py` is now the current CN800 production/paper baseline after adding volatility target and ROE stability.

Do not treat v8/v9/v10 as the current strategy unless the user explicitly asks for historical comparison.

## Most Important Recent Evidence

Read these first:

```text
README.md
docs/24_cn800_v5_paper_trading_plan.md
docs/AUDIT_专业量化审计报告.md
```

Then use the older history for context:

```text
docs/21_joinquant_v7_failure_v8_recovery.md
docs/22_platform_backtest_and_v9.md
docs/23_v9_2025_cold_start_v10.md
```

Key lesson:

```text
Before optimizing parameters, distinguish historical backtest alpha from paper/live deployment evidence.
```

The one-month CN800 v5 paper run is primarily an execution and data-fidelity test. It should not be overinterpreted as enough evidence to tune alpha.

## Safe Work Order

For strategy work:

1. Inspect current code and docs.
2. Identify the exact data universe, date window and benchmark.
3. Check future-function risk: price look-ahead, financial-statement availability, and universe membership look-ahead.
4. Run local validation only when the local pool and platform pool are aligned.
5. If JoinQuant exports exist, run:

```bash
PYTHONPATH=. python scripts/analyze_joinquant_exports.py jointquant/<version-dir> <version>
```

6. For CN800 paper-trading evidence, write a short tracked summary in `jointquant/cn800_v5_paper/` and update `docs/24_cn800_v5_paper_trading_plan.md` or a follow-up doc.

## Files To Read

```text
README.md
AGENTS.md
CONTRIBUTING.md
docs/24_cn800_v5_paper_trading_plan.md
scripts/joinquant_cn800_strategy_v5.py
scripts/joinquant_cn800_strategy_v4.py
scripts/cn800_walkforward.py
scripts/cn800_v4_engine.py
scripts/analyze_joinquant_exports.py
```

For historical lessons:

```text
scripts/joinquant_cn_sim_strategy_v9.py
scripts/joinquant_cn_sim_strategy_v10.py
scripts/joinquant_v9_2025_attribution.py
scripts/joinquant_v9_path_sensitivity.py
docs/21_joinquant_v7_failure_v8_recovery.md
docs/22_platform_backtest_and_v9.md
docs/23_v9_2025_cold_start_v10.md
```

## Common Mistakes

- Optimizing on full-sample return and calling it alpha.
- Treating the latest static CN800 constituent list as historically available without disclosure.
- Ignoring that A-share local pools and JoinQuant pools may differ.
- Adding a new factor without checking coverage and timestamp availability.
- Reporting strategy return without benchmark, drawdown, fees, turnover and execution constraints.
- Forgetting A-share 100-share lot constraints in small-capital simulation.
- Letting an LLM/agent directly change live or paper-trading parameters without a validation gate.
- Committing `data/raw/`, `.env`, webhook URLs, broker credentials, or raw platform exports by accident.

## Current Next Step

Do not start a new strategy version by default.

Current priority:

1. Let `scripts/joinquant_cn800_strategy_v5.py` run in JoinQuant paper trading for one month.
2. Import the paper-trading exports into `jointquant/cn800_v5_paper/`.
3. Reconcile paper trading against the historical backtest: holdings, cash, order failures, turnover, slippage and drawdown path.
4. Only after that decide whether the next iteration should target alpha, execution, risk, or data quality.
