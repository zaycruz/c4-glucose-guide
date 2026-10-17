# Autoresearch: predicting a meal's rise — how it could fail (written before the code)

Goal (zay, 2026-10-17): "i want to absoluely crush hese metrics, run an autoresearch". Scope is **prediction only**:
the rise forecast and the prediction benchmark. Effect estimation is not optimised. Its planted-effect bias (+0.3,
95% coverage) is already at the ceiling, and searching covariates there would buy narrower intervals with bias.
New features enter the double-ML effect models only if they pass the planted-effect benchmark with no rise in bias.

## Protocol
- `research/score.py` is frozen and committed before the first experiment. The loop edits only `research/features.py`.
- **Locked holdout:** people whose sha256(id) mod 5 == 0 (about 20%, both datasets). Every experiment scores on
  grouped 5-fold CV over the remaining people. The holdout is scored once, at the end (`--final`).
- Fixed seeds. A change is kept only if the paired person-bootstrap 95% CI of ΔMAE excludes 0 on its target split
  and no other split gets significantly worse. The final winner is rerun once before it is reported.
- Budget: about 20 experiments. Stop early once 3 kept changes in a row each gain less than the noise.
- A big jump is a red flag. If program new-people R² goes above about 0.7, run two checks before believing it:
  a target-shuffle run (R² must be about 0) and a drop-one-group ablation that shows where the gain comes from.
- The program data is synthetic: trees can learn its generator. A small real (CGMacros) gain outweighs a large synthetic one.

## Splits (each reports MAE, RMSE, R²)
1. `program_new_cold`: new people, no history of their own. Only enrolment, week-0 and meal-time inputs.
2. `program_new_online`: new people, but each meal may use that person's own **earlier** meals (what the product sees after a few days).
3. `program_known_later`: the same people's later 25% of days. Training is the earlier 75%; history is allowed.
4. `cgmacros_new_online`: real adults, new people, own-history allowed. A `cold` variant is reported too.

## Leakage list: each item has a check in score.py, or a rule in features.py
| Risk | Rule | Check |
|---|---|---|
| Outcome columns (peak, iauc, c15…c180, rise) as features | never | score.py rejects any column whose name matches them, and any column with abs corr > 0.98 to the target |
| Week-12 checkup (after the outcome) | week 0 only | the feature namespace `chk0_` is built from visit == week 0 only; score.py asserts there is no `week12`/`wk12` column |
| Same-day daily columns (steps, active_minutes, walked_after_dinner_min, calories, last_ate, eating_window_hours, evening_calories_pct, energy, hunger, alcohol, sleep of the coming night) | use the **previous day's** row only (`prev_` prefix) | score.py asserts every daily-derived column starts with `prev_` |
| Walk after the meal (`walked_after_min`, post-meal steps) | excluded ("before you eat") | name check in score.py |
| `finished_program`, `days_in_program` | excluded (known only afterwards) | name check |
| History of the person's own rises | only meals whose 2-hour window ended before this meal started | the shift test: score.py recomputes history with a +1-meal look-ahead, and that copy must score clearly better (proves the guard is doing something) |
| Person target encoding across people | inside each fold, from training people only | done in score.py, not in features.py |
| Pre-meal glucose trajectory | readings at or before the meal's 15-minute slot only | features.py builds them from offsets ≤ 0 |
| Holdout peeking | the holdout is dropped before features.py sees any outcomes from it in CV runs | score.py drops holdout people unless `--final` |
| Tuning on noise | paired bootstrap ΔMAE; rerun the winner | logged in research/results.tsv |
| Test meals selected using the future (the "clean" filter uses the next meal) | it is a selection rule, the same for every model; not a feature | – |
