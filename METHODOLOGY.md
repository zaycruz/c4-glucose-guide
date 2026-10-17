# Methodology — C4 Glucose Guide

C4 Glucose Guide estimates how a walk after a meal and a late dinner go with the blood-sugar rise after that meal. It uses double machine learning (double ML) within person. It turns the estimates into advice with an `evidence[]` item behind every number. It also forecasts each person's typical rise after dinner. A large language model (Llama 3.3 70B on Workers AI) may rephrase the advice. A code check then rejects any rewrite that adds a number or a causal word. The program data is synthetic. Real data (CGMacros, NHANES) is used as a check.

**Where the numbers live**

- `README.md`: headline results and limits.
- `REPRODUCE.md`: the README-number → file → key table (52 rows), seeds and pinned versions.
- `ML_COMPARISON.md`: statistics vs ML benchmark; `public/data/mlcompare.json`, `public/data/pycaret.json`.
- `ROBUSTNESS.md`: negative control, E-value, instrument, real-data checks, power, subgroup test.
- `research/LEAKAGE.md`: leakage list for the prediction research loop.
- `public/data/ml_effects.json` (ML engine), `public/data/cohort.json`, `public/data/realworld.json`.
- `runs/reproduce-20261017T184202Z.md` (fresh-clone run), `runs/e2e-latest-summary.json`, `public/data/verification.json`.
- `CONTRACT.md`: API shapes, including `/advice`.

---

## Part A. Causal effect estimation (double ML)

### 1. Question and target

- **Question.** In the same person, is the rise after a meal smaller when they walk after it? Is it larger when dinner starts late?
- **Outcome (rise).** Peak of the 15-minute readings from 15 to 120 minutes after the meal, minus the baseline. The baseline is the last reading at or before the meal (at most 15 minutes earlier). Unit: mg/dL. A second outcome, iAUC, is the area above baseline over 0 to 120 minutes (mg/dL × h). Code: `scripts/analyze.py` (`meal_responses`).
- **Horizon.** 2 hours after the meal start.
- **Treatment 1, walk.** `walked_after_min > 0` (`walked`). Walk-length bins: 1-9, 10-19, 20-29, 30+ minutes (`WALK_BINS`).
- **Treatment 2, late dinner.** Dinner starting at 20:00 or later, or before 04:00 (`late`, `LATE_HOUR = 20`). Dinner-hour bins against "before 6 pm" (`HOUR_BINS` in `scripts/ml_effects.py`).
- **Estimand.** The within-person average effect in a partially linear model: the change in rise for a meal with the exposure vs the same person's meals without it, given the meal's content. Walking was self-chosen. So this is an association, not proof (`README.md`, `scripts/ml_effects.py` docstring).
- **Overnight fast** is a separate, randomized question: the arm contrast (coaching + overnight fast vs coaching) on the week 12 minus week 0 change in HbA1c. It uses no ML (`scripts/analyze.py`, `main`, `trial`).

Source: `scripts/analyze.py` docstring, `README.md`.

### 2. Data

| Part | Real or synthetic | Size | Unit of analysis |
|---|---|---|---|
| Challenge 4 synthetic program (Crane Lab) | Synthetic. "Nobody in it is real." | 2,000 people, 305,154 meals, 15-minute glucose for 500 sensor wearers | Meal (clean meal curve), within person |
| Program meals used | Synthetic | 500 sensor wearers, 18,218 clean meal curves | Meal |
| CGMacros v1.0.0 (PhysioNet 2025) | Real | 45 adults, 10 days, Libre CGM + Fitbit; 44 with walk data (1,329 meals), 41 with late dinners (314 dinners) | Meal, within person |
| NHANES 2011-2014 | Real | 11,329 US adults, 808 cancer survivors | Person (one snapshot), survey-weighted |

Exclusions (program, `meal_responses`):
- A meal followed by another item of 100 kcal or more within 120 minutes is dropped, so the curve belongs to one meal.
- The meal needs a baseline reading, at least 6 post-meal readings, and at least 10 g of carbs.
- Double ML keeps people with at least 3 meals (`dml`, `min_meals=3`).
- Night-shift workers are left out of the dinner and late-dinner analyses.

CGMacros exclusions (`scripts/realworld.py`, `cgmacros_meals`): another logged meal within 120 minutes, under 10 g carbs, fewer than 100 minutes of the 2-hour window. The real walk is at least 10 straight minutes at 3+ METs (Fitbit) in the hour after the meal, on meals with 50+ minutes of Fitbit data. The 3 METs and 10 minutes are a researcher choice (`ROBUSTNESS.md`).

No data is committed. `data/SHA256SUMS` checks every input file.

Source: `data/README.md`, `README.md`, `ROBUSTNESS.md`, `public/data/ml_effects.json` (`real.walk10`, `real.lateDinner`).

### 3. Features and leakage controls

**Nuisance features in the shipped engine** (`scripts/ml_effects.py`, `MACROS`, `MT`, `dml`):
- Meal content: calories, carbs, protein, fat, fiber, sugar.
- Meal type (breakfast, dinner, snack; lunch is the reference) for the all-meal walk models. Dinner-only models use the macros only.
- Mundlak person means: each person's mean of every feature (`pm_` prefix).
- Within person: the outcome and the exposure are demeaned per person (`demean`).
- CGMacros has no sugar column, so its models use calories, carbs, protein, fat and fiber.

The shipped effect models do not use pre-meal sugar or hour as features. The benchmark in `ML_COMPARISON.md` section D does, and gives −18.0 for the walk against the shipped −17.8.

**Leakage controls.**
- The rise, peak, iAUC and the curve points are outcomes. They are never features (`ML_COMPARISON.md`, "How it could fail").
- For effect estimation, the walk is the exposure. It is known only after the meal. `ML_COMPARISON.md` says this is fine for effects and is leakage for "predict before you eat". So the forecast (Part B) does not use it.
- Negative control outcome uses the reading 15 minutes before the meal, because the meal time is floored to 15 minutes (`ROBUSTNESS.md`).
- `research/LEAKAGE.md` holds the full leakage list (outcome columns, week-12 checkup, same-day daily columns, walk after the meal, own-history look-ahead, cross-person target encoding, holdout peeking). It applies to the prediction research loop (`research/score.py`, `research/features.py`), not to the effect engine.

Source: `scripts/ml_effects.py`, `ML_COMPARISON.md`, `ROBUSTNESS.md`, `research/LEAKAGE.md`.

### 4. Split and cross-fitting

- **Cross-fitting.** 5 folds, repeated 3 times; repeat `rep` uses `random_state=SEED + rep` with `SEED = 20261017`. The final estimate is the median over the 3 repeats. The SE is the median of `sqrt(se² + (est − median)²)` (`dml`, `FOLDS`, `REPEATS`).
- **How folds split the data.** Folds are `StratifiedKFold` on the person code. So each person's meals spread across all folds, and every person is in every training fold. The split is **not** grouped by participant. This is by design for a within-person estimand (the benchmark code says "rows stratified by person, so every person is in every training fold", `scripts/ml_compare.py`, `est_dml`).
- Person demeaning and the Mundlak person means use all of that person's rows, before the folds are made.
- **Inference is by person.** The 95% CI uses a person-clustered sandwich SE (Part A, section 7).
- **New-people checks** exist only for prediction: grouped 5-fold by person (`GroupKFold`, `scripts/ml_compare.py`, `predict_task`).

**Tuning protection.** No hyperparameter search was run for the effect engine. Gradient boosting uses one fixed setting everywhere: `max_iter=200, learning_rate=0.05, min_samples_leaf=40, random_state=0` (`HGB` in `scripts/ml_effects.py` and `scripts/ml_compare.py`). `ML_COMPARISON.md` wrote its decision rule and failure list before the run, and says "Use library defaults with light, fixed settings on both sides; no per-scenario tuning." There is no separate validation or test set for effects. The engine choice itself was changed after a look: the pre-written rule kept fixed effects as the engine, and zay then overrode it ("lets use ML ... update our submission to use ML instead"). `ML_COMPARISON.md` records this override.

Source: `scripts/ml_effects.py`, `scripts/ml_compare.py`, `ML_COMPARISON.md`.

### 5. Model and selection

- **Model.** Partially linear double ML, within person (`scripts/ml_effects.py`, `dml`). One `HistGradientBoostingRegressor` learns the exposure from the features. A second one learns the outcome. The effect is the least-squares slope of the outcome residual on the exposure residual.
- **Joint exposures.** Walk-length bins and dinner-hour bins are fitted together, like dummies in one regression.
- **Fits run.** Walk (all meals, after dinner, after other meals), late dinner, carbs per 10 g, iAUC variants, by blood-sugar group, walk dose, dinner hour, the negative control, the fairness groups, and CGMacros walk and late dinner (`main`).
- **Personal effects.** Each person's effect is the slope on their own cross-fitted residuals. Their meals are bootstrapped 300 times for the SE (`person_effects`, `B_PERSON = 300`). People need at least 3 meals with and 3 without the exposure. `scripts/analyze.py` then blends the personal effect with the cohort effect by empirical Bayes (`shrink`, `dl_tau2`). A personal number is used in advice only if the person's own CI and the blended CI both exclude 0 and their own data has at least 25% of the weight (`main`, `sw["weight"] >= 0.25`).
- **Why this engine.** `ML_COMPARISON.md` tested four estimators on a planted −10 mg/dL effect (20 repetitions; `effect_benchmark`). The benchmark uses 3-fold DML (`est_dml`, `folds=3`).

| Estimator | Linear confounding: bias · RMSE · coverage | Nonlinear confounding: bias · RMSE · coverage |
|---|---|---|
| Naive difference | +10.9 · 11.0 · 0% | +12.0 · 12.1 · 0% |
| Fixed effects (statistics) | +1.0 · 1.2 · 65% | +3.5 · 3.6 · 0% |
| S-learner (gradient boosting) | +1.0 · 1.2 · no CI | +1.4 · 1.5 · no CI |
| Double ML (gradient boosting, shipped) | −0.2 · 0.7 · 85% | +0.3 · 0.7 · 95% |

- **Planted personal effects** (SD 5, 10 repetitions; `personal_benchmark`): own data only 11.0 RMSE; empirical-Bayes blend on fixed-effects residuals 4.6; double-ML residuals blended (shipped) 4.7; S-learner per person 4.9.
- **Calibration.** Not applicable to the effect estimates. No calibration step exists.

Source: `ML_COMPARISON.md`, `public/data/mlcompare.json`.

### 6. Baseline

There is no organisers' baseline in the sources. The comparators are:
- **Naive difference** and **fixed-effects regression** in the planted-effect benchmark (table above), same features, same budget, fixed settings (`ML_COMPARISON.md`).
- **Fixed-effects regression as a live cross-check** on the program data (`scripts/analyze.py`, `fe_effect`; evidence items `cohort_walk_fe`, `cohort_late_fe`). Walk: −17.8 (double ML) vs −17.6 (regression). Late dinner: +9.2 vs +9.1. If they ever disagree, the evidence item says so.

Source: `ML_COMPARISON.md`, `README.md`, `REPRODUCE.md`.

### 7. Metrics with 95% CIs

**How the CIs are computed.**
- Double ML: person-clustered sandwich SE, est ± 1.96 SE, median over 3 cross-fitting repeats (`dml`).
- Fixed-effects cross-check and CGMacros fixed effects: cluster bootstrap over people, 400 resamples, 2.5th and 97.5th percentiles (`fe_effect`, `B_COHORT = 400`).
- Randomized-arm contrasts: bootstrap of each arm's people, 2,000 resamples, percentile interval (`main`, `trial`).
- Personal effects: bootstrap of that person's meals, then the empirical-Bayes blend (`person_effects`, `shrink`).

**Program results (synthetic, double ML).** Signs: negative means a smaller rise.

| Effect | Estimate (mg/dL) | 95% CI | People · meals | CI excludes 0? |
|---|---|---|---|---|
| Walk, all meals | −17.77 | −19.19 to −16.35 | 500 · 18,218 | yes |
| Walk after dinner | −18.31 | −20.08 to −16.55 | 471 · 4,693 | yes |
| Walk after other meals | −14.61 | −16.51 to −12.70 | 500 · 13,323 | yes |
| Late dinner (8 pm or later) | +9.21 | +7.09 to +11.32 | 471 · 4,693 | yes |
| Walk, type 2 diabetes | −23.32 | −26.75 to −19.89 | 125 people | yes |
| Walk, normal blood sugar | −13.18 | −14.62 to −11.74 | 230 people | yes |

README rounding: walk after dinner −18.3 (16.6 to 20.1 lower), late dinner +9.2 (7.1 to 11.3), all meals 17.8.

Walk dose (vs no walk): 10-19 min 17.5 lower; 20-29 min 19.5; 30+ min 16.6. Dinner hour (vs before 6 pm): 6-7 pm +3.9, 7-8 pm +8.4, 8-9 pm +13.3, 9-10 pm +18.2, 10 pm or later +25.4.

**Real-data results (CGMacros, double ML, same engine).** This is the primary result in the README.

| Effect | Estimate (mg/dL) | 95% CI | People · meals | CI excludes 0? |
|---|---|---|---|---|
| Walk after a meal | −3.2 | −6.89 to 0.48 | 44 · 1,329 | **no** ("not distinguishable from zero") |
| Late dinner | +2.61 | −5.11 to 10.33 | 41 · 314 | **no** |

- The real walk interval excludes the program's −17.8. The README says: "expect a much smaller effect in real people."
- The program's late-dinner +9.2 is still inside the real interval.
- CGMacros fixed effects for the walk: −4.4 (−11.4 to 1.4).

**Randomized trial (no ML).** Adding the overnight fast to coaching changed HbA1c by −0.03 points (−0.06 to 0.00). Coaching vs email tips: −0.17. The fast arm fasted 13.05 h vs 12.07 h, with 62% vs 36% of nights at 13 h or more.

**Significant vs not, in the repo's own words.**
- Program walk and late-dinner effects: CIs exclude 0.
- Real walk and real late dinner: "Both point the same way but are not proven."
- Overnight fast by blood-sugar group: "a formal test can't tell the blood sugar groups apart (Q p = 0.15; Holm-adjusted p = 0.13), so the advice uses the overall number."

Source: `public/data/ml_effects.json` (`cohort`, `byGroup`, `real`), `README.md`, `ROBUSTNESS.md`, `REPRODUCE.md`.

### 8. Robustness and falsification

Each check, with how it could fail, is in `ROBUSTNESS.md` (failure modes written before the code).

| Check | Result | Verdict (repo wording) |
|---|---|---|
| Negative control: pre-meal reading 15 min before, walk meals vs other meals, same person (`dml` on `pre15`) | −0.35 (−1.38 to 0.68), against a walk effect of −17.8 | "Passes: walk meals don't start from a different place" |
| E-value (`e_value`, within-person residual SD) | 3.9 (CI bound 3.6) | A hidden factor would need a risk ratio of about 3.9 with both walking and the rise |
| Randomized arm as instrument, coaching vs email tips (`iv_walk`) | First stage +3.7 pts (2.5 to 5.0); walk effect −31 (−122 to 59) | "Inconclusive." Exclusion restriction fails. "Not counted as support" |
| Real data, CGMacros, same engine | −3.2 (−6.9 to 0.5), 1,329 meals, 44 people | "Same direction, not proven." The real effect is "probably much smaller" |
| NHANES survivors, survey-weighted, Rao-Wu bootstrap of PSUs within strata (`nhanes_weighted`, `design_boot_weights`, 500 replicates) | Fast: +0.013 per hour (−0.025 to 0.048). Movement: −0.021 per 1,000 MIMS (−0.052 to 0.006) | Fast: no link. "Movement loses its clear signal once weighted" |
| Power, scaled from CGMacros (`people_needed`) | Late dinner: 59 people for +9.2, about 730 for +2.6. Walk: 4 for the program size, 120 for −3.2 | The program's walk number "is too big for real people" |
| Fast effect by blood-sugar group: Cochran Q and Holm across 10 subgroup contrasts | Q p = 0.15; type 2 diabetes p = 0.13 after Holm | "Does not hold." Advice uses the overall −0.03 |

**Real-data label on every recommendation** (`agreement`): `agrees` (same sign, CI excludes 0), `same direction, not proven`, `no clear signal`, `disagrees` (opposite sign, CI excludes 0), `not tested`. Current labels: walk and late dinner "same direction, not proven"; overnight fast "no clear signal" (NHANES); safety rules "not tested".

**Claims retracted after the checks** (`ROBUSTNESS.md`): "The fast gain is concentrated in type 2 diabetes." "More movement goes with lower HbA1c in NHANES."

**Fairness by group** (`fairness_table`, `q_test`; served as `cohort.fairness`). Attributes: race/ethnicity, sex, age, preferred language, cancer type. Groups under 20 people get no effect estimate. Cochran Q per attribute, Holm across 10 tests.
- Only the walk effect by sex differs after Holm (p = 0.02): −20.3 in men vs −15.4.
- Forecast error is similar in every group: 11.9 to 16.2 mg/dL.
- Race and ethnicity: walk Q p = 0.12, late dinner Q p = 0.37 (Holm p = 0.59 and 1.0).
- Age 70+ walk −14.7 vs −19.4 under 60, not after Holm (p = 0.07). Kreyòl speakers' late-dinner +1.9 (Q p = 0.03), not after Holm (p = 0.20).
- The data is synthetic. This checks that the engine treats groups alike. It does not measure real-world fairness.

Source: `ROBUSTNESS.md`, `README.md`, `public/data/cohort.json`.

### 9. Limits

Copied from the README and ROBUSTNESS.md, not softened:
- The program data is synthetic. "Nobody in it is real." The generator "is probably additive, which favours linear models" (`ML_COMPARISON.md`).
- Walking was self-chosen. The walk effect is an association.
- In 44 real adults the walk effect is −3.2 (−6.9 to +0.5): "not distinguishable from zero". The real interval excludes the program's −17.8, so expect a much smaller effect in real people. So the advice offers the walk "as a low-risk habit, not a proven fix".
- The real late-dinner effect (+2.6, −5.1 to 10.3) is not proven. Confirming an effect that small "would take hundreds of people".
- The instrument check failed its exclusion restriction and is not counted as support.
- Power scaling "assumes new people look like these 45" CGMacros adults.
- Medicines are not recorded in this data.
- Fairness is checked on synthetic data only.
- Who the 45 CGMacros adults are (race, ethnicity, language, insurance) is not described (README "Future work").

### 10. Reproduce

- Command: `scripts/reproduce.sh <Challenge4_Dose_and_Timing dir> repro_out`. It checks the data hashes, runs `scripts/realworld.py`, `scripts/ml_effects.py`, `scripts/ml_compare.py`, `scripts/analyze.py`, then `scripts/headline.py`, which diffs every README number. It exits 0 only if all match.
- Seeds: 20261017 for the engine, the benchmark and the analysis; 7 for the real-data bootstrap; `random_state=0` for every model. `PYTHONHASHSEED=0`.
- Pinned environment: Python 3.12 and `uv.lock` (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1). The optional pycaret leaderboard is not pinned.
- Proof run: a fresh clone at `f242363` on a Cloudflare Sandbox (4 vCPU), exit 0, **52/52 README numbers reproduced**, total 17 min 10 s. Later commits change wording and the `/advice` shape, not the model or any README number.
- Where each number lives: the table in `REPRODUCE.md` section 3.

Source: `REPRODUCE.md`, `scripts/reproduce.sh`, `runs/reproduce-20261017T184202Z.md`.

---

## Part B. Dinner-rise forecast (gradient boosting)

### 1. Question and target

Forecast the rise (same definition as Part A) after a meal, before eating. Each person's summary shows the mean forecast over their dinners (`ml_forecast_dinner`).

### 2. Data

Synthetic program: 18,218 meals, 500 people. Real check: CGMacros, 1,338 meals, 45 people (`mlcompare.json`, `predictCgmacros`).

### 3. Features and leakage controls

`scripts/ml_effects.py` (`forecast`): calories, carbs, protein, fat, fiber, sugar, meal type, baseline (pre-meal sugar), hour, the 15-minute pre-meal trend (`pre_slope`), Mediterranean style, and the person's mean rise. The person's mean rise is computed from training rows only. The walk is not a feature. Outcome columns are not features.

### 4. Split

- The shipped forecast is out-of-fold: `StratifiedKFold`, 5 folds on the person code, seed 20261017. "Each meal forecast by a model that never saw it." The same person's other meals are in training. The folds are random across meals, not by time.
- New-people and later-days splits are measured in the benchmark (`scripts/ml_compare.py`, `predict_task`): each person's later 25% of meals by time (`order >= 0.75`; `ML_COMPARISON.md` calls it "later 25% of days"), and grouped 5-fold by person.
- Tuning protection: fixed `HGB` settings, no search (as in Part A, section 4). The autoresearch protocol in `research/LEAKAGE.md` defines a locked holdout (people with sha256(id) mod 5 == 0), scored once at the end. Only its frozen scorer and baseline features are committed; no experiment results are committed, and the shipped forecast does not come from it.

### 5. Model and selection

`HistGradientBoostingRegressor` with the fixed `HGB` settings. Selection came from the benchmark below and from the pycaret leaderboard (18 models, default settings, grouped 5-fold): LightGBM 16.9 · 0.44 and gradient boosting 17.2 · 0.43 on program new people; gradient boosting 23.7 · 0.29 on CGMacros. No calibration check was done ("a calibration check" is listed under README "Future work").

### 6. Baseline

"Person average (baseline)" and a linear model with per-person intercept, same features, same folds (`models_for`).

### 7. Metrics with 95% CIs

CIs: bootstrap over people, 200 resamples (`metrics`, `B_METRIC = 200`).

| Data and split | Model | MAE (95% CI) | R² (95% CI) |
|---|---|---|---|
| Program, out-of-fold (shipped forecast) | Gradient boosting | 14.0 | 0.62 |
| Program, later days, same people | Person average | 20.3 | 0.20 |
| | Linear, per-person intercept | 15.0 (14.28 to 15.68) | 0.57 (0.53 to 0.60) |
| | Gradient boosting | 14.73 (14.08 to 15.37) | 0.56 (0.50 to 0.60) |
| Program, new people (grouped 5-fold) | Person average | 23.58 | 0.00 |
| | Linear | 18.45 (17.90 to 19.10) | 0.35 (0.33 to 0.37) |
| | Gradient boosting | 16.91 (16.23 to 17.55) | 0.44 (0.42 to 0.46) |
| CGMacros, new people | Person average | 28.83 | −0.01 |
| | Linear | 26.54 (23.95 to 29.24) | 0.11 (−0.01 to 0.20) |
| | Random forest | 23.68 (21.51 to 26.05) | 0.30 (0.20 to 0.37) |
| | Gradient boosting | 24.34 (21.93 to 27.17) | 0.24 (0.12 to 0.35) |

No CI is published for the shipped 14.0 / 0.62.

Significant vs not (`ML_COMPARISON.md`): for known people "it's a tie: the MAE intervals overlap (linear 14.3–15.7, boosting 14.1–15.4)". For new people ML wins by about 1.5 mg/dL MAE and 0.09 R². On real data "ML wins clearly (R² 0.30 vs 0.11)".

### 8. Robustness

Forecast error by group: 11.9 to 16.2 mg/dL (`fairness`). Real-data R² 0.30 (random forest, CGMacros new people).

### 9. Limits

README: "Expect the lower figures for a new patient" (R² 0.44 for new people, 0.30 on real adults, against 0.62 for people it knows). No calibration check. No clinic-baseline comparison with a CI. No model-file hash proving the served forecast equals the evaluated one (README "Future work").

### 10. Reproduce

Same command as Part A. `forecast.mae` and `forecast.r2` are in `ml_effects.json`; the benchmark rows are in `mlcompare.json` (`REPRODUCE.md` section 3).

Source: `public/data/ml_effects.json` (`forecast`), `public/data/mlcompare.json`, `ML_COMPARISON.md`, `README.md`.

---

## Part C. Generative AI: advice wording (Llama on Workers AI)

**One call.** `GET /api/c4/survivors/:id/advice` (`src/index.ts`, `phrase`, `advice`).

- **Model.** `@cf/meta/llama-3.3-70b-instruct-fp8-fast` only. `fallbackModel: null`. Workers AI labels its responses `@cf/meta/llama-3.3-70b-instruct-sd`. The API passes this through as `providerModelName` and says it is the provider's name for the same deployment. Settings: `temperature: 0.3`, `max_tokens: 1200`.
- **Input.** The system prompt asks for warm, plain English (reading age 12), two short sentences per item, "Use ONLY numbers that appear in the given text; never add, change or round numbers", "Do not add medical claims", and no upgrade of the wording to caused, helped, lowered or reduced. The user message is the person's template recommendations as `[{id, text}]`. No raw glucose data, profile or evidence values go to the model.
- **Allowed output.** JSON `{"items":[{"id","text"}]}`. An item is used only if its id matches a template recommendation.
- **Guardrail 1, numbers** (`numbers`, `allowedNumbers`, `untraced`). Every number in the rewrite (absolute value) must match, within 0.051, a number in the template text or in the cited evidence items (value, n, CI bounds).
- **Guardrail 2, causal words** (`CAUSAL`, `addedCausal`). The rewrite may not add any of caus\*, help/helps/helped, lower/lowers/lowered, reduc\*, prevent\*, cure/cures/cured that the template does not already use.
- **On failure of either check**, that item falls back to the checked template text: `source: "template"`, with `rejected: {text, untracedNumbers, addedCausalWords}`.
- **On model error**, every item uses the template. `error` reads "AI rewrite unavailable; showing the checked template text". The raw error and raw provider response are never returned.
- **Dosing.** The model never sets content. The templates carry the safety routing: people with lows get only the care-team card, and walk cards carry a medication caution ("The coach does not change medication"). There is no dedicated dosing filter in the Worker. A dose number would be rejected only if it is not already in the template or the evidence. The prompt's "Do not add medical claims" is an instruction, not a check.
- **Cache.** `scripts/warm-advice.mjs` precomputes the rewrite for every survivor into `public/data/advice/<id>.json`. The Worker serves it only while `recHash`, the SHA-256 of the current `[id, text]` recommendations, matches (`recHash`). `?fresh=1` calls the model live.
- **What users see.** The demo button "Rephrase with Workers AI" replaces the card text, and a note reads "N of M rephrased in X ms · R rejected (number not in evidence, or causal wording)". Each API item has `source: "ai"` or `"template"`, and the response names `model`. The survivor-facing `patientView` (en, es, ht) is built from fixed rule text in `scripts/patient_copy.py`, not by the model.

**What was measured** (latest E2E, `runs/e2e-latest-summary.json`, `public/data/verification.json`):
- `scripts/e2e.mjs` checks that one survivor's advice adds no untraced numbers and no new causal verbs, that the model fields name the one model and the no-fallback rule, that no raw model error or raw provider response leaks, and (with `--strict`) that cached advice answers in under 2 s.
- Latest run: pass, 50 survivors, 147 recommendations, 737 advice numbers traced to evidence. The AI check ran on one survivor (c4-N0072): 1 of 3 items rephrased by AI. Cached advice was served in 58 ms. That run was not `--strict`.

**What is NOT evaluated.**
- Readability, warmth or reading age of the rewrites. No human or judge review was run.
- Meaning drift that keeps the same numbers and avoids the listed causal words (for example a changed direction, dropped caveat, or a new claim without a number).
- Medical safety of new wording beyond the two code checks; dosing language is not checked as such.
- The share of items rejected or falling back to template across all survivors. It is not recorded in a committed file.
- p95 latency, cost and fallback rate over 30+ calls, and an alert on a changed `providerModelName` (README "Future work").
- Spanish and Kreyòl text is machine-translated and marked `reviewed: false`. It is not made by this call and not reviewed by a native speaker.

Source: `src/index.ts`, `CONTRACT.md`, `README.md`, `scripts/e2e.mjs`, `scripts/warm-advice.mjs`, `public/demo.html`, `runs/e2e-latest-summary.json`.

---

## Code map

| Step | File (function) |
|---|---|
| Data check | `scripts/reproduce.sh` (step 0), `data/SHA256SUMS` |
| Load data | `scripts/analyze.py` (`load`) |
| Meal response, outcome, exposures, exclusions | `scripts/analyze.py` (`meal_responses`) |
| Double ML effects, cross-fitting, clustered CI | `scripts/ml_effects.py` (`dml`, `demean`) |
| Personal effects | `scripts/ml_effects.py` (`person_effects`); blend in `scripts/analyze.py` (`shrink`, `dl_tau2`) |
| Dinner-rise forecast | `scripts/ml_effects.py` (`forecast`) |
| Fairness table | `scripts/ml_effects.py` (`fairness_table`, `q_test`) |
| Fixed-effects cross-check | `scripts/analyze.py` (`fe_effect`, `fe_design`, `fe_fit`) |
| Randomized-arm contrasts, subgroup Q, Holm | `scripts/analyze.py` (`main`, `holm`) |
| Negative control | `scripts/ml_effects.py` (`main`, `negativeControl`) |
| E-value | `scripts/analyze.py` (`e_value`) |
| Instrument | `scripts/analyze.py` (`iv_walk`) |
| Power | `scripts/analyze.py` (`people_needed`) |
| Real-data label | `scripts/analyze.py` (`agreement`) |
| CGMacros meals and checks | `scripts/realworld.py` (`cgmacros_meals`, `cgmacros`) |
| NHANES survey-weighted | `scripts/realworld.py` (`nhanes_weighted`, `design_boot_weights`, `wls_design`) |
| Status rule (CGM consensus) | `scripts/analyze.py` (`status_from`) |
| Stats vs ML benchmark | `scripts/ml_compare.py` (`predict_task`, `metrics`, `effect_benchmark`, `personal_benchmark`, `est_dml`, `est_fe`) |
| pycaret leaderboard | `scripts/pycaret_compare.py` |
| Prediction research loop (protocol only) | `research/score.py`, `research/features.py`, `research/LEAKAGE.md`, `research/run.sh` |
| Patient copy (fixed text, 3 languages) | `scripts/patient_copy.py` (`patient_view`, `plain_summary`) |
| AI rewrite and guards | `src/index.ts` (`phrase`, `allowedNumbers`, `untraced`, `addedCausal`) |
| Advice cache | `src/index.ts` (`advice`, `recHash`), `scripts/warm-advice.mjs` |
| README-number diff | `scripts/headline.py` (`main`) |
| E2E check | `scripts/e2e.mjs` |
