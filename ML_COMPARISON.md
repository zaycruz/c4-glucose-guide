# Statistics vs machine learning

Should Glucose Guide use an ML model instead of (or next to) the statistical models? The benchmark below tests both kinds on the two jobs the product has:

1. **Predict** how much a meal will raise sugar.
2. **Estimate the effect** of a walk (or a late dinner), with an interval, overall and per person. This is what the advice runs on.

It is set up so either side can win. Script: `scripts/ml_compare.py` → `public/data/mlcompare.json`.

## Decision rule (written before running)

- If DML (ML nuisance models) recovers planted effects better than fixed effects under nonlinear confounding, **and** agrees with fixed effects on the real data, it becomes a cross-check evidence item. It does not replace fixed effects.
- If ML predicts the rise much better, that is a candidate feature ("predicted rise for this meal"). It gets proposed, not built.
- If an ML personal-effect estimate beats the empirical-Bayes blend on planted per-person effects, the "from your own data" gate is revisited.

## How it could fail (written before the code)

**Leakage**
- Columns that define the outcome (peak, iAUC, the curve points c15…c180) must never be features.
- A person's average rise used as a feature (target encoding) must come from training rows only, including inside every cross-fitting fold.
- Temporal split: the test weeks must come strictly after the training weeks for each person.
- Walk length is only known after the meal. That is fine for estimating effects. For "predict before you eat" it is leakage, so report both variants of the prediction task: with and without walk.

**Unfair comparison**
- The program data comes from a generator that is probably additive, which favours linear models. Say so, and also run the prediction task on real data (CGMacros), where nonlinearity would actually show.
- Linear confounding only would rig the effect benchmark for fixed effects. Also run a nonlinear-confounding scenario (assignment driven by a threshold on pre-meal glucose and by hour), where fixed effects should struggle.
- Tuning ML heavily but not the linear model, or the reverse. Use library defaults with light, fixed settings on both sides; no per-scenario tuning.

**Effect benchmark**
- The real walk effect contaminating the planted one. Use only meals with no real walk as the base.
- Planted treatment too common or too rare compared with reality. Target about 10% of meals, like real walks.
- An S-learner has no native confidence interval. Report coverage as not available; do not invent one.
- DML standard errors that ignore clustering by person would be too narrow. Cluster by person.
- Too few repetitions to tell estimators apart. Use at least 20 per scenario, with fixed seeds, and report the Monte Carlo spread.

**Personal-effect benchmark**
- Personal effects planted with no real spread would make shrinkage look perfect. Draw them with a realistic spread (SD about 5 mg/dL around -10).
- Scoring people who got no planted treatment at all. Score only people with at least 3 treated and 3 untreated meals, and use the same people for every estimator.

**Runtime**
- Gradient boosting × folds × repetitions can grow. Keep it a bounded command; if it passes a few minutes, it moves to Proxmox.

## Results (`public/data/mlcompare.json`, seed 20261017, rerun gives identical output)

Ran on Proxmox (pve-03, about 2 minutes). Every model got the same pre-meal inputs: macros, calories, pre-meal sugar and its 15-minute trend, hour, meal type and Mediterranean style.

### A. Predicting the rise (mg/dL; 95% CI from a bootstrap over people)

| Data and split | Person average | Linear (statistics) | Random forest | Gradient boosting |
|---|---|---|---|---|
| Program, same people, later 25% of days | MAE 20.3, R² 0.20 | **MAE 15.0, R² 0.57** | MAE 15.1, R² 0.52 | **MAE 14.7, R² 0.56** |
| Program, new people (grouped 5-fold) | MAE 23.6, R² 0.00 | MAE 18.5, R² 0.35 | MAE 17.1, R² 0.43 | **MAE 16.9, R² 0.44** |
| CGMacros real adults, new people | MAE 28.8, R² −0.01 | MAE 26.5, R² 0.11 | **MAE 23.7, R² 0.30** | MAE 24.3, R² 0.24 |

- For people we already know, it's a tie: the MAE intervals overlap (linear 14.3–15.7, boosting 14.1–15.4). A personal baseline plus straight lines already gets most of what there is.
- For new people, ML wins by about 1.5 mg/dL MAE and 0.09 R².
- On real data, ML wins clearly (R² 0.30 vs 0.11). Real responses are less linear than the synthetic generator.
- With walk length added (known only after the meal), every model improves a little. The ranking stays the same.
- For new people, "linear" means one pooled regression, because nobody in the test fold has a personal intercept.

**pycaret leaderboard** (`public/data/pycaret.json`): all 18 regression models pycaret ships (CatBoost excluded for time), default settings, the same features, 5-fold grouped by person. Run on pve-01, Python 3.11.

| Rank, program data (new people) | MAE · R² | Rank, CGMacros (new people) | MAE · R² |
|---|---|---|---|
| 1. LightGBM | 16.9 · 0.44 | 1. Gradient boosting | 23.7 · 0.29 |
| 2. Gradient boosting | 17.2 · 0.43 | 2. Random forest | 24.2 · 0.26 |
| 3. Random forest | 17.6 · 0.40 | 3. Extra trees | 24.3 · 0.25 |
| 6. Linear regression | 18.5 · 0.35 | 12. Linear regression | 26.4 · 0.12 |
| 15. Dummy (mean) | 23.6 · 0.00 | 16. Dummy (mean) | 28.7 · −0.02 |

This independently confirms the hand-built benchmark: tree ensembles on top, linear models in the middle, the same R² gap. No exotic model changes the picture.

### B. Recovering a planted −10 mg/dL effect from confounded data (20 repetitions)

| Estimator | Linear confounding: bias · RMSE · 95% CI coverage | Nonlinear confounding: bias · RMSE · coverage |
|---|---|---|
| Naive difference | +10.9 · 11.0 · 0% | +12.0 · 12.1 · 0% |
| Fixed effects (statistics) | +1.0 · 1.2 · 65% | **+3.5 · 3.6 · 0%** |
| S-learner (gradient boosting) | +1.0 · 1.2 · no CI | +1.4 · 1.5 · no CI |
| **Double ML (gradient boosting, shipped)** | **−0.2 · 0.7 · 85%** | **+0.3 · 0.7 · 95%** |

Double ML wins both scenarios. Even with "linear" assignment, fixed effects understates the effect by about 10%, because the real rise depends on pre-meal sugar and carbs in ways a straight line does not fully capture. Under threshold-shaped confounding, fixed effects understates it by a third and its interval never covers the truth. That was the risk this benchmark was built to expose.

### C. Planted personal effects (spread SD 5, about 300 people scored per repetition, 10 repetitions)

| Estimator | RMSE vs planted personal effect |
|---|---|
| Own data only (OLS) | 11.0 |
| **Blend with cohort (empirical Bayes on fixed-effects residuals)** | **4.6** |
| Double ML residuals, blended with cohort (shipped) | 4.7 |
| S-learner per person (gradient boosting) | 4.9 |

The two blends are within 0.15 mg/dL of each other (4.57 vs 4.72); both beat own-data-only by more than half. The shipped version keeps the same empirical-Bayes blend and gate, but takes each person's raw effect from the double-ML residuals. The plain S-learner is close behind and has no interval, so it can't drive the gate.

### D. Real program effects, same inputs for both

| Effect | Fixed effects | Double ML |
|---|---|---|
| Walk after a meal | −19.2 (−20.5 to −17.9) | −18.0 (−19.2 to −16.7) |
| Dinner at 8 pm or later | +9.0 (6.8 to 11.2) | +9.7 (7.4 to 12.0) |

The intervals overlap. With pre-meal sugar and hour added, fixed effects gives −19.2, against −17.6 in the advice, which adjusts for carbs and meal type only.

## What changed

The first decision, per the rule above, kept fixed effects as the engine and added double ML as a cross-check. zay then overrode that: "lets use ML ... update our submission to use ML instead". The submission now runs on ML:

- **Every meal-level effect comes from double ML** (`scripts/ml_effects.py`, run on Proxmox by `scripts/build-ml.sh`): the walk and late-dinner effects, walk dose, dinner hour, groups, the negative control and the CGMacros real-data checks. It is within-person (outcome and exposure demeaned per person, plus person-mean features), uses gradient-boosting nuisance models, 5-fold cross-fitting repeated 3 times, and person-clustered intervals. Several exposures are fitted jointly (walk-length bins, dinner-hour bins), so hours with little overlap are still estimated.
- **Fixed effects is the cross-check now** (`cohort_walk_fe`, `cohort_late_fe`). The two agree: walk −17.8 (ML) vs −17.6; late dinner +9.2 vs +9.1. If they ever disagree, the evidence item says so.
- **Personal effects** use each person's own double-ML residuals, then the same empirical-Bayes blend and the same gate (own CI excludes 0, own data ≥25% of the weight). On planted personal effects this scores 4.7 vs 4.6 for the fixed-effects version, a negligible cost.
- **Built: an ML forecast** of each person's typical dinner rise (gradient boosting, out-of-fold, so no person's own meals train their forecast): MAE 14.0 mg/dL, R² 0.62. It appears in each summary and as evidence `ml_forecast_dinner`, with the error in `ml_forecast_error`.
- **Still statistics, because no model is needed:** randomized-arm contrasts for the overnight fast, survey-weighted NHANES, and the sensor consensus rules for status.
