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

### B. Recovering a planted −10 mg/dL effect from confounded data (20 repetitions)

| Estimator | Linear confounding: bias · RMSE · 95% CI coverage | Nonlinear confounding: bias · RMSE · coverage |
|---|---|---|
| Naive difference | +10.9 · 11.0 · 0% | +12.0 · 12.1 · 0% |
| Fixed effects (statistics, used now) | +1.0 · 1.2 · 65% | **+3.5 · 3.6 · 0%** |
| S-learner (gradient boosting) | +1.0 · 1.2 · no CI | +1.4 · 1.5 · no CI |
| **Double ML (gradient boosting)** | **−0.2 · 0.7 · 85%** | **+0.3 · 0.7 · 95%** |

Double ML wins both scenarios. Even with "linear" assignment, fixed effects understates the effect by about 10%, because the real rise depends on pre-meal sugar and carbs in ways a straight line does not fully capture. Under threshold-shaped confounding, fixed effects understates it by a third and its interval never covers the truth. That was the risk this benchmark was built to expose.

### C. Planted personal effects (spread SD 5, about 300 people scored per repetition, 10 repetitions)

| Estimator | RMSE vs planted personal effect |
|---|---|
| Own data only (OLS) | 11.0 |
| **Blend with cohort (empirical Bayes, used now)** | **4.7** |
| S-learner per person (gradient boosting) | 5.1 |

The empirical-Bayes blend wins narrowly, and both beat own-data-only by more than half. ML pulls toward the cohort too (trees share strength), but less precisely.

### D. Real program effects, same inputs for both

| Effect | Fixed effects | Double ML |
|---|---|---|
| Walk after a meal | −19.2 (−20.5 to −17.9) | −18.0 (−19.2 to −16.7) |
| Dinner at 8 pm or later | +9.0 (6.8 to 11.2) | +9.7 (7.4 to 12.0) |

The intervals overlap. With pre-meal sugar and hour added, fixed effects gives −19.2, against −17.6 in the advice, which adjusts for carbs and meal type only.

## What changed (per the decision rule)

- **Double ML is now a cross-check evidence item** (`cohort_walk_dml`, `cohort_late_dml`), cited on the walk and late-dinner advice. It beat fixed effects on planted effects and agrees with it on the real data, so the advice numbers don't depend on a straight-line model. Fixed effects stays the advice engine: it agrees, it is simpler to explain, and the personal blend builds on it.
- **Personal effects stay with the empirical-Bayes blend.** It beat the ML per-person estimate.
- **Proposed, not built: "predicted rise for this meal".** For new people and real data, gradient boosting or a random forest predicts the rise better than linear models. That fits a meal-planning feature ("this dinner will likely raise you about 45"), but it is a product call.
- **Why ML isn't the whole engine:** the advice needs effect sizes with honest intervals that trace to evidence, not predictions. Plain ML gives predictions; it gives causal effects only when wrapped in a design like double ML, and then it agrees with the statistics.
