# Reproduce C4 Glucose Guide

One command takes the raw Challenge 4 data to every headline number in [README.md](README.md) and diffs them.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.9 or later. It installs Python 3.12 and the exact packages in `uv.lock` (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1). Nothing else is needed for the numbers.
- git, bash, curl, and `sha256sum` or `shasum`.
- For the Worker and the E2E only: Node 22 and `npm ci` (versions locked in `package-lock.json`).
- A machine with 4 CPU cores and 8 GB of RAM: about 10 minutes. It also runs on a laptop CPU, more slowly.

## 1. Data

No data is committed. Get `Challenge4_Dose_and_Timing.zip` from the organizers and unzip it outside the repo. [data/README.md](data/README.md) lists the files, their SHA-256 sums (`data/SHA256SUMS`, checked automatically) and the public sources: CGMacros on PhysioNet, and NHANES from the CDC.

## 2. One command

```sh
git clone https://github.com/zaycruz/c4-glucose-guide && cd c4-glucose-guide
scripts/reproduce.sh /path/to/Challenge4_Dose_and_Timing repro_out
```

What it runs, in order (each step's log is `repro_out/<step>.log`, and timings go to `repro_out/steps.tsv`):

| Step | Script | Writes | Time (4 vCPU) |
|---|---|---|---|
| 0 | check the data against `data/SHA256SUMS` | | seconds |
| 1 | `uv sync --frozen` | `.venv` | under 1 min |
| 2 | `scripts/realworld.py`: CGMacros and survey-weighted NHANES checks | `realworld.json` | under 1 min |
| 3 | `scripts/ml_effects.py`: **the ML engine** (double ML, personal effects, dinner forecast) | `ml_effects.json` | about 4 min |
| 4 | `scripts/ml_compare.py`: statistics vs ML (prediction, planted effects, personal effects) | `mlcompare.json` | about 4 min |
| 5 | `scripts/analyze.py`: evidence, advice, roster, cohort (the Worker's data) | `roster.json`, `cohort.json`, `survivors/*.json` | under 1 min |
| 6 | `scripts/headline.py`: diff every README number | `headline.md` | seconds |

It exits 0 only if every number matches. It writes only to the out dir; `public/data/` (what the Worker serves) is left alone. To rebuild what is served instead, see "Run" in the README.

Seeds are fixed in the code: 20261017 for the engine, the benchmark and the analysis; 7 for the real-data bootstrap; `random_state=0` for every model. A rerun gives the same JSON.

Not part of the one command: the optional pycaret leaderboard (`scripts/pycaret_compare.py`, Python 3.11). It confirms the benchmark, and no README number depends on it. Its pycaret version was not recorded, so it is not pinned; the result is committed in `public/data/pycaret.json`.

## 3. README number → where it comes from

`scripts/headline.py` holds this table and checks it. "Expected" is the README's figure; the produced value is rounded the same way. For each row the script also checks the figure still appears in README.md, and that the raw value equals the committed `public/data/` value. Values are in the out dir (or `public/data/`).

| README claim | File | Key | Expected |
|---|---|---|---|
| Sensor wearers in the program | `ml_effects.json` | `cohort.walk.people` | 500 |
| Clean meal curves | `ml_effects.json` | `cohort.walk.rows` | 18,218 |
| Walk after dinner: rise lower by | `ml_effects.json` | `cohort.walkAfterDinner.est` | 18.3 |
| 95% CI bound (near) | `ml_effects.json` | `cohort.walkAfterDinner.hi` | 16.6 |
| 95% CI bound (far) | `ml_effects.json` | `cohort.walkAfterDinner.lo` | 20.1 |
| Walk after other meals | `ml_effects.json` | `cohort.walkAfterOtherMeals.est` | 14.6 |
| Walk, all meals (double ML) | `ml_effects.json` | `cohort.walk.est` | 17.8 |
| Walk, all meals (regression cross-check) | `cohort.json` | `statistics.effects.walk.est` | 17.6 |
| Walk, type 2 diabetes | `ml_effects.json` | `byGroup.type 2 diabetes.walk.est` | 23.3 |
| Walk, normal blood sugar | `ml_effects.json` | `byGroup.normal.walk.est` | 13.2 |
| Negative control (pre-meal sugar) | `ml_effects.json` | `cohort.negativeControl.est` | -0.35 |
| CI low | `ml_effects.json` | `cohort.negativeControl.lo` | -1.38 |
| CI high | `ml_effects.json` | `cohort.negativeControl.hi` | 0.68 |
| E-value | `cohort.json` | `robustness.eValue.estimate` | 3.9 |
| Walk 10-19 min | `ml_effects.json` | `walkDose.1.est` | 17.5 |
| Walk 20-29 min | `ml_effects.json` | `walkDose.2.est` | 19.5 |
| Walk 30+ min | `ml_effects.json` | `walkDose.3.est` | 16.6 |
| Late dinner (8 pm or later) | `ml_effects.json` | `cohort.lateDinner.est` | 9.2 |
| CI low | `ml_effects.json` | `cohort.lateDinner.lo` | 7.1 |
| CI high | `ml_effects.json` | `cohort.lateDinner.hi` | 11.3 |
| Dinner 6-7 pm vs before 6 | `ml_effects.json` | `dinnerHour.1.est` | 3.9 |
| Dinner 7-8 pm | `ml_effects.json` | `dinnerHour.2.est` | 8.4 |
| Dinner 8-9 pm | `ml_effects.json` | `dinnerHour.3.est` | 13.3 |
| Dinner 9-10 pm | `ml_effects.json` | `dinnerHour.4.est` | 18.2 |
| Dinner 10 pm or later | `ml_effects.json` | `dinnerHour.5.est` | 25.4 |
| Fast vs coaching, HbA1c (randomized) | `cohort.json` | `trial.contrasts.hba1c_pct.coaching + overnight fast vs coaching.est` | -0.03 |
| CI low | `cohort.json` | `trial.contrasts.hba1c_pct.coaching + overnight fast vs coaching.lo` | -0.06 |
| CI high | `cohort.json` | `trial.contrasts.hba1c_pct.coaching + overnight fast vs coaching.hi` | 0.00 |
| Fast effect, type 2 diabetes | `cohort.json` | `trial.fastVsCoachingBySubgroup.hba1c_pct.blood sugar: type 2 diabetes.est` | -0.08 |
| Interaction test Q p | `cohort.json` | `trial.heterogeneity.hba1c_pct.p` | 0.15 |
| Holm-adjusted p (type 2 diabetes) | `cohort.json` | `trial.fastVsCoachingBySubgroup.hba1c_pct.blood sugar: type 2 diabetes.pHolm` | 0.13 |
| Fast arm: hours fasted | `cohort.json` | `trial.adherence.coaching + overnight fast.overnightFastHours` | 13.05 |
| Coaching arm: hours fasted | `cohort.json` | `trial.adherence.coaching.overnightFastHours` | 12.07 |
| Fast arm: nights at 13 h+ (%) | `cohort.json` | `trial.adherence.coaching + overnight fast.nightsAtLeast13hPct` | 62 |
| Coaching arm: nights at 13 h+ (%) | `cohort.json` | `trial.adherence.coaching.nightsAtLeast13hPct` | 36 |
| Coaching vs email tips, HbA1c | `cohort.json` | `trial.contrasts.hba1c_pct.coaching vs tips by email.est` | -0.17 |
| CGMacros real adults | `realworld.json` | `cgmacros.people` | 45 |
| CGMacros late dinner (double ML) | `ml_effects.json` | `real.lateDinner.est` | 2.6 |
| CI low | `ml_effects.json` | `real.lateDinner.lo` | -5.1 |
| CI high | `ml_effects.json` | `real.lateDinner.hi` | 10.3 |
| CGMacros walk after a meal (double ML) | `ml_effects.json` | `real.walk10.est` | -3.2 |
| CI low | `ml_effects.json` | `real.walk10.lo` | -6.9 |
| CI high | `ml_effects.json` | `real.walk10.hi` | 0.5 |
| Dinner forecast: typical miss (MAE) | `ml_effects.json` | `forecast.mae` | 14.0 |
| Dinner forecast: R² | `ml_effects.json` | `forecast.r2` | 0.62 |
| Planted -10 effect, double ML error | `mlcompare.json` | `effectBenchmark.nonlinear.Double ML (gradient boosting, shipped).bias` | 0.3 |
| Planted -10 effect, regression error | `mlcompare.json` | `effectBenchmark.nonlinear.Fixed effects (statistics).bias` | 3.5 |
| Regression interval coverage (never) | `mlcompare.json` | `effectBenchmark.nonlinear.Fixed effects (statistics).coverage95` | 0 |
| New people R², gradient boosting | `mlcompare.json` | `predictProgram.splits.new people (5-fold).Gradient boosting.r2.est` | 0.44 |
| New people R², linear | `mlcompare.json` | `predictProgram.splits.new people (5-fold).Linear, per-person intercept (fixed effects).r2.est` | 0.35 |
| Real data R², random forest | `mlcompare.json` | `predictCgmacros.splits.new people (5-fold).Random forest.r2.est` | 0.30 |
| Real data R², linear | `mlcompare.json` | `predictCgmacros.splits.new people (5-fold).Linear, per-person intercept (fixed effects).r2.est` | 0.11 |

## 4. E2E against the live Worker

```sh
npm ci
node scripts/e2e.mjs            # or: node scripts/e2e.mjs https://<your deployment>
node scripts/e2e.mjs --strict   # also requires cached advice and a published /verification that matches the live data
```

It checks the API contract, safety routing, the three-language patient view, and that every measured number in the advice traces to its evidence. The full artifact goes to `runs/e2e-<UTC time>.json` and a summary to `runs/e2e-latest-summary.json`. The deployed Worker publishes the latest result at `/api/c4/verification`.

## 5. Proof runs

Each fresh-clone run is logged in `runs/reproduce-<UTC stamp>.md`: where it ran, the commands, versions, timings, and the full diff table. On 2026-10-17 it ran on Cloudflare: a Sandbox container (4 vCPU) started by `infra/cf-runner`, with the data served from a private R2 bucket.

```sh
cd infra/cf-runner && npm ci && npx wrangler deploy        # once; needs RUNNER_TOKEN set with `wrangler secret put`
curl -X POST -H "authorization: Bearer $T" -d '{"repo":"https://github.com/zaycruz/c4-glucose-guide","ref":"main"}' https://c4-repro-runner.<account>.workers.dev/start
curl -H "authorization: Bearer $T" https://c4-repro-runner.<account>.workers.dev/status
```
