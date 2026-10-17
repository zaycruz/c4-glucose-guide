# C4 Glucose Guide

Coach Console module C4 (eMerge hackathon, Healthcare track, Sylvester): **how meals, meal timing and movement shape blood sugar**, per survivor and across the cohort, with honest uncertainty, turned into advice a survivor can follow. Every number in the advice traces to an `evidence[]` item.

- Live: https://c4-glucose-guide.zay-259.workers.dev (cohort view: `/#cohort`)
- API: see [CONTRACT.md](CONTRACT.md) (`/api/c4/survivors`, `/api/c4/survivors/:id`, `/api/c4/survivors/:id/advice`, `/api/c4/cohort`)
- Data: the Challenge 4 "Dose and Timing" set (the README inside calls it "Challenge 2"). Synthetic program for the survivors; CGMacros and NHANES as the real-data check.

## What it finds

Synthetic 12-week program: 500 sensor wearers, 18,218 clean meal curves. **The advice runs on machine learning:** double ML with gradient boosting, within person, with person-clustered 95% CIs (`scripts/ml_effects.py`). The classic fixed-effects regression stays as a cross-check, and the two agree. Randomized-arm contrasts and NHANES stay statistical, because randomization and survey weights need no model.

| Question | Answer |
|---|---|
| Is an evening walk worth it? | Yes. After dinner the rise is **18.3 mg/dL lower** (16.6 to 20.1), more than after other meals (14.6). All meals: 17.8 lower (regression cross-check: 17.6). Stronger in type 2 diabetes (23.3) than in normal blood sugar (13.2). Walking was self-chosen, so this is an association. It passes a negative control (pre-meal sugar −0.35, −1.38 to 0.68), and its E-value is 3.9. |
| How much is enough? | **10-19 min gets most of it** (17.5 lower); 20-29 min 19.5; 30+ min 16.6. Benefit levels off after 10 minutes. |
| Does a late dinner matter? | Dinner at 8 pm or later: **+9.2 mg/dL** (7.1 to 11.3). Each hour later adds more vs before 6 pm: 6-7 pm +3.9, 7-8 pm +8.4, 8-9 pm +13.3, 9-10 pm +18.2, 10 pm or later +25.4. |
| Overnight fast? | Randomized: adding the fast to coaching changed HbA1c by -0.03 points overall (-0.06 to 0.00). Type 2 diabetes looked larger (-0.08), but a formal test can't tell the blood sugar groups apart (Q p = 0.15; Holm-adjusted p = 0.13), so the advice uses the overall number. The fast arm did fast longer: 13.05 vs 12.07 h, with 62% vs 36% of nights at 13 h or more. Night shift (n=28) points the other way, but the CI is wide. Coaching itself vs email tips: -0.17. |
| Does it hold in real people? | CGMacros (45 real adults), same ML engine: late dinner +2.6 (-5.1 to 10.3) and walk after a meal -3.2 (-6.9 to 0.5). Both point the same way but are not proven. The real walk interval excludes the program's -17.8, so expect a much smaller walk effect in real people. The program's late-dinner +9.2 is still inside the real interval. NHANES survivors, survey-weighted: neither fasting length nor daily movement shows a clear link with HbA1c. |
| What does ML add for a person? | A **forecast of the rise after their typical dinner** (gradient boosting; each meal forecast by a model that never saw it). Typical miss: 14.0 mg/dL; R² 0.62 on meals it never saw from people it knows. |
| Why ML, and not the regression? | We benchmarked them head to head ([ML_COMPARISON.md](ML_COMPARISON.md)). With confounding shaped like a threshold, a planted -10 effect is recovered with an error of +0.3 by double ML and +3.5 by the regression, whose interval never covered the truth. ML also predicts the rise better for new people (R² 0.44 vs 0.35) and in real data (0.30 vs 0.11). On this program's data the two engines agree, so the switch changes the method, not the story. |

Every recommendation carries a real-data check label (`agrees`, `same direction, not proven`, `no clear signal`, `disagrees`, `not tested`). [ROBUSTNESS.md](ROBUSTNESS.md) lists each check, how it could fail, and what it found.

Status (ok / watch / alert) uses fixed CGM consensus thresholds (time in range, highs above 250, lows below 70). Personal effects come from the same ML engine: each person's effect is fitted on their own cross-fitted residuals, their meals are bootstrapped, and the result is blended with the cohort (empirical Bayes). They are reported only where the person's own data supports them. If they don't have enough data, the advice says so and uses the cohort number. Night-shift workers get no clock-time advice. People with frequent lows are told not to stretch their fasts.

## Demo story (2 minutes)

1. **Cohort first** (`/#cohort`, 40 s). "Does a late dinner matter? Same person, same carbs: every hour later adds to the rise, up to +25 mg/dL after 10 pm. Is an evening walk worth it? Yes: about 18 mg/dL lower after dinner, more than after other meals, and 10 minutes gets most of it. These numbers come from machine learning (double ML), checked against a classic regression. Overnight fast: the randomized arm changed HbA1c by -0.03, and we could not show it differs by blood sugar group. And here is the honest part: in 45 real people the walk and late-dinner effects point the same way, but they aren't proven yet."
2. **Roster** (`/`, 20 s). Each tile shows a plain headline and up to four signal tags (red = risk, amber = watch, green = working), so you can scan without opening anyone. Filter chips narrow it to Alert / Watch / Ok. Click **N0632** (alert, 76F, colorectal).
3. **N0632** (50 s). Headline: "Frequent lows: 5.3% of readings below 70". The tags read Lows 5.3%, Late dinners 57%, Walks work: -18.7 mg/dL, Fast 9.9 h. The KPI strip and the typical-day chart sit under the header. Each piece of advice comes with its own chart: tell the care team about the lows and don't stretch the fast (check-ups table); walk after dinner (her own walk vs no-walk curve, a personal estimate with a CI); eat dinner earlier (her early vs late dinner curves next to the cohort's). "Show the numbers" opens the evidence behind each number.
4. **Rephrase with Workers AI** (10 s). The text becomes warmer. The note shows how many rewrites were rejected, either for a number that isn't in the evidence or for causal words like "helped".

## Run

```sh
npm install
D=~/projects/emerge-hackathon/c4-data/Challenge4_Dose_and_Timing   # dataset extracted outside the repo
# 1. ML engine + statistics-vs-ML benchmark, on a Proxmox node (a few minutes of CPU; not on the Mac):
#    writes public/data/ml_effects.json and public/data/mlcompare.json. CLEAN=1 removes the node copy afterwards.
scripts/build-ml.sh root@pve-03 $D
# 2. the data the Worker serves (offline, ~10 s). analyze.py stops if ml_effects.json is missing.
scripts/build-data.sh $D
# optional: pycaret leaderboard on the same features (Python 3.11; FEATURES_DIR=<dir> on ml_compare.py writes them)
uv run --python 3.11 --with pycaret python -I scripts/pycaret_compare.py <dir> public/data/pycaret.json
npx tsc --noEmit          # after `npx wrangler types`
npx wrangler deploy       # -> https://c4-glucose-guide.<account>.workers.dev
node scripts/e2e.mjs [baseUrl]   # E2E against the deployment; writes runs/e2e-<time>.json
python3 -I scripts/contract.py [baseUrl]   # regenerate CONTRACT.md from live responses
```

No dev server needed; verification runs against the deployed Worker. Raw data is never committed (`.gitignore`). Only the derived JSON in `public/data/` is.

## E2E check

`scripts/e2e.mjs` hits `/`, the roster, every survivor panel, the cohort, one AI advice call, the 404 and malformed-id cases, and the CORS preflight. It asserts:
- the contract shapes;
- roster status matches panel status;
- every `evidenceId` resolves;
- **every measured number in every recommendation is in its cited evidence**: exact wording constants such as "8 pm", "13 hours", "70-180" and "95% CI" are removed first, and whatever number remains must trace;
- headline and summary numbers match the person's evidence within ±0.5;
- every tag's `evidenceId` resolves, its numbers match that evidence, and the tags on the roster tile match the panel;
- AI rewrites add no untraced numbers and no new causal verbs (helped, caused, lowered, reduced).

It saves the full responses and a summary under `runs/` (`runs/e2e-latest-summary.json`).

## Layout

- `scripts/analyze.py`: synthetic-program analysis → `public/data/{roster,cohort}.json`, `public/data/survivors/*.json`
- `scripts/realworld.py`: CGMacros + NHANES check → `public/data/realworld.json`
- `scripts/ml_effects.py`: **the ML engine**: double-ML effects, personal effects, dinner forecast → `public/data/ml_effects.json` (run via `scripts/build-ml.sh` on Proxmox)
- `scripts/ml_compare.py`: statistics vs machine learning (prediction, planted effects, personal effects, double ML) → `public/data/mlcompare.json`; see `ML_COMPARISON.md`
- `scripts/pycaret_compare.py`: pycaret leaderboard of every regression model on the same features → `public/data/pycaret.json`
- `src/index.ts`: Worker (routes, CORS, Workers AI rewrite with numeric guard)
- `public/index.html`: standalone demo page (dependency-free SVG charts)

## Sources

- Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. *CGMacros* (version 1.0.0). PhysioNet, 2025. https://physionet.org/content/cgmacros/1.0.0/
- CDC, National Health and Nutrition Examination Survey 2011-2014 with 2019 linked mortality files (public domain). Survey weights and design applied (Rao-Wu bootstrap of PSUs within strata).
- Synthetic program data: Crane Lab, Sylvester Comprehensive Cancer Center, for this hackathon. Nobody in it is real.
