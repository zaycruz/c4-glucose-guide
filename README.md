# C4 Glucose Guide

**In 44 real adults, a walk after a meal went with a sugar rise only 3.2 mg/dL smaller (95% CI −6.9 to +0.5): not distinguishable from zero. In the synthetic program, a walk after dinner shows −18.3 mg/dL.**
So the advice offers the walk as a low-risk habit, not a proven fix. For the care team, the API flags who the care team should contact and how soon: today for level 2 lows (below 54), this week for other lows, a very high average or sustained highs, each with the reason.

| Result | Compared with | 95% CI |
|---|---|---|
| Real adults (CGMacros, 44 people), walk after a meal: **−3.2 mg/dL** rise | the same person's meals without a walk | −6.9 to 0.5 (not distinguishable from zero) |
| Synthetic program, walk after dinner: **−18.3 mg/dL** rise | the same person's dinners without a walk | 16.6 to 20.1 lower |
| Synthetic program, dinner at 8 pm or later: **+9.2 mg/dL** rise | the same person's earlier dinners | 7.1 to 11.3 |

The real-data row is the primary result. Program rows come from a synthetic 12-week cohort (500 people, double ML within person); the real interval excludes the program's all-meal −17.8, so expect a much smaller effect in real people. The advice leads with the real number.

**Reproduce in 3 commands** (needs [uv](https://docs.astral.sh/uv/) and the Challenge 4 data, see [data/README.md](data/README.md)):

```sh
git clone https://github.com/zaycruz/c4-glucose-guide && cd c4-glucose-guide
D=/path/to/Challenge4_Dose_and_Timing        # checksums in data/SHA256SUMS
scripts/reproduce.sh $D                      # about 10 min on 4 vCPU; ends by matching all 52 README numbers
```

Live API: https://c4-glucose-guide.zay-259.workers.dev/api/c4/survivors · contract: [CONTRACT.md](CONTRACT.md) · proof: [REPRODUCE.md](REPRODUCE.md), [`/api/c4/verification`](https://c4-glucose-guide.zay-259.workers.dev/api/c4/verification)

---

## Details

Coach Console module C4 (eMerge hackathon, Healthcare track, Sylvester): **how meals, meal timing and movement shape blood sugar**, per survivor and across the cohort, with honest uncertainty, turned into advice a survivor can follow. Every number in the advice traces to an `evidence[]` item.

- Live: https://c4-glucose-guide.zay-259.workers.dev (landing page in the Patient 360 design; the interactive demo is at `/demo.html`, cohort view `/demo.html#cohort`)
- API: see [CONTRACT.md](CONTRACT.md) (`/api/c4/survivors`, `/api/c4/survivors/:id`, `/api/c4/survivors/:id/advice`, `/api/c4/cohort`, `/api/c4/verification`)
- Reproduce every number below: [REPRODUCE.md](REPRODUCE.md) (`scripts/reproduce.sh`)
- Data: the Challenge 4 "Dose and Timing" set (the README inside calls it "Challenge 2"). Synthetic program for the survivors; CGMacros and NHANES as the real-data check.

## What it finds

Synthetic 12-week program: 500 sensor wearers, 18,218 clean meal curves. **The advice runs on machine learning:** double ML with gradient boosting, within person, with person-clustered 95% CIs (`scripts/ml_effects.py`). The classic fixed-effects regression stays as a cross-check, and the two agree. Randomized-arm contrasts and NHANES stay statistical, because randomization and survey weights need no model.

| Question | Answer |
|---|---|
| Is an evening walk worth it? | Yes. After dinner the rise is **18.3 mg/dL lower** (16.6 to 20.1), more than after other meals (14.6). All meals: 17.8 lower (regression cross-check: 17.6). Stronger in type 2 diabetes (23.3) than in normal blood sugar (13.2). Walking was self-chosen, so this is an association. It passes a negative control (pre-meal sugar −0.35, −1.38 to 0.68), and its E-value is 3.9. |
| How much is enough? | **10-19 min gets most of it** (17.5 lower); 20-29 min 19.5; 30+ min 16.6. Benefit levels off after 10 minutes. |
| Does a late dinner matter? | Dinner at 8 pm or later: **+9.2 mg/dL** (7.1 to 11.3). Each hour later adds more vs before 6 pm: 6-7 pm +3.9, 7-8 pm +8.4, 8-9 pm +13.3, 9-10 pm +18.2, 10 pm or later +25.4. |
| Overnight fast? | Randomized: adding the fast to coaching changed HbA1c by -0.03 points overall (-0.06 to 0.00). Type 2 diabetes looked larger (-0.08), but a formal test can't tell the blood sugar groups apart (Q p = 0.15; Holm-adjusted p = 0.13), so the advice uses the overall number. The fast arm did fast longer: 13.05 vs 12.07 h, with 62% vs 36% of nights at 13 h or more. Night shift (n=28) points the other way, but the CI is wide. Coaching itself vs email tips: -0.17. |
| Does it hold in real people? | CGMacros (45 real adults; 44 with walk data, 41 with late dinners), same ML engine: late dinner +2.6 (-5.1 to 10.3) and walk after a meal -3.2 (-6.9 to 0.5). Both point the same way but are not proven. The real walk interval excludes the program's -17.8, so expect a much smaller walk effect in real people. The program's late-dinner +9.2 is still inside the real interval. NHANES survivors, survey-weighted: neither fasting length nor daily movement shows a clear link with HbA1c. The API serves this weighted fit as `realWorld.nhanesSurvivors`; the unweighted fit is kept only as a labelled sensitivity (`nhanesSurvivorsUnweighted`). |
| What does ML add for a person? | A **forecast of the rise after their typical dinner** (gradient boosting; each meal forecast by a model that never saw it). Typical miss: 14.0 mg/dL; R² 0.62 on later meals from people it knows, 0.44 for people it has never seen, and 0.30 on real CGMacros adults. Expect the lower figures for a new patient. |
| Why ML, and not the regression? | We benchmarked them head to head ([ML_COMPARISON.md](ML_COMPARISON.md)). With confounding shaped like a threshold, a planted -10 effect is recovered with an error of +0.3 by double ML and +3.5 by the regression, whose interval never covered the truth. ML also predicts the rise better for new people (R² 0.44 vs 0.35) and in real data (0.30 vs 0.11). On this program's data the two engines agree, so the switch changes the method, not the story. |

Every recommendation carries a real-data check label (`agrees`, `same direction, not proven`, `no clear signal`, `disagrees`, `not tested`). [ROBUSTNESS.md](ROBUSTNESS.md) lists each check, how it could fail, and what it found.

Status (ok / watch / alert) uses fixed CGM consensus thresholds (time in range, highs above 250, lows below 70). Personal effects come from the same ML engine: each person's effect is fitted on their own cross-fitted residuals, their meals are bootstrapped, and the result is blended with the cohort (empirical Bayes). They are reported only where the person's own data supports them. If they don't have enough data, the advice says so and uses the cohort number. Night-shift workers get no clock-time advice. People with frequent lows are told not to stretch their fasts.

## Safety and patient-facing fields (live in the API)

- **Care team, with urgency:** `careTeamFlag.urgency` (and the roster's `careTeamUrgency`) is `"today"` for more than 1% of readings below 54 mg/dL (level 2 lows, a same-day safety issue) and `"this-week"` for more than 4% below 70 mg/dL, an average above 250 mg/dL, or more than 5% of readings above 250 (sustained highs). The card, the rule text and the Spanish and Kreyòl lines say "today" or "this week" to match. Other alerts get `level: "next visit"`. `careTeamToday` keeps its old meaning (flagged, any urgency) so existing clients still work; tier by `urgency`. The reason cites the evidence, for example "Glucose: Frequent lows: 5.3% of readings below 70 (target under 4%)". Ten of the 50 survivors are flagged for the care team (all this week; none has level 2 lows, so none is today) and one for the next visit. The E2E check fails if this sentence, the live roster and the published `/verification` disagree.
- **Alert load:** 21 of 50 survivors are alert or watch, but only the 10 flagged survivors go on the care team's list, all for this week; none needs a same-day call. The one "next visit" alert and the 10 watch survivors wait for the routine review; `todayReason` and `priority` let the Coach Console rank C4 inside one Today queue.
- **Lows mean care team first:** anyone with lows (above 4% below 70, or above 1% below 54) gets one card only, "tell your care team" (today for level 2 lows, otherwise this week), with no walking, fasting or meal-timing advice and no behaviour tags. `careTeamFlag.medicationCheck` says medicines are not recorded in this data and asks about insulin or a sulfonylurea (glipizide, glyburide, glimepiride), the usual causes of lows. They also get `hypoglycaemiaWarning` (the 15-15 rule, and when to call 911).
- **Medicines and walking:** because medicines are not recorded, every walk card ends "If you take insulin or a sulfonylurea (glipizide, glyburide, glimepiride), ask your care team about low sugar before adding walks after meals." and carries `medicationCaution` for the clinician: "Medicines are not recorded in this data. If the survivor takes insulin or a sulfonylurea (glipizide, glyburide, glimepiride), ask their care team about lows before adding post-meal walks. The coach does not change medication."
- **Plain language:** `plainSummary` per survivor. The cohort's `plainLanguage` leads with the real figure: "Walk after a meal: about -3 mg/dL in 44 real adults, not distinguishable from zero; -18 after dinner in the synthetic program."
- **Spanish and Haitian Creole (Kreyòl):** `patientView.{en,es,ht}` holds up to three actions, the care-team line, the low-sugar warning, and a help line for food, money or transport (211, or a social worker). It uses fixed rule numbers only. The Spanish and Kreyòl text is machine-translated and marked `reviewed: false` until a native speaker checks it.
- **One AI model, no fallback model:** `/advice` requests `@cf/meta/llama-3.3-70b-instruct-fp8-fast` only (`fallbackModel: null`). Workers AI labels its responses `@cf/meta/llama-3.3-70b-instruct-sd`; the API passes that through as `providerModelName` and says it is the provider's name for the same deployment. If the AI call fails, the checked template text is shown (`source: "template"`).
- **AI advice is precomputed:** `/advice` is served from a cache keyed to the current recommendations (about 0.1 s instead of 3-26 s). `?fresh=1` calls the model live.
- **Fairness by group** (`/api/c4/cohort` → `fairness`): by race/ethnicity, sex, age, preferred language and cancer type. For each group it gives the forecast error and the walk and late-dinner effects with CIs; groups under 20 people get no effect estimate. Cochran Q tests whether the effects differ, and Holm corrects for running 10 tests (5 attributes × 2 effects). It comes from the same engine run, reproduced from a fresh clone on Cloudflare with zero differences.
  - **In plain words:** after correcting for 10 tests, only the walk effect by sex differs (Holm p = 0.02). The forecast misses by a similar amount in every group: 11.9 to 16.2 mg/dL.
  - Race and ethnicity: the effects can't be told apart (walk Q p = 0.12; late dinner Q p = 0.37; Holm p = 0.59 and 1.0).
  - Sex: the walk effect is larger in men (−20.3 vs −15.4; Holm p = 0.02). The late-dinner difference does not survive Holm (p = 0.08).
  - Age: the walk effect looks smaller at 70+ (−14.7 vs −19.4 under 60), but not after Holm (p = 0.07).
  - Kreyòl speakers (24 people): the late-dinner effect looks smaller (+1.9; Q p = 0.03), but not after Holm (p = 0.20).
  - Monitoring before real use (`fairness.monitoring`): recompute the table on real patients every quarter; where a group differs after Holm, use the pooled cohort number for that group and route its alerts to care-team review instead of automatic advice.
  - The data is synthetic, so this checks that the engine treats groups alike. It does not measure real-world fairness.

## Demo story (2 minutes)

1. **Cohort first** (`/demo.html#cohort`, 40 s). "Does a late dinner matter? Same person, same carbs: every hour later adds to the rise, up to +25 mg/dL after 10 pm. Is a walk after a meal worth it? In 44 real adults it went with a rise only 3.2 mg/dL smaller, and we can't tell that from zero. In the synthetic program it is 18 mg/dL after dinner, and 10 minutes gets most of it. These numbers come from machine learning (double ML), checked against a classic regression. Overnight fast: the randomized arm changed HbA1c by -0.03, and we could not show it differs by blood sugar group."
2. **Roster** (`/demo.html`, 20 s). Each tile shows a plain headline and up to four signal tags (red = risk, amber = watch, green = working), so you can scan without opening anyone. Filter chips narrow it to Alert / Watch / Ok. Click **N0632** (alert, 76F, colorectal).
3. **N0632** (30 s). Headline: "Frequent lows: 5.3% of readings below 70". One tag, Lows 5.3%, and one card: tell the care team this week, ask whether insulin or a sulfonylurea needs changing (medicines are not in this data), and don't change exercise or meal times until then. No walk or fasting advice for anyone with lows.
3b. **N0013** (20 s). A survivor without lows: the walk card opens with the real-data figure (about 3.2 mg/dL, not distinguishable from zero), then the synthetic program's. "Show the numbers" opens the evidence behind each number.
4. **Rephrase with Workers AI** (10 s). The text becomes warmer. The note shows how many rewrites were rejected, either for a number that isn't in the evidence or for causal words like "helped".

## Future work (from the judges' panel; not built)

- **Red flags beyond glucose:** a model-free route to a human for chest pain, neurological symptoms, fever on chemo or suicidal thoughts (911 / 988), not only glucose rules.
- **Medicines:** this data records none, so the lows rule asks about insulin or a sulfonylurea. With a medication list, anyone on those drugs should get the lows protection before lows appear.
- **Who the real data covers:** describe the 45 CGMacros adults by race, ethnicity, language and insurance (or state they are unknown), and compare the synthetic cohort's make-up with Miami-Dade, including Haitian and Caribbean patients.
- **Review the Spanish and Kreyòl text** with native speakers, then set `reviewed: true` in the API.
- **Forecast:** a calibration check and a clinic-baseline comparison with a CI. **Serving:** p95 latency and cost per AI call, and a model-file hash proving served = evaluated.
- **Survivor-facing numbers:** show only the real-data figure to survivors; keep the synthetic −18.3 / +9.2 in the clinician view.
- **Same-day path, tested:** no survivor in this cohort has level 2 lows, so the "today" route is enforced by the E2E rule but never exercised by a real case. Add a fixture with readings below 54, and consider a same-day route for frequent lows in older adults.
- **AI path in `/verification`:** measured p95 latency, cost and fallback rate over 30+ calls, and an alert if Workers AI ever reports a model name other than the usual `providerModelName`.
- **Plainer survivor text:** keep CIs in the clinician view and give survivors one short sentence per action.

## Run

See [REPRODUCE.md](REPRODUCE.md) for the one-command path from the raw data to every number in this README (`scripts/reproduce.sh <data dir>`). The steps it wraps:

```sh
npm install
D=~/projects/emerge-hackathon/c4-data/Challenge4_Dose_and_Timing   # dataset extracted outside the repo
# 0. check every README number from scratch (about 10 min on 4 vCPU; writes repro_out/, not public/data):
scripts/reproduce.sh $D repro_out
# 1. ML engine + statistics-vs-ML benchmark on a remote Linux host over ssh (a few minutes of CPU):
#    writes public/data/ml_effects.json and public/data/mlcompare.json. CLEAN=1 removes the remote copy afterwards.
scripts/build-ml.sh <ssh host> $D
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
- safety: everyone with lows (above 4% below 70 or above 1% below 54), an average above 250 or sustained highs is flagged for the care team with a reason; the lows group gets only the care-team card, the medication check, the warning and no behaviour tags; the README's flagged count equals the live roster and the published `/verification`; and `todayReason` is at most 90 characters with traced numbers;
- `patientView` has all three languages, the same action ids in each, fixed rule numbers only, and the 211 help line; `plainSummary` and the cohort `plainLanguage` numbers trace, and the "fifth / quarter" wording matches the actual ratio;
- advice comes from the cache, is never a raw model error, and with `--strict` answers in under 2 s; the published `/verification` record matches the live data.

It saves the full responses and a summary under `runs/` (`runs/e2e-latest-summary.json`).

## Layout

- `scripts/analyze.py`: synthetic-program analysis → `public/data/{roster,cohort}.json`, `public/data/survivors/*.json`
- `scripts/realworld.py`: CGMacros + NHANES check → `public/data/realworld.json`
- `scripts/ml_effects.py`: **the ML engine**: double-ML effects, personal effects, dinner forecast → `public/data/ml_effects.json` (run by `scripts/reproduce.sh`, or `scripts/build-ml.sh` on a remote host)
- `scripts/ml_compare.py`: statistics vs machine learning (prediction, planted effects, personal effects, double ML) → `public/data/mlcompare.json`; see `ML_COMPARISON.md`
- `scripts/pycaret_compare.py`: pycaret leaderboard of every regression model on the same features → `public/data/pycaret.json`
- `scripts/reproduce.sh`, `scripts/headline.py`: one-command reproduction and the README-number diff (see REPRODUCE.md); `infra/cf-runner/`: runs it from a fresh clone in a Cloudflare Sandbox
- `pyproject.toml`, `uv.lock`: pinned Python 3.12 environment; `data/README.md`: where the data goes and its SHA-256 sums
- `src/index.ts`: Worker (routes, CORS, Workers AI rewrite with numeric guard)
- `public/index.html`, `public/p360.css`: landing page in the shared Patient 360 design; `public/demo.html`: the interactive demo (dependency-free SVG charts)

## Sources

- Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. *CGMacros* (version 1.0.0). PhysioNet, 2025. https://physionet.org/content/cgmacros/1.0.0/
- CDC, National Health and Nutrition Examination Survey 2011-2014 with 2019 linked mortality files (public domain). Survey weights and design applied (Rao-Wu bootstrap of PSUs within strata).
- Synthetic program data: Crane Lab, Sylvester Comprehensive Cancer Center, for this hackathon. Nobody in it is real.
