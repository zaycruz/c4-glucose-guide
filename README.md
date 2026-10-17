# C4 Glucose Guide

Coach Console module C4 (eMerge hackathon, Healthcare track, Sylvester): **how meals, meal timing and movement shape blood sugar**, per survivor and across the cohort, with honest uncertainty, turned into advice a survivor can follow. Every number in the advice traces to an `evidence[]` item.

- Live: https://c4-glucose-guide.zay-259.workers.dev (cohort view: `/#cohort`)
- API: see [CONTRACT.md](CONTRACT.md) (`/api/c4/survivors`, `/api/c4/survivors/:id`, `/api/c4/survivors/:id/advice`, `/api/c4/cohort`)
- Data: the Challenge 4 "Dose and Timing" set (the README inside calls it "Challenge 2"). Synthetic program for the survivors; CGMacros and NHANES as the real-data check.

## What it finds

Synthetic 12-week program: 500 sensor wearers, 18,218 clean meal curves. Within-person estimates, cluster-bootstrap 95% CIs:

| Question | Answer |
|---|---|
| Is an evening walk worth it? | Yes. After dinner the rise is **18.7 mg/dL lower** (17.2 to 20.4), more than after other meals (13.5). All meals: 17.6 lower. Stronger in type 2 diabetes (25.0) than normal blood sugar (13.4). This is an association, because walking was self-chosen. |
| How much is enough? | **10-19 min gets most of it** (17.1 lower); 20-29 min 20.1; 30+ min 16.8. Benefit levels off after 10 minutes. |
| Does a late dinner matter? | Dinner at 8 pm or later: **+9.1 mg/dL** (7.1 to 11.2). Each hour later adds more: 6-7 pm +6.3 … 10 pm+ +28.8 vs before 6 pm. |
| Overnight fast? | Randomized: adding the fast to coaching changed HbA1c by -0.03 points overall (-0.06 to -0.01). **The gain is concentrated in type 2 diabetes: -0.08 (-0.15 to -0.01)**, with fasting glucose -3.4 mg/dL; it is near zero for normal blood sugar. The fast arm did fast longer: 13.05 vs 12.07 h, with 62% vs 36% of nights at 13 h or more. Night shift (n=28) points the other way, but the CI is wide. Coaching itself vs email tips: -0.17. |
| Does it hold in real people? | CGMacros (45 real adults): late dinner +4.6 (-1.5 to 11.4), same direction but not proven; no clear post-meal movement effect. NHANES survivors: more daily movement goes with slightly lower HbA1c (-0.023 per 1,000 MIMS); fasting length shows no link. |

Advice uses the survivor's own blood sugar group for the fast estimate.

Status (ok / watch / alert) uses fixed CGM consensus thresholds (time in range, highs above 250, lows below 70). Personal effects are reported only where the person's own data supports them: their meals are bootstrapped, then blended with the cohort (empirical Bayes). If they don't have enough data, the advice says so and uses the cohort number. Night-shift workers get no clock-time advice. People with frequent lows are told not to stretch their fasts.

## Demo story (2 minutes)

1. **Cohort first** (`/#cohort`, 40 s). "Does a late dinner matter? Same person, same carbs: every hour later adds to the rise, up to +29 mg/dL after 10 pm. Is an evening walk worth it? Yes: about 19 mg/dL lower after dinner, more than after other meals, and 10 minutes gets most of it. Overnight fast: the randomized arm helps most in type 2 diabetes (HbA1c -0.08). And here is the honest part: in 45 real people the late-dinner effect points the same way, but it isn't proven yet."
2. **Roster** (`/`, 20 s). Each tile shows a plain headline and up to four signal tags (red = risk, amber = watch, green = working), so you can scan without opening anyone. Filter chips narrow it to Alert / Watch / Ok. Click **N0632** (alert, 76F, colorectal).
3. **N0632** (50 s). Headline: "Frequent lows: 5.3% of readings below 70". The tags read Lows 5.3%, Late dinners 57%, Walks work: -18.2 mg/dL, Fast 9.9 h. The KPI strip and the typical-day chart sit under the header. Each piece of advice comes with its own chart: tell the care team about the lows and don't stretch the fast (check-ups table); walk after dinner (her own walk vs no-walk curve, a personal estimate with a CI); eat dinner earlier (her early vs late dinner curves next to the cohort's). "Show the numbers" opens the evidence behind each number.
4. **Rephrase with Workers AI** (10 s). The text becomes warmer. The note shows how many rewrites were rejected, either for a number that isn't in the evidence or for causal words like "helped".

## Run

```sh
npm install
# rebuild the data the Worker serves (offline, ~10 s). Dataset extracted outside the repo:
scripts/build-data.sh ~/projects/emerge-hackathon/c4-data/Challenge4_Dose_and_Timing
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
- `src/index.ts`: Worker (routes, CORS, Workers AI rewrite with numeric guard)
- `public/index.html`: standalone demo page (dependency-free SVG charts)

## Sources

- Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. *CGMacros* (version 1.0.0). PhysioNet, 2025. https://physionet.org/content/cgmacros/1.0.0/
- CDC, National Health and Nutrition Examination Survey 2011-2014 with 2019 linked mortality files (public domain). Survey weights not applied.
- Synthetic program data: Crane Lab, Sylvester Comprehensive Cancer Center, for this hackathon. Nobody in it is real.
