# Robustness checks

How sure are we about each headline finding? This file lists the checks, how they can fail, and what they found.

## How each check could fail (written before the code)

**Negative control: a walk cannot change the reading before the meal**
- Using a reading that is already after the meal (the meal time is floored to 15 min, so offset 0 can be up to 14 min after the bite). Fix: use the reading 15 min before.
- Walk meals have more carbs or a different meal type, and those predict pre-meal glucose. Adjust for both, as in the main model.
- Passing by noise: a wide CI that includes 0 is not proof of no confounding. Report the CI next to the main effect so the size can be compared.

**Instrument: the randomized arm changes how often people walk**
- Weak first stage. If the arm barely moves walking, the ratio blows up. Report the first-stage difference and its CI; flag the result if the CI includes 0.
- Exclusion restriction. Coaching also changes diet and meal timing, so the arm can move the rise without walking. Mitigations: adjust the rise for carbs and meal type before averaging; use coaching vs email tips only, leaving out the fast arm, which also moves dinner time. Still a check, not a replacement.
- Person-level analysis: walk share and mean rise per person, so bootstrap over people, not meals.
- People without walk-eligible meals or with very few meals: require at least 10 scored meals.

**E-value**
- Converting a mean difference to a risk ratio needs a standard deviation. Use the within-person residual SD of the rise, not the raw SD, since the estimate is within-person. Say which.
- Report it for the estimate and for the CI bound nearer to null.

**CGMacros walk definition**
- Fitbit METs are missing for some minutes. Require at least 50 of the 60 minutes.
- Too few walk meals to estimate anything. Report the count; give no estimate under 20 walk meals.
- Threshold choice (3 METs, 10 min) is a researcher choice. State it.

**NHANES survey weights**
- Wrong weight: HbA1c and movement come from the exam (MEC) weight; eating times come from the 2-day diet weight. Use the matching one per model.
- Domain analysis: subsetting to survivors before resampling breaks the design. Resample PSUs within strata on the full file, then subset.
- Strata with a single PSU break the Rao-Wu bootstrap. Check counts; collapse if needed.
- Missing weights (0 or NaN) for people outside a subsample. Drop those rows for that model only.

**Power**
- Treating the observed real-data effect as the truth. Report the people needed for both the program effect and the real-data effect.
- Scaling SE by sqrt(people) assumes new people look like these 45. State it.

**Subgroup heterogeneity**
- Picking the best subgroup after seeing results. Test the arm x group interaction (Cochran Q across the three blood-sugar groups) and Holm-adjust across all subgroups looked at.
- If the interaction does not hold, the advice must stop using group-specific numbers.

**Advice tracking real data**
- Calling "same direction, CI includes 0" agreement. Use four labels: agrees (same sign, CI excludes 0), same direction not proven, no clear signal, disagrees (opposite sign, CI excludes 0).

## Results (rerun 2026-10-17 on the ML engine, double ML; numbers from `public/data/cohort.json` → `robustness`, `trial.heterogeneity`, `realWorld`)

| Weak spot | Check | Result | Verdict |
|---|---|---|---|
| Walk effect is observational | Negative control: pre-meal reading 15 min before walk meals vs other meals, same person | -0.35 mg/dL (-1.38 to 0.68), against a walk effect of -17.8 (double ML for both) | Passes: walk meals don't start from a different place |
| | E-value | 3.9 (CI bound 3.6) | A hidden factor would need a risk ratio of about 3.9 with both walking and the rise to explain the effect away |
| | Randomized arm as instrument (coaching vs email tips) | First stage +3.7 pts of meals walked (2.5 to 5.0); walk effect -31 (-122 to 59). Dinner-only version: -61 to -98 per walked meal, which is impossible | **Inconclusive.** Coaching moves the rise through more than walking, so the exclusion restriction fails. Not counted as support |
| | Real data with the program's walk definition (CGMacros, ≥10 min at 3+ METs within an hour) | Double ML, same engine as the program: -3.2 mg/dL (-6.9 to 0.5), 1,329 meals, 44 people (fixed effects: -4.4, -11.4 to 1.4) | Same direction, not proven. **The interval stops short of the program's -17.8, so the real effect is probably much smaller** |
| NHANES unweighted | Survey weights (exam weight for HbA1c and movement, 2-day diet weight for eating times), Rao-Wu bootstrap of PSUs within strata, domain estimate | Fast: +0.013 per hour (-0.025 to 0.048). Movement: -0.021 per 1,000 MIMS (-0.052 to 0.006) | Fast: no link, as before. **Movement loses its clear signal once weighted** |
| Synthetic headline numbers | Power: people needed for 80% power, scaled from CGMacros | Late dinner: 59 people for the program size (+9.2), about 730 for the real size (+2.6). Walk: 4 for the program size, 120 for the real size (-3.2) | Walk: CGMacros could easily have seen the program's size, and its interval excludes it, so the program's walk number is too big for real people. Late dinner: the program's +9.2 is still inside the real interval (-5.1 to 10.3); confirming an effect as small as the real estimate would take hundreds of people |
| | Real-data label on every recommendation | walk and late dinner: same direction, not proven; overnight fast: no clear signal (NHANES); safety rules: not tested | Shown in the advice text, on a badge in the UI, and in `recommendations[].realDataCheck` |
| Subgroup claim (new) | Cochran Q across blood-sugar groups for the fast effect on HbA1c; Holm across 10 subgroup contrasts | Q p = 0.15; type 2 diabetes p = 0.13 after Holm | **Does not hold.** Advice now uses the overall fast number (-0.03) for everyone and says the groups could not be told apart (For fasting glucose the groups do differ: Q p = 0.02. The advice is about HbA1c.) |

Two headline claims changed:
- **"The fast gain is concentrated in type 2 diabetes."** Retracted: the groups could not be told apart.
- **"More movement goes with lower HbA1c in NHANES."** Retracted: the link disappears with survey weights.

Two claims got firmer caveats. The walk effect points the same way in real people, but it is clearly smaller there than in the program. The late-dinner effect also points the same way, and the program's size is still possible.

## Engine switch to machine learning (2026-10-17)

zay, after the statistics-vs-ML benchmark: "lets use ML ... update our submission to use ML instead". Every meal-level effect now comes from double ML (`scripts/ml_effects.py`). The checks above were rerun on it: the negative control, the E-value, the real-data checks (CGMacros on the same engine) and the power numbers. The randomized-arm contrasts, the subgroup test and the survey-weighted NHANES estimates are unchanged, because they need no model. The fixed-effects regression is kept as a cross-check evidence item (`cohort_walk_fe`, `cohort_late_fe`). See ML_COMPARISON.md.

## Judges'-panel API fixes: how they could fail (written before the code, 2026-10-17)

The fixes are safety routing, cached advice, plain language, Spanish and Kreyòl copy, and verification online. Each failure below has an E2E assertion in `scripts/e2e.mjs`. The per-group fairness table from `scripts/ml_effects.py` is now served as `cohort.fairness` (from the engine rerun on Cloudflare, which reproduced every other number exactly), and item 8 applies.

1. Someone with frequent lows (above 4% of readings below 70) has no `careTeamFlag.today`, no `hypoglycaemiaWarning`, or a walk recommendation without the low-sugar `caution`. Someone in that group still gets "aim for a 13-hour fast".
2. An average above 250 mg/dL is not flagged for the care team today.
3. Someone is flagged with no reason, or not flagged but given a reason. The roster's flag differs from the detail's.
4. `todayReason` is longer than 90 characters, or one of its numbers is not in the person's evidence.
5. A number in `plainSummary` or in the cohort `plainLanguage` does not trace to evidence. The "real data is about a fifth / a quarter" wording disagrees with the actual ratio.
6. `patientView`:
   - It is missing a language (en, es, ht), or the action ids differ between languages.
   - It holds a measured number (only fixed rule numbers are allowed).
   - A flagged person has no care-team line in some language, or the help route (211) is missing.
7. Cached advice:
   - It is stale (rebuilt advice text with old cached wording).
   - It is not served from the cache, so stage latency stays slow.
   - It drifts from the evidence (the same trace check as live AI).
8. The fairness table reports groups too small to read (under 20 people) without saying so, or its numbers do not match the engine output.
9. The `/verification` endpoint serves an old E2E result, or one that does not match the deployed data.
