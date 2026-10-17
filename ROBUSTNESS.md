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

## Results (run 2026-10-17; numbers from `public/data/cohort.json` → `robustness`, `trial.heterogeneity`, `realWorld`)

| Weak spot | Check | Result | Verdict |
|---|---|---|---|
| Walk effect is observational | Negative control: pre-meal reading 15 min before walk meals vs other meals, same person | -0.4 mg/dL (-1.4 to 0.7), against a walk effect of -17.6 | Passes: walk meals don't start from a different place |
| | E-value | 3.6 (CI bound 3.4) | A hidden factor would need a risk ratio of about 3.6 with both walking and the rise to explain the effect away |
| | Randomized arm as instrument (coaching vs email tips) | First stage +3.7 pts of meals walked (2.5 to 5.0); walk effect -31 (-122 to 59). Dinner-only version: -61 to -98 per walked meal, which is impossible | **Inconclusive.** Coaching moves the rise through more than walking, so the exclusion restriction fails. Not counted as support |
| | Real data with the program's walk definition (CGMacros, ≥10 min at 3+ METs within an hour) | -4.4 mg/dL (-11.4 to 1.4), 192 walk meals, 44 people | Same direction, not proven. **The interval stops short of the program's -17.6, so the real effect is probably much smaller** |
| NHANES unweighted | Survey weights (exam weight for HbA1c and movement, 2-day diet weight for eating times), Rao-Wu bootstrap of PSUs within strata, domain estimate | Fast: +0.013 per hour (-0.025 to 0.048). Movement: -0.021 per 1,000 MIMS (-0.052 to 0.006) | Fast: no link, as before. **Movement loses its clear signal once weighted** |
| Synthetic headline numbers | Power: people needed for 80% power, scaled from CGMacros | Late dinner: 46 for the program size (+9.1), 178 for the real size (+4.6). Walk: 12 for the program size, 192 for the real size | Walk: CGMacros could easily have seen the program's size, and its interval excludes it, so the program's walk number is too big for real people. Late dinner: the program's +9.1 is still inside the real interval, and confirming the real-data size would take about 180 people |
| | Real-data label on every recommendation | walk and late dinner: same direction, not proven; overnight fast: no clear signal (NHANES); safety rules: not tested | Shown in the advice text, on a badge in the UI, and in `recommendations[].realDataCheck` |
| Subgroup claim (new) | Cochran Q across blood-sugar groups for the fast effect on HbA1c; Holm across 10 subgroup contrasts | Q p = 0.15; type 2 diabetes p = 0.13 after Holm | **Does not hold.** Advice now uses the overall fast number (-0.03) for everyone and says the groups could not be told apart (For fasting glucose the groups do differ: Q p = 0.02. The advice is about HbA1c.) |

Two headline claims changed:
- **"The fast gain is concentrated in type 2 diabetes."** Retracted: the groups could not be told apart.
- **"More movement goes with lower HbA1c in NHANES."** Retracted: the link disappears with survey weights.

Two claims got firmer caveats. The walk effect points the same way in real people, but it is clearly smaller there than in the program. The late-dinner effect also points the same way, and the program's size is still possible.
