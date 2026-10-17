"""Real-data cross-check for C4: do the synthetic patterns hold in real people?

Usage: uv run --with pandas --with numpy python -I scripts/realworld.py <dataset_dir> <out_dir>
Writes <out_dir>/realworld.json.

- CGMacros (45 real adults, minute-level Libre sensor + Fitbit): within-person effect of
  movement in the hour after a meal and of a late dinner (>= 20:00) on the post-meal rise.
  Rise = max Libre glucose 0-120 min after the meal minus the reading at the meal minute.
  Meals with another logged meal in the next 120 min are dropped. CIs: cluster bootstrap
  over people.
- NHANES 2011-2014 cancer survivors (real, one snapshot each): association of overnight
  fast, last eating hour and daily movement with HbA1c, adjusted for age and sex. Survey
  weights are NOT applied in nhanesSurvivors (describes the sample). nhanesSurvivorsWeighted applies the
  survey weights and design (strata, PSUs) and describes US adult cancer survivors 2011-2014.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analyze import fe_effect, r  # noqa: E402

B_SURVEY = 500

RNG = np.random.default_rng(7)


WALK_METS = 3.0  # brisk-walk intensity
WALK_MIN = 10    # sustained minutes, matching the program's shortest walk bin


def longest_run(mask):
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def cgmacros(d):
    mm = pd.read_csv(f"{d}/CGMacros/cgmacros_minute_by_minute.csv", parse_dates=["timestamp"],
                     usecols=["participant_id", "timestamp", "glucose_libre_mg_dl", "fitbit_mets_x10"])
    meals = pd.read_csv(f"{d}/CGMacros/cgmacros_meals.csv", parse_dates=["timestamp"])
    meals["meal_type"] = meals.meal_type.str.lower().str.replace(r"snacks?( 1)?", "snack", regex=True)
    meals = meals.sort_values(["participant_id", "timestamp"])
    meals["next"] = meals.groupby("participant_id").timestamp.shift(-1)
    meals = meals[(meals.next.isna()) | ((meals.next - meals.timestamp) > pd.Timedelta(minutes=120))]
    meals = meals[meals.meal_carbs_g >= 10]
    rows = []
    by = {pid: g.set_index("timestamp") for pid, g in mm.groupby("participant_id")}
    for m in meals.itertuples():
        g = by.get(m.participant_id)
        if g is None:
            continue
        win = g.loc[m.timestamp: m.timestamp + pd.Timedelta(minutes=120)]
        if len(win) < 100 or pd.isna(win.glucose_libre_mg_dl.iloc[0]):
            continue
        base = win.glucose_libre_mg_dl.iloc[0]
        hour1 = win.loc[: m.timestamp + pd.Timedelta(minutes=60)].fitbit_mets_x10 / 10
        mets = hour1.mean()
        rows.append({"participant_id": m.participant_id, "meal_type": m.meal_type, "carbs_g": m.meal_carbs_g,
                     "rise": win.glucose_libre_mg_dl.max() - base, "mets_1h": mets,
                     "walk10": longest_run(hour1.to_numpy() >= WALK_METS) >= WALK_MIN if hour1.notna().sum() >= 50 else np.nan,
                     "hour": m.timestamp.hour + m.timestamp.minute / 60})
    df = pd.DataFrame(rows).dropna(subset=["rise", "mets_1h", "carbs_g"])
    df["active"] = (df.mets_1h >= 1.5).astype(int)  # mean >= 1.5 METs over the hour = some walking
    df["late"] = ((df.hour >= 20) | (df.hour < 4)).astype(int)
    din = df[df.meal_type == "dinner"]
    df["walk10"] = df.walk10.astype(float)
    wk = df[df.walk10.notna()]
    walk_meals = int(wk.walk10.sum())
    out = {
        "walk10MealCount": walk_meals, "walk10Meals": int(len(wk)),
        "walk10": fe_effect(wk, ["walk10", "carbs_g"]) if walk_meals >= 20 else None,
        "walkDefinition": f"at least {WALK_MIN} straight minutes at {WALK_METS:g}+ METs (Fitbit) in the hour after the meal; meals with 50+ minutes of Fitbit data",
        "source": "CGMacros v1.0.0 (PhysioNet 2025): 45 real adults, 10 days, Libre sensor + Fitbit",
        "meals": int(len(df)), "people": int(df.participant_id.nunique()),
        "activeHourShare": r(100 * df.active.mean()),
        "movementPerMet": fe_effect(df, ["mets_1h", "carbs_g"]),
        "activeHourVsNot": fe_effect(df, ["active", "carbs_g"]),
        "lateDinner": fe_effect(din, ["late", "carbs_g"]),
        "carbsPer10g": fe_effect(df, ["carbs_g"]),
        "lateDinnerCount": int(din.late.sum()), "dinnerCount": int(len(din)),
    }
    for k in ("est", "lo", "hi"):
        out["carbsPer10g"][k] *= 10
    return out


def ols_boot(df, y, x, covars, B=1000):
    d = df[[y, x] + covars].dropna()
    X = np.column_stack([np.ones(len(d)), d[[x] + covars].to_numpy(float)])
    Y = d[y].to_numpy(float)
    est = np.linalg.lstsq(X, Y, rcond=None)[0][1]
    boots = []
    for _ in range(B):
        i = RNG.integers(0, len(d), len(d))
        boots.append(np.linalg.lstsq(X[i], Y[i], rcond=None)[0][1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": est, "lo": lo, "hi": hi, "n": int(len(d))}


def wls(X, Y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], Y * sw, rcond=None)[0][1]


def design_boot_weights(n, weight_col, B=B_SURVEY):
    """Rao-Wu rescaling bootstrap: in each stratum draw (n_h - 1) PSUs with replacement from n_h, on the FULL file,
    so a later subset to survivors is a proper domain estimate. Returns a (B, rows) array of replicate weights."""
    base = n[weight_col].fillna(0).to_numpy(float)
    psu_key = n.survey_strata.astype(str) + "/" + n.survey_psu.astype(str)
    reps = np.empty((B, len(n)))
    strata = {h: sorted(g.unique()) for h, g in psu_key.groupby(n.survey_strata)}
    for b in range(B):
        mult = {}
        for h, psus in strata.items():
            k = len(psus)
            draws = RNG.choice(len(psus), k - 1, replace=True)
            counts = np.bincount(draws, minlength=k)
            for p_, c in zip(psus, counts):
                mult[p_] = c * k / (k - 1)
        reps[b] = base * psu_key.map(mult).to_numpy(float)
    return base, reps


def wls_design(n, mask, y, x, covars, weight_col, reps_cache):
    """Survey-weighted regression of y on x (+covars) within the domain `mask`; design-based bootstrap 95% CI."""
    if weight_col not in reps_cache:
        reps_cache[weight_col] = design_boot_weights(n, weight_col)
    base, reps = reps_cache[weight_col]
    ok = mask & n[[y, x] + covars].notna().all(axis=1).to_numpy() & (base > 0)
    d = n[ok]
    X = np.column_stack([np.ones(len(d)), d[[x] + covars].to_numpy(float)])
    Y = d[y].to_numpy(float)
    est = wls(X, Y, base[ok])
    boots = []
    for b in range(len(reps)):
        w = reps[b][ok]
        if (w > 0).sum() > X.shape[1] + 5:
            boots.append(wls(X[w > 0], Y[w > 0], w[w > 0]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": est, "lo": lo, "hi": hi, "n": int(ok.sum()), "weight": weight_col}


def nhanes_weighted(d):
    """Same models as nhanes(), with survey weights and the NHANES design (strata, PSUs)."""
    n = pd.read_csv(f"{d}/nhanes_us_adults.csv")
    n["male"] = (n.sex.str.lower() == "male").astype(float)
    n["mims_k"] = n.movement_per_day_mims / 1000
    surv = (n.cancer_survivor == "yes").to_numpy()
    nodia = surv & (n.diabetes_diagnosed == "no").to_numpy()
    cov = ["age", "male"]
    cache = {}
    return {
        "source": "NHANES 2011-2014 cancer survivors, survey-weighted (exam weight for HbA1c and movement, 2-day diet weight for eating times); "
                  "95% CI from a Rao-Wu bootstrap of PSUs within strata on the full file (domain estimate)",
        "hba1cPerFastHour": wls_design(n, surv, "hba1c_pct", "overnight_fast_hours_avg", cov, "survey_weight_diet_2day", cache),
        "hba1cPerLaterLastMealHour": wls_design(n, surv, "hba1c_pct", "last_eating_hour_avg", cov, "survey_weight_diet_2day", cache),
        "hba1cPer1000Mims": wls_design(n, surv, "hba1c_pct", "mims_k", cov, "survey_weight_exam", cache),
        "noDiabetes": {
            "hba1cPerFastHour": wls_design(n, nodia, "hba1c_pct", "overnight_fast_hours_avg", cov, "survey_weight_diet_2day", cache),
            "hba1cPer1000Mims": wls_design(n, nodia, "hba1c_pct", "mims_k", cov, "survey_weight_exam", cache),
        },
    }


def nhanes(d):
    n = pd.read_csv(f"{d}/nhanes_us_adults.csv")
    s = n[n.cancer_survivor == "yes"].copy()
    s["male"] = (s.sex.str.lower() == "male").astype(float)
    s["mims_k"] = s.movement_per_day_mims / 1000
    cov = ["age", "male"]
    nd = s[s.diabetes_diagnosed == "no"]
    out = {
        "source": "NHANES 2011-2014, 808 real cancer survivors; unweighted (survey weights not applied)",
        "survivors": int(len(s)),
        "medians": {"overnightFastHours": r(s.overnight_fast_hours_avg.median()), "lastEatingHour": r(s.last_eating_hour_avg.median()),
                    "movementMims": r(s.movement_per_day_mims.median(), 0), "hba1c": r(s.hba1c_pct.median(), 2)},
        "hba1cPerFastHour": ols_boot(s, "hba1c_pct", "overnight_fast_hours_avg", cov),
        "hba1cPerLaterLastMealHour": ols_boot(s, "hba1c_pct", "last_eating_hour_avg", cov),
        "hba1cPer1000Mims": ols_boot(s, "hba1c_pct", "mims_k", cov),
        "noDiabetes": {
            "hba1cPerFastHour": ols_boot(nd, "hba1c_pct", "overnight_fast_hours_avg", cov),
            "hba1cPer1000Mims": ols_boot(nd, "hba1c_pct", "mims_k", cov),
        },
    }
    return out


def main(d, out_dir):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (float, np.floating)):
            return r(o, 3)
        if isinstance(o, np.integer):
            return int(o)
        return o

    res = clean({"cgmacros": cgmacros(d), "nhanesSurvivors": nhanes(d), "nhanesSurvivorsWeighted": nhanes_weighted(d), "methods": __doc__.strip()})
    with open(f"{out_dir}/realworld.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "methods"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
