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
  weights are NOT applied (demo), so this describes the sample, not all US survivors.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analyze import fe_effect, r  # noqa: E402

RNG = np.random.default_rng(7)


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
        mets = win.loc[: m.timestamp + pd.Timedelta(minutes=60)].fitbit_mets_x10.mean() / 10
        rows.append({"participant_id": m.participant_id, "meal_type": m.meal_type, "carbs_g": m.meal_carbs_g,
                     "rise": win.glucose_libre_mg_dl.max() - base, "mets_1h": mets,
                     "hour": m.timestamp.hour + m.timestamp.minute / 60})
    df = pd.DataFrame(rows).dropna()
    df["active"] = (df.mets_1h >= 1.5).astype(int)  # mean >= 1.5 METs over the hour = some walking
    df["late"] = ((df.hour >= 20) | (df.hour < 4)).astype(int)
    din = df[df.meal_type == "dinner"]
    out = {
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

    res = clean({"cgmacros": cgmacros(d), "nhanesSurvivors": nhanes(d), "methods": __doc__.strip()})
    with open(f"{out_dir}/realworld.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "methods"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
