"""Machine-learning engine for C4 Glucose Guide: the effect sizes the advice states.

Writes <out>/ml_effects.json, which scripts/analyze.py requires:
  cohort     double ML (DML) effects: walk, walk after dinner / other meals, late dinner, carbs,
             iAUC variants, effects by blood sugar group, walk-length and dinner-hour curves,
             negative control (pre-meal sugar)
  personal   each sensor wearer's own walk and late-dinner effect from cross-fitted DML residuals
             (analyze.py blends these with the cohort effect, empirical Bayes)
  forecast   gradient-boosting forecast of the rise after a meal, out-of-fold for every meal
  real       DML on CGMacros (45 real adults): walk and late dinner

DML (partially linear model): gradient boosting learns what the meal (calories, carbs, protein,
fat, fiber, sugar, meal type) predicts about the rise and about the exposure; the effect is
fitted on what is left. Person handled within person: outcome and exposure are demeaned per
person and each person's mean meal is a feature (Mundlak). 5-fold cross-fitting, repeated 3
times (median estimate); 95% CI from person-clustered standard errors. Still observational:
walks were self-chosen. See ML_COMPARISON.md for the benchmark that chose this engine.

Minutes-long; run by scripts/reproduce.sh or scripts/build-ml.sh (a server or container, not a laptop if you can help it):
  uv run --frozen python -I scripts/ml_effects.py <dataset_dir> <out_dir>
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import WALK_BINS, load, meal_responses, r  # noqa: E402
from realworld import cgmacros_meals  # noqa: E402

warnings.filterwarnings("ignore", message="The least populated class")
SEED = 20261017
FOLDS = 5
REPEATS = 3
B_PERSON = 300
HGB = dict(max_iter=200, learning_rate=0.05, min_samples_leaf=40, random_state=0)
MACROS = ["calories", "carbs_g", "protein_g", "fat_g", "fiber_g", "sugar_g"]
MT = ["mt_breakfast", "mt_dinner", "mt_snack"]  # lunch is the reference
HOUR_BINS = [(15, 17, "before 6pm"), (18, 18, "6-7pm"), (19, 19, "7-8pm"), (20, 20, "8-9pm"), (21, 21, "9-10pm"), (22, 27, "10pm or later")]
METHOD = ("Double machine learning: gradient boosting learns what the meal (calories, carbs, protein, fat, fiber, sugar"
          "{mt}) predicts about the rise and about {what}; the effect is fitted on what is left, within person. "
          "5-fold cross-fitting repeated 3 times; person-clustered 95% CI.")


def demean(v, g):
    return v - pd.Series(v).groupby(g).transform("mean").to_numpy()


def cluster_se(x, y, g, th):
    e = y - th * x
    s = pd.Series(x * e).groupby(g).sum().to_numpy()
    return float(np.sqrt(np.sum(s ** 2)) / np.sum(x ** 2))


def dml(df, t, y, covars, keep=False, min_meals=3):
    """Partially linear DML of y on t given covars, within person. Returns est/lo/hi/se/people/rows/residSd.
    t may be a list of exposure columns (e.g. one dummy per walk-length or dinner-hour bin): they are fitted
    jointly, like dummies in one regression, and a list of results comes back."""
    ts = t if isinstance(t, list) else [t]
    d = df.dropna(subset=ts + [y] + covars)
    d = d[d.groupby("participant_id")[y].transform("size") >= min_meals].reset_index(drop=True)
    g = d.participant_id.to_numpy()
    X = d[covars].astype(float)
    X = pd.concat([X, X.groupby(g).transform("mean").add_prefix("pm_")], axis=1)
    T = np.column_stack([demean(d[c].to_numpy(float), g) for c in ts])
    Y = demean(d[y].to_numpy(float), g)
    codes = pd.factorize(g)[0]
    ests, ses, rts, rys = [], [], [], []
    for rep in range(REPEATS):
        rt, ry = np.zeros_like(T), np.zeros(len(d))
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + rep).split(X, codes):
            for j in range(T.shape[1]):
                rt[te, j] = T[te, j] - HistGradientBoostingRegressor(**HGB).fit(X.iloc[tr], T[tr, j]).predict(X.iloc[te])
            ry[te] = Y[te] - HistGradientBoostingRegressor(**HGB).fit(X.iloc[tr], Y[tr]).predict(X.iloc[te])
        A = rt.T @ rt
        th = np.linalg.solve(A, rt.T @ ry)
        e = ry - rt @ th
        S = pd.DataFrame(rt * e[:, None]).groupby(g).sum().to_numpy()  # person-clustered sandwich
        Ai = np.linalg.inv(A)
        ests.append(th)
        ses.append(np.sqrt(np.diag(Ai @ (S.T @ S) @ Ai)))
        rts.append(rt)
        rys.append(ry)
    ests, ses = np.array(ests), np.array(ses)
    est = np.median(ests, axis=0)
    se = np.median(np.sqrt(ses ** 2 + (ests - est) ** 2), axis=0)  # median-aggregated over splits
    rt, ry = np.mean(rts, axis=0), np.mean(rys, axis=0)
    resid_sd = float(np.std(ry - rt @ est))
    outs = [{"est": float(est[j]), "lo": float(est[j] - 1.96 * se[j]), "hi": float(est[j] + 1.96 * se[j]), "se": float(se[j]),
             "people": int(len(np.unique(g))), "rows": int(len(d)), "residSd": resid_sd} for j in range(len(ts))]
    if keep:
        outs[0]["_frame"] = pd.DataFrame({"participant_id": g, "flag": d[ts[0]].to_numpy(float), "rt": rt[:, 0], "ry": ry})
    return outs if isinstance(t, list) else outs[0]


def person_effects(frame, rng):
    """Each person's effect on their own cross-fitted residuals; bootstrap of their meals for the SE."""
    res = {}
    for pid, sub in frame.groupby("participant_id"):
        n_flag = int(sub.flag.sum())
        if n_flag < 3 or len(sub) - n_flag < 3:
            res[pid] = None
            continue
        rt, ry = sub.rt.to_numpy(), sub.ry.to_numpy()
        est = float(np.sum(rt * ry) / np.sum(rt ** 2))
        idx = rng.integers(0, len(sub), (B_PERSON, len(sub)))
        num, den = (rt[idx] * ry[idx]).sum(1), (rt[idx] ** 2).sum(1)
        boots = num[den > 0] / den[den > 0]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[pid] = {"est": est, "se": float(np.std(boots)), "lo": float(lo), "hi": float(hi),
                    "n_flag": n_flag, "n_other": int(len(sub) - n_flag)}
    return res


def forecast(mr):
    """Out-of-fold gradient-boosting forecast of every meal's rise (pre-meal inputs only)."""
    X = mr[MACROS + MT + ["baseline", "hour"]].astype(float).copy()
    X["pre_slope"] = (mr.baseline - mr.pre15).fillna(0.0)
    X["med"] = (mr.mediterranean_style == "yes").astype(float)
    y = mr.rise.to_numpy(float)
    g = mr.participant_id.to_numpy()
    pred = np.zeros(len(mr))
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, pd.factorize(g)[0]):
        pm = pd.Series(y[tr]).groupby(g[tr]).mean()  # person's typical rise, training rows only
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        Xtr["person_mean"] = pd.Series(g[tr]).map(pm).to_numpy()
        Xte["person_mean"] = pd.Series(g[te]).map(pm).fillna(y[tr].mean()).to_numpy()
        pred[te] = HistGradientBoostingRegressor(**HGB).fit(Xtr, y[tr]).predict(Xte)
    err = y - pred
    q = np.percentile(err, [10, 90])
    return pred, {"mae": float(np.mean(np.abs(err))), "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
                  "err10": float(q[0]), "err90": float(q[1]), "meals": int(len(mr))}


FAIR_MIN_PEOPLE = 20
FAIR_ATTRS = (("race_ethnicity", "Race and ethnicity"), ("sex", "Sex"), ("age_band", "Age"),
              ("preferred_language", "Preferred language"), ("cancer_type", "Cancer type"))


def q_test(effects):
    """Cochran Q: do the subgroup effects differ more than chance? Returns p (None if under 2 groups)."""
    from scipy.stats import chi2
    e = [x for x in effects if x and x.get("se")]
    if len(e) < 2:
        return None
    w = np.array([1 / x["se"] ** 2 for x in e])
    est = np.array([x["est"] for x in e])
    pooled = (w * est).sum() / w.sum()
    return float(chi2.sf((w * (est - pooled) ** 2).sum(), len(e) - 1))


def fairness_table(mr, parts, walk_cov, rng):
    """Forecast error and the two effects by subgroup. Groups under FAIR_MIN_PEOPLE get no effect estimate."""
    info = parts.set_index("participant_id")
    d = mr.copy()
    d["age_band"] = pd.cut(d.participant_id.map(info.age), [0, 59, 69, 200], labels=["under 60", "60-69", "70+"]).astype(str)
    for col in ("race_ethnicity", "sex", "preferred_language", "cancer_type"):
        d[col] = d.participant_id.map(info[col])
    d["abs_err"] = (d.rise - d.pred).abs()
    out = {}
    for col, label in FAIR_ATTRS:
        rows = []
        for val, sub in d.groupby(col):
            people = sub.participant_id.unique()
            per = sub.groupby("participant_id").abs_err.agg(["sum", "count"])
            boots = [per.loc[k].sum()["sum"] / per.loc[k].sum()["count"] for k in
                     (rng.choice(people, len(people)) for _ in range(300))]
            row = {"group": str(val), "people": int(len(people)), "meals": int(len(sub)),
                   "forecastMae": {"est": r(float(sub.abs_err.mean()), 2), "lo": r(float(np.percentile(boots, 2.5)), 2),
                                   "hi": r(float(np.percentile(boots, 97.5)), 2)}, "walk": None, "lateDinner": None}
            if len(people) >= FAIR_MIN_PEOPLE:
                sd = sub[(sub.meal_type == "dinner") & (sub.work_schedule != "night shift")]
                try:
                    row["walk"] = eff(dml(sub, "walked", "rise", walk_cov))
                    if sd.participant_id.nunique() >= FAIR_MIN_PEOPLE:
                        row["lateDinner"] = eff(dml(sd, "late", "rise", MACROS))
                except Exception as ex:  # too little variation inside a small group
                    row["note"] = f"effect not estimable: {type(ex).__name__}"
            else:
                row["note"] = f"fewer than {FAIR_MIN_PEOPLE} people: effects not estimated"
            rows.append(row)
        out[col] = {"label": label, "groups": rows,
                    "walkDiffersP": q_test([x["walk"] for x in rows]), "lateDinnerDiffersP": q_test([x["lateDinner"] for x in rows])}
    return out


def eff(e, scale=1.0):
    return {k: (r(v * scale, 2) if k in ("est", "lo", "hi", "se") else (r(v, 2) if isinstance(v, float) else v))
            for k, v in e.items() if not k.startswith("_")}


def main(data_dir, out_dir):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    parts, checks, daily, meals, glu, demo_ids = load(data_dir)
    mr = meal_responses(meals, glu, parts).reset_index(drop=True)
    for c in MT:
        mr[c] = (mr.meal_type == c[3:]).astype(float)
    day = mr[mr.work_schedule != "night shift"]
    din = day[day.meal_type == "dinner"]
    oth = mr[mr.meal_type != "dinner"]
    walk_cov = MACROS + MT
    out = {"engine": "double machine learning (gradient boosting), within person", "folds": FOLDS, "repeats": REPEATS, "seed": SEED,
           "method": {"walk": METHOD.format(mt=", meal type", what="whether you walked"),
                      "walkDinner": METHOD.format(mt="", what="whether you walked"),
                      "late": METHOD.format(mt="", what="whether dinner started at 8 pm or later"),
                      "hour": METHOD.format(mt="", what="the dinner start time"),
                      "negctrl": METHOD.format(mt=", meal type", what="whether you walked").replace("the rise", "the pre-meal reading")}}

    c = {}
    walk = dml(mr, "walked", "rise", walk_cov, keep=True)
    c["walk"] = walk
    c["walkIauc"] = dml(mr, "walked", "iauc", walk_cov)
    c["walkAfterDinner"] = dml(din, "walked", "rise", MACROS)
    c["walkAfterOtherMeals"] = dml(oth, "walked", "rise", walk_cov)
    late = dml(din, "late", "rise", MACROS, keep=True)
    c["lateDinner"] = late
    c["lateDinnerIauc"] = dml(din, "late", "iauc", MACROS)
    c["carbsPer10g"] = dml(mr, "carbs_g", "rise", ["protein_g", "fat_g", "fiber_g", "walked"] + MT)
    c["negativeControl"] = dml(mr[mr.pre15.notna()], "walked", "pre15", walk_cov)
    print(f"cohort {time.time() - t0:.0f}s", file=sys.stderr)

    by_group = {}
    for bg, sub in mr.groupby("blood_sugar_group"):
        sd = sub[(sub.meal_type == "dinner") & (sub.work_schedule != "night shift")]
        by_group[bg] = {"walk": eff(dml(sub, "walked", "rise", walk_cov)), "lateDinner": eff(dml(sd, "late", "rise", MACROS))}

    dose = [{"bin": "none", "est": 0.0, "lo": 0.0, "hi": 0.0, "n_meals": int((mr.walked_after_min == 0).sum())}]
    bins = [(lo_, hi_, lab) for lo_, hi_, lab in WALK_BINS[1:] if ((mr.walked_after_min >= lo_) & (mr.walked_after_min <= hi_)).any()]
    md = mr.copy()
    for lo_, hi_, lab in bins:
        md[f"w_{lab}"] = ((md.walked_after_min >= lo_) & (md.walked_after_min <= hi_)).astype(float)
    for (lo_, hi_, lab), e in zip(bins, dml(md, [f"w_{lab}" for *_, lab in bins], "rise", walk_cov)):
        dose.append({"bin": lab, **eff(e), "n_meals": int(md[f"w_{lab}"].sum())})
    hh = din.ts.dt.hour.where(din.ts.dt.hour >= 4, din.ts.dt.hour + 24)
    dh = din.copy()
    dh["hb"] = None
    for a, b, lab in HOUR_BINS:
        dh.loc[(hh >= a) & (hh <= b), "hb"] = lab
    dh = dh[dh.hb.notna()]
    labs = [lab for *_, lab in HOUR_BINS[1:]]
    for lab in labs:
        dh[f"h_{lab}"] = (dh.hb == lab).astype(float)
    hours = [{"bin": "before 6pm", "est": 0.0, "lo": 0.0, "hi": 0.0, "n_meals": int((dh.hb == "before 6pm").sum())}]
    for lab, e in zip(labs, dml(dh, [f"h_{lab}" for lab in labs], "rise", MACROS)):
        hours.append({"bin": lab, **eff(e), "n_meals": int((dh.hb == lab).sum())})
    print(f"curves {time.time() - t0:.0f}s", file=sys.stderr)

    pw, pl = person_effects(walk["_frame"], rng), person_effects(late["_frame"], rng)
    personal = {pid: {"walk": pw.get(pid), "late": pl.get(pid)} for pid in sorted(set(mr.participant_id))}

    pred, fc = forecast(mr)
    mr["pred"] = pred
    dd = mr[mr.meal_type == "dinner"].groupby("participant_id")
    fc["dinnerByPerson"] = {pid: {"pred": float(s.pred.mean()), "n": int(len(s))} for pid, s in dd}
    print(f"personal+forecast {time.time() - t0:.0f}s", file=sys.stderr)
    fairness = fairness_table(mr, parts, walk_cov, rng)
    print(f"fairness {time.time() - t0:.0f}s", file=sys.stderr)

    cg = cgmacros_meals(data_dir)
    cgm = ["calories", "carbs_g", "protein_g", "fat_g", "fiber_g"]
    for t_ in ("breakfast", "dinner", "snack"):
        cg[f"mt_{t_}"] = (cg.meal_type == t_).astype(float)
    real = {"walk10": eff(dml(cg[cg.walk10.notna()], "walk10", "rise", cgm + MT)),
            "lateDinner": eff(dml(cg[cg.meal_type == "dinner"], "late", "rise", cgm)),
            "note": "Same DML as the program, on CGMacros (calories, carbs, protein, fat, fiber; no sugar column)."}

    out["cohort"] = {k: eff(v, 10 if k == "carbsPer10g" else 1) for k, v in c.items()}
    out["byGroup"] = by_group
    out["walkDose"] = dose
    out["dinnerHour"] = hours
    out["personal"] = {pid: {k: (eff(v) if v else None) for k, v in d_.items()} for pid, d_ in personal.items()}
    out["forecast"] = {k: (r(v, 2) if isinstance(v, float) else v) for k, v in fc.items() if k != "dinnerByPerson"}
    out["forecast"]["dinnerByPerson"] = {k: {"pred": r(v["pred"], 1), "n": v["n"]} for k, v in fc["dinnerByPerson"].items()}
    out["real"] = real
    out["fairness"] = fairness
    out["seconds"] = round(time.time() - t0)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/ml_effects.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("cohort", "walkDose", "dinnerHour", "byGroup", "real")}, indent=1))
    print(json.dumps({k: v for k, v in out["forecast"].items() if k != "dinnerByPerson"}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
