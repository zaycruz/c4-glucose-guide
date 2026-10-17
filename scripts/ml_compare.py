"""Statistics vs machine learning on the two jobs Glucose Guide has (see ML_COMPARISON.md).

  A. Predict a meal's sugar rise (program data: later-days split and new-people split;
     CGMacros: new-people split).
  B. Recover a planted walk-like effect from confounded data: naive, fixed effects (FE),
     gradient-boosting S-learner, double ML (DML). Linear and nonlinear confounding.
  C. Recover planted per-person effects: own-data OLS, empirical-Bayes blend, ML (S-learner).
  D. Real walk and late-dinner effects: FE vs DML.

Usage: uv run --with pandas --with numpy --with scikit-learn python -I scripts/ml_compare.py <dataset_dir> <out_json>
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import dl_tau2, load, meal_responses, person_effect, r, shrink  # noqa: E402

SEED = 20261017
REPS_EFFECT = 20
REPS_PERSON = 10
B_METRIC = 200
TAU = -10.0          # planted overall effect, mg/dL
TAU_SD = 5.0         # planted spread of personal effects
TREAT_SHARE = 0.10   # planted treatment share, like real walks
HGB = dict(max_iter=200, learning_rate=0.05, min_samples_leaf=40, random_state=0)

MACROS = ["calories", "carbs_g", "protein_g", "fat_g", "fiber_g", "sugar_g"]
MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]


def features(df, macros):
    X = df[macros + ["baseline", "hour"]].astype(float).copy()
    X["pre_slope"] = (df.baseline - df.pre15).fillna(0.0) if "pre15" in df else 0.0
    for t in MEAL_TYPES:
        X[f"mt_{t}"] = (df.meal_type == t).astype(float)
    if "mediterranean_style" in df:
        X["med"] = (df.mediterranean_style == "yes").astype(float)
    return X


# ---------- A. prediction ----------

def metrics(y, p, groups, rng):
    err = y - p
    people = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in people}

    def calc(idx):
        e = err[idx]
        yy = y[idx]
        return [np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2)), 1 - np.sum(e ** 2) / np.sum((yy - yy.mean()) ** 2)]

    est = calc(np.arange(len(y)))
    boots = []
    for _ in range(B_METRIC):
        idx = np.concatenate([by[g] for g in rng.choice(people, len(people))])
        boots.append(calc(idx))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {k: {"est": r(est[i], 2), "lo": r(lo[i], 2), "hi": r(hi[i], 2)} for i, k in enumerate(["mae", "rmse", "r2"])}


def person_mean_feature(train_y, train_g, g, fallback):
    means = pd.Series(train_y).groupby(train_g).mean()
    return pd.Series(g).map(means).fillna(fallback).to_numpy()


def models_for(split):
    """Same model menu for every split. Person information enters only via training-row means."""
    return {
        "Person average (baseline)": "person_mean",
        "Linear, per-person intercept (fixed effects)": "fe",
        "Ridge, per-person intercept": "ridge",
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=0),
        "Gradient boosting": HistGradientBoostingRegressor(**HGB),
    }


def fit_predict(kind, Xtr, ytr, gtr, Xte, gte, known_people):
    """known_people: test people appear in training (later-days split); else new people."""
    gm = ytr.mean()
    if kind == "person_mean":
        return person_mean_feature(ytr, gtr, gte, gm) if known_people else np.full(len(Xte), gm)
    if kind in ("fe", "ridge"):
        if known_people:
            # within-person slopes, then person intercept from training residuals
            Xd = Xtr - Xtr.groupby(gtr).transform("mean")
            yd = ytr - pd.Series(ytr).groupby(gtr).transform("mean").to_numpy()
            m = (LinearRegression() if kind == "fe" else Ridge(alpha=10.0)).fit(Xd, yd)
            icpt = pd.Series(ytr - m.predict(Xtr) + m.intercept_).groupby(gtr).mean()
            return m.predict(Xte) - m.intercept_ + pd.Series(gte).map(icpt).fillna(icpt.mean()).to_numpy()
        m = (LinearRegression() if kind == "fe" else Ridge(alpha=10.0)).fit(Xtr, ytr)
        return m.predict(Xte)
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    if known_people:  # person average from training rows only
        Xtr2["person_mean"] = person_mean_feature(ytr, gtr, gtr, gm)
        Xte2["person_mean"] = person_mean_feature(ytr, gtr, gte, gm)
    return kind.fit(Xtr2, ytr).predict(Xte2)


def predict_task(df, X, rng, label):
    y = df.rise.to_numpy(float)
    g = df.participant_id.to_numpy()
    out = {"label": label, "meals": int(len(df)), "people": int(len(np.unique(g))), "splits": {}}
    splits = {}
    if "order" in df:
        test = (df.order >= 0.75).to_numpy()
        splits["later days (same people)"] = [(np.where(~test)[0], np.where(test)[0], True)]
    gkf = GroupKFold(n_splits=5)
    splits["new people (5-fold)"] = [(tr, te, False) for tr, te in gkf.split(X, y, g)]
    for sname, folds in splits.items():
        res = {}
        for mname, kind in models_for(sname).items():
            pred = np.full(len(y), np.nan)
            for tr, te, known in folds:
                k = kind if isinstance(kind, str) else type(kind)(**kind.get_params())
                pred[te] = fit_predict(k, X.iloc[tr], y[tr], g[tr], X.iloc[te], g[te], known)
            te_all = np.isfinite(pred)
            res[mname] = metrics(y[te_all], pred[te_all], g[te_all], rng)
        out["splits"][sname] = res
    return out


# ---------- B. planted effect, confounded ----------

def zs(x):
    return (x - x.mean()) / x.std()


def plant(base, scenario, rng, tau=TAU, tau_i=None):
    d = base.copy()
    u = d.participant_id.map(pd.Series(rng.normal(0, 0.7, d.participant_id.nunique()), index=d.participant_id.unique()))
    if scenario == "linear":
        logit = 0.8 * zs(d.carbs_g) + 0.8 * zs(d.baseline) - 0.6 * zs(d.hour) + u
        extra = np.zeros(len(d))
    elif scenario == "nonlinear":
        high = (d.baseline > d.baseline.quantile(0.75)).astype(float)
        late_big = ((d.hour >= 19) & (d.carbs_g > 60)).astype(float)
        logit = 2.0 * high + 1.5 * late_big + u
        extra = 12.0 * high + 8.0 * late_big  # the same thresholds also move the rise
    else:  # random, for personal effects
        logit = u * 0
        extra = np.zeros(len(d))
    # intercept so about TREAT_SHARE of meals are treated
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        share = (1 / (1 + np.exp(-(logit + mid)))).mean()
        lo, hi = (mid, hi) if share < TREAT_SHARE else (lo, mid)
    p = 1 / (1 + np.exp(-(logit + mid)))
    d["T"] = (rng.random(len(d)) < p).astype(float)
    eff = tau if tau_i is None else d.participant_id.map(tau_i).to_numpy()
    d["y"] = d.rise + eff * d["T"] + extra
    return d


def cluster_se(x_res, y_res, groups, theta):
    """Person-clustered sandwich SE for a one-regressor residual-on-residual fit."""
    e = y_res - theta * x_res
    s = pd.Series(x_res * e).groupby(groups).sum().to_numpy()
    return np.sqrt(np.sum(s ** 2)) / np.sum(x_res ** 2)


def est_naive(d, Xc):
    t, y = d["T"].to_numpy(), d.y.to_numpy()
    th = y[t == 1].mean() - y[t == 0].mean()
    se = cluster_se(t - t.mean(), y - y.mean(), d.participant_id.to_numpy(), th)
    return th, se


def est_fe(d, Xc):
    g = d.participant_id.to_numpy()
    M = pd.concat([d[["T", "y"]], Xc], axis=1)
    Md = M - M.groupby(g).transform("mean")
    Z = Md[list(Xc.columns)].to_numpy()
    # partial the covariates out of T and y (Frisch-Waugh), then one-regressor fit
    bt = np.linalg.lstsq(Z, Md["T"].to_numpy(), rcond=None)[0]
    by = np.linalg.lstsq(Z, Md["y"].to_numpy(), rcond=None)[0]
    tr = Md["T"].to_numpy() - Z @ bt
    yr = Md["y"].to_numpy() - Z @ by
    th = np.sum(tr * yr) / np.sum(tr ** 2)
    return th, cluster_se(tr, yr, g, th)


def with_person(Xc, g):
    """Person handled the Mundlak way: raw covariates plus each person's covariate means
    (500 people is too many for a categorical split). Outcomes are demeaned within person."""
    X = Xc.copy()
    means = Xc.groupby(g).transform("mean")
    for c in Xc.columns:
        X[f"pm_{c}"] = means[c].to_numpy()
    return X


def demean(v, g):
    return v - pd.Series(v).groupby(g).transform("mean").to_numpy()


def est_slearner(d, Xc, per_person=False):
    g = d.participant_id.to_numpy()
    X = with_person(Xc, g)
    X.insert(0, "T", d["T"].to_numpy())
    X["pm_T"] = pd.Series(d["T"].to_numpy()).groupby(g).transform("mean").to_numpy()
    m = HistGradientBoostingRegressor(**HGB).fit(X, demean(d.y.to_numpy(float), g))
    X1, X0 = X.copy(), X.copy()
    X1["T"], X0["T"] = 1.0, 0.0
    diff = m.predict(X1) - m.predict(X0)
    if per_person:
        return pd.Series(diff).groupby(d.participant_id.to_numpy()).mean()
    return diff.mean(), None


def est_dml(d, Xc, folds=3):
    g = d.participant_id.to_numpy()
    X = with_person(Xc, g)
    t, y = demean(d["T"].to_numpy(float), g), demean(d.y.to_numpy(float), g)
    th_res, y_res = np.zeros(len(d)), np.zeros(len(d))
    # rows stratified by person, so every person is in every training fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for tr, te in skf.split(X, pd.factorize(g)[0]):
        mt = HistGradientBoostingRegressor(**HGB).fit(X.iloc[tr], t[tr])
        my = HistGradientBoostingRegressor(**HGB).fit(X.iloc[tr], y[tr])
        th_res[te] = t[te] - mt.predict(X.iloc[te])
        y_res[te] = y[te] - my.predict(X.iloc[te])
    th = np.sum(th_res * y_res) / np.sum(th_res ** 2)
    return th, cluster_se(th_res, y_res, g, th)


ESTIMATORS = {
    "Naive difference": est_naive,
    "Fixed effects (statistics)": est_fe,
    "S-learner (gradient boosting)": est_slearner,
    "Double ML (gradient boosting, shipped)": est_dml,
}


def effect_benchmark(base, Xc, rng):
    out = {}
    for scen in ("linear", "nonlinear"):
        rows = {k: [] for k in ESTIMATORS}
        for _ in range(REPS_EFFECT):
            d = plant(base, scen, rng)
            for k, f in ESTIMATORS.items():
                th, se = f(d, Xc)
                rows[k].append((th, se))
        res = {}
        for k, v in rows.items():
            th = np.array([a for a, _ in v])
            se = [b for _, b in v]
            cov = None
            if se[0] is not None:
                se = np.array(se)
                cov = r(100 * np.mean((th - 1.96 * se <= TAU) & (TAU <= th + 1.96 * se)), 0)
            res[k] = {"mean": r(th.mean(), 2), "bias": r(th.mean() - TAU, 2), "rmse": r(np.sqrt(np.mean((th - TAU) ** 2)), 2),
                      "mcSd": r(th.std(ddof=1), 2), "coverage95": cov}
        out[scen] = res
    return out


# ---------- C. planted personal effects ----------

def personal_benchmark(base, Xc, rng):
    people = base.participant_id.unique()
    from ml_effects import MACROS as ML_MACROS, MT as ML_MT, dml, person_effects
    errs = {k: [] for k in ("Own data only (OLS)", "Blend with cohort (empirical Bayes, statistics)", "S-learner per person (gradient boosting)",
                            "Double ML residuals, blended with cohort (shipped)")}
    scored = []
    for _ in range(REPS_PERSON):
        tau_i = pd.Series(rng.normal(TAU, TAU_SD, len(people)), index=people)
        d = plant(base, "random", rng, tau_i=tau_i)
        d["rise"] = d.y  # person_effect fits rise
        own = {}
        for pid, sub in d.groupby("participant_id"):
            own[pid] = person_effect(sub, "T", ["carbs_g", "baseline"])
        ok = [p for p, e in own.items() if e is not None]
        fe = est_fe(d, Xc)[0]
        tau2 = dl_tau2([own[p] for p in ok])
        boots = np.array([own[p]["est"] for p in ok])
        coh = {"est": fe, "lo": fe - 1.96 * boots.std() / np.sqrt(len(ok)), "hi": fe + 1.96 * boots.std() / np.sqrt(len(ok))}
        ml = est_slearner(d, Xc, per_person=True)
        for c_ in ML_MT:
            d[c_] = (d.meal_type == c_[3:]).astype(float)
        dm = dml(d, "T", "y", ML_MACROS + ML_MT, keep=True, min_meals=1)
        pdm = person_effects(dm["_frame"], rng)
        tau2_dm = dl_tau2([e for e in pdm.values() if e])
        for p in ok:
            errs["Own data only (OLS)"].append(own[p]["est"] - tau_i[p])
            errs["Blend with cohort (empirical Bayes, statistics)"].append(shrink(own[p], coh, tau2)["est"] - tau_i[p])
            errs["S-learner per person (gradient boosting)"].append(ml[p] - tau_i[p])
            own_dm = pdm.get(p)
            est_dm = shrink(own_dm, dm, tau2_dm)["est"] if own_dm else dm["est"]
            errs["Double ML residuals, blended with cohort (shipped)"].append(est_dm - tau_i[p])
        scored.append(len(ok))
    return {"peopleScoredPerRep": int(np.median(scored)), "reps": REPS_PERSON,
            "rmse": {k: r(np.sqrt(np.mean(np.square(v))), 2) for k, v in errs.items()},
            "plantedSpreadSd": TAU_SD}


# ---------- CGMacros frame ----------

def cgmacros_frame(d):
    mm = pd.read_csv(f"{d}/CGMacros/cgmacros_minute_by_minute.csv", parse_dates=["timestamp"],
                     usecols=["participant_id", "timestamp", "glucose_libre_mg_dl"])
    meals = pd.read_csv(f"{d}/CGMacros/cgmacros_meals.csv", parse_dates=["timestamp"])
    meals["meal_type"] = meals.meal_type.str.lower().str.replace(r"snacks?( 1)?", "snack", regex=True)
    meals = meals.sort_values(["participant_id", "timestamp"])
    meals["next"] = meals.groupby("participant_id").timestamp.shift(-1)
    meals = meals[(meals.next.isna()) | ((meals.next - meals.timestamp) > pd.Timedelta(minutes=120))]
    meals = meals[meals.meal_carbs_g >= 10]
    by = {pid: g.set_index("timestamp").glucose_libre_mg_dl for pid, g in mm.groupby("participant_id")}
    rows = []
    for m in meals.itertuples():
        g = by.get(m.participant_id)
        if g is None:
            continue
        win = g.loc[m.timestamp: m.timestamp + pd.Timedelta(minutes=120)]
        pre = g.loc[m.timestamp - pd.Timedelta(minutes=15): m.timestamp - pd.Timedelta(minutes=14)]
        if len(win) < 100 or pd.isna(win.iloc[0]):
            continue
        rows.append({"participant_id": m.participant_id, "meal_type": m.meal_type, "calories": m.meal_calories,
                     "carbs_g": m.meal_carbs_g, "protein_g": m.meal_protein_g, "fat_g": m.meal_fat_g,
                     "fiber_g": m.meal_fiber_g, "baseline": win.iloc[0],
                     "pre15": pre.iloc[0] if len(pre) else np.nan,
                     "hour": m.timestamp.hour + m.timestamp.minute / 60, "rise": win.max() - win.iloc[0]})
    return pd.DataFrame(rows).dropna(subset=["rise", "carbs_g", "protein_g", "fat_g", "fiber_g", "calories"])


def main(data_dir, out_path):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    parts, checks, daily, meals, glu, _ = load(data_dir)
    mr = meal_responses(meals, glu, parts).reset_index(drop=True)
    mr["order"] = mr.groupby("participant_id").ts.rank(pct=True)
    out = {"generated": time.strftime("%Y-%m-%d"), "seed": SEED, "plantedEffect": TAU, "treatShare": TREAT_SHARE}

    # A. prediction, before the walk (walk is only known afterwards) and with it
    X = features(mr, MACROS)
    feat_dir = os.environ.get("FEATURES_DIR")  # optional: feature tables for scripts/pycaret_compare.py
    if feat_dir:
        X.assign(participant_id=mr.participant_id.to_numpy(), rise=mr.rise.to_numpy()).to_csv(f"{feat_dir}/program_features.csv", index=False)
    out["predictProgram"] = predict_task(mr, X, rng, "Program data, before eating (no walk)")
    Xw = X.copy()
    Xw["walked_after_min"] = mr.walked_after_min.astype(float)
    out["predictProgramWithWalk"] = predict_task(mr, Xw, rng, "Program data, with walk length")
    print(f"A program done {time.time() - t0:.0f}s", file=sys.stderr)
    cg = cgmacros_frame(data_dir).reset_index(drop=True)
    Xcg = features(cg, ["calories", "carbs_g", "protein_g", "fat_g", "fiber_g"])
    if feat_dir:
        Xcg.assign(participant_id=cg.participant_id.to_numpy(), rise=cg.rise.to_numpy()).to_csv(f"{feat_dir}/cgmacros_features.csv", index=False)
    out["predictCgmacros"] = predict_task(cg, Xcg, rng,
                                          "CGMacros real adults, before eating")
    print(f"A cgmacros done {time.time() - t0:.0f}s", file=sys.stderr)

    # B/C. planted effects on meals with no real walk
    base = mr[mr.walked == 0].reset_index(drop=True)
    Xc = features(base, ["carbs_g"])[["carbs_g", "baseline", "hour", "mt_breakfast", "mt_dinner", "mt_snack"]]
    out["effectBenchmark"] = effect_benchmark(base, Xc, rng)
    out["effectBenchmark"]["meals"] = int(len(base))
    out["effectBenchmark"]["reps"] = REPS_EFFECT
    print(f"B done {time.time() - t0:.0f}s", file=sys.stderr)
    out["personalBenchmark"] = personal_benchmark(base, Xc, rng)
    print(f"C done {time.time() - t0:.0f}s", file=sys.stderr)

    # D. real effects: FE vs DML, same covariates
    real = {}
    allm = mr.copy()
    allm["T"], allm["y"] = allm.walked.astype(float), allm.rise
    Xa = features(allm, ["carbs_g"])[["carbs_g", "baseline", "hour", "mt_breakfast", "mt_dinner", "mt_snack"]]
    din = mr[(mr.meal_type == "dinner") & (mr.work_schedule != "night shift")].reset_index(drop=True)
    din["T"], din["y"] = din.late.astype(float), din.rise
    Xd = features(din, ["carbs_g"])[["carbs_g", "baseline"]]
    for name, d, Xr in (("walk", allm, Xa), ("lateDinner", din, Xd)):
        real[name] = {}
        for k in ("Fixed effects (statistics)", "Double ML (gradient boosting, shipped)"):
            th, se = ESTIMATORS[k](d, Xr)
            real[name][k] = {"est": r(th), "lo": r(th - 1.96 * se), "hi": r(th + 1.96 * se)}
    real["note"] = "Same covariates for both (carbs, pre-meal sugar, hour, meal type for walks; carbs and pre-meal sugar for late dinners). FE here adds pre-meal sugar and hour, so it can differ slightly from the advice numbers."
    out["realEffects"] = real
    out["seconds"] = round(time.time() - t0)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
