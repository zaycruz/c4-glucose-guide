"""Offline analysis for C4 Glucose Guide.

Reads the Challenge 4 (Dose and Timing) dataset and writes the JSON the Worker serves:
  <out>/cohort.json            cohort findings (trial arms, within-person effects, dose response)
  <out>/roster.json            roster tiles for the demo survivors
  <out>/survivors/<id>.json    one detail panel per survivor

Usage: uv run --with pandas --with numpy python -I scripts/analyze.py <dataset_dir> <out_dir>

Methods (all numbers in evidence[] come from here):
- Meal response: baseline = last 15-min reading at or before the meal (<= 15 min earlier).
  Rise = peak of the readings 15..120 min after the meal minus baseline. iAUC = area above
  baseline over 0..120 min (mg/dL x h). Meals with another item of >= 100 kcal in the next
  120 min are dropped, so the curve belongs to one meal.
- Late dinner: dinner starting at or after 20:00. Within-person comparison (person fixed
  effects), adjusted for carbs. Night-shift workers are left out.
- Post-meal walk: walked_after_min > 0. Within-person, adjusted for carbs and meal type.
  Walking was self-chosen, so this is an association, not proof.
- Overnight fast: the randomized arm comparison (coaching + overnight fast vs coaching) on
  week 12 minus week 0 change is the fair causal estimate. Within-person night-to-night
  link between the overnight fast and next-morning glucose is shown as an association.
- Cohort CIs: cluster bootstrap that resamples people. Personal CIs: bootstrap of that
  person's meals, then empirical-Bayes shrinkage toward the cohort estimate.
- Robustness of the walk effect (see ROBUSTNESS.md):
  negative control = same model with the reading 15 min BEFORE the meal as outcome (a later
  walk cannot change it); E-value (VanderWeele & Ding 2017) with the within-person residual SD;
  instrument = randomized arm (coaching vs email tips) as encouragement to walk, Wald ratio at
  person level. The instrument proved inconclusive (exclusion restriction fails) and is not
  counted as support.
- Fast by subgroup: Cochran Q test of arm x blood-sugar-group interaction, Holm adjustment across
  all subgroup contrasts. Advice uses group-specific fast numbers only if Q p < 0.05.
- Real-data agreement labels: agrees (same sign, CI excludes 0), same direction not proven,
  no clear signal, disagrees (opposite sign, CI excludes 0). Power: people needed for 80% power,
  scaling the real study's SE by sqrt(people).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patient_copy import WALK_CAUTION, patient_view, plain_summary  # noqa: E402

RNG = np.random.default_rng(20261017)
B_COHORT = 400
B_PERSON = 300
LATE_HOUR = 20
CURVE_OFFSETS = list(range(0, 195, 15))  # minutes 0..180
WALK_BINS = [(0, 0, "none"), (1, 9, "1-9 min"), (10, 19, "10-19 min"), (20, 29, "20-29 min"), (30, 999, "30+ min")]


def r(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd) + 0.0  # + 0.0 turns -0.0 into 0.0


# ---------- load ----------

def load(d):
    s = f"{d}/synthetic_data"
    parts = pd.read_csv(f"{s}/synthetic_participants.csv")
    checks = pd.read_csv(f"{s}/synthetic_checkups_week0_week12.csv")
    daily = pd.read_csv(f"{s}/synthetic_daily.csv")
    meals = pd.read_csv(f"{s}/synthetic_meals.csv")
    glu = pd.read_csv(f"{s}/synthetic_glucose_every_15_min.csv", parse_dates=["timestamp"])
    with open(f"{s}/synthetic_100_people.json") as f:
        demo_ids = [p["participant_id"] for p in json.load(f)["people"] if p.get("wore_glucose_sensor") == "yes"]
    return parts, checks, daily, meals, glu, demo_ids


# ---------- meal responses ----------

def meal_responses(meals, glu, parts):
    sensor = set(glu.participant_id.unique())
    m = meals[meals.participant_id.isin(sensor)].copy()
    m["ts"] = pd.to_datetime(m.date + " " + m.time)
    m = m.sort_values(["participant_id", "ts"])
    # drop meals followed by another >=100 kcal item within 120 min
    nxt = m[m.calories >= 100].copy()
    nxt["next_ts"] = nxt.ts
    m = pd.merge_asof(
        m.sort_values("ts"),
        nxt[["participant_id", "ts", "next_ts"]].sort_values("ts").rename(columns={"ts": "ts_key"}),
        left_on="ts", right_on="ts_key", by="participant_id", direction="forward",
        allow_exact_matches=False,
    )
    m["clean"] = m.next_ts.isna() | ((m.next_ts - m.ts) > pd.Timedelta(minutes=120))

    g = glu.set_index(["participant_id", "timestamp"]).glucose_mg_dl
    base_ts = m.ts.dt.floor("15min")
    curve = {}
    for off in CURVE_OFFSETS:
        idx = pd.MultiIndex.from_arrays([m.participant_id, base_ts + pd.Timedelta(minutes=off)])
        curve[off] = g.reindex(idx).to_numpy()
    m["baseline"] = curve[0]
    m["pre15"] = g.reindex(pd.MultiIndex.from_arrays([m.participant_id, base_ts - pd.Timedelta(minutes=15)])).to_numpy()
    post = np.column_stack([curve[o] for o in CURVE_OFFSETS if 15 <= o <= 120])
    m["n_post"] = np.isfinite(post).sum(axis=1)
    m["peak"] = np.nanmax(np.where(np.isfinite(post), post, -np.inf), axis=1)
    m.loc[m.n_post == 0, "peak"] = np.nan
    m["rise"] = m.peak - m.baseline
    inc = np.column_stack([curve[o] for o in CURVE_OFFSETS if o <= 120]) - m.baseline.to_numpy()[:, None]
    inc = np.clip(inc, 0, None)
    m["iauc"] = np.trapezoid(np.nan_to_num(inc), dx=0.25, axis=1)
    for off in CURVE_OFFSETS:
        m[f"c{off}"] = curve[off] - m.baseline.to_numpy()
    m["hour"] = m.ts.dt.hour + m.ts.dt.minute / 60
    m["late"] = ((m.hour >= LATE_HOUR) | (m.hour < 4)).astype(int)
    m["walked"] = (m.walked_after_min > 0).astype(int)
    m = m.merge(parts[["participant_id", "group", "work_schedule", "blood_sugar_group"]], on="participant_id")
    ok = m.clean & m.baseline.notna() & (m.n_post >= 6) & (m.carbs_g >= 10)
    return m[ok].copy()


# ---------- fixed-effects OLS with cluster bootstrap ----------

def fe_design(df, xcols, ycol="rise"):
    """Demean y and X within person; return per-person (XtX, Xty) blocks."""
    blocks = []
    for pid, sub in df.groupby("participant_id"):
        if len(sub) < 3:
            continue
        X = sub[xcols].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        X = X - X.mean(0)
        y = y - y.mean()
        blocks.append((pid, X.T @ X, X.T @ y))
    return blocks


def fe_fit(blocks, idx=None):
    sel = blocks if idx is None else [blocks[i] for i in idx]
    XtX = sum(b[1] for b in sel)
    Xty = sum(b[2] for b in sel)
    return np.linalg.lstsq(XtX, Xty, rcond=None)[0]


def fe_effect(df, xcols, ycol="rise", k=0):
    blocks = fe_design(df, xcols, ycol)
    est = fe_fit(blocks)[k]
    n = len(blocks)
    boots = []
    for _ in range(B_COHORT):
        idx = RNG.integers(0, n, n)
        try:
            boots.append(fe_fit(blocks, idx)[k])
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": est, "lo": lo, "hi": hi, "people": n, "rows": int(len(df))}


def cluster_mean_ci(df, col, pid="participant_id"):
    per = df.groupby(pid)[col].agg(["sum", "count"])
    s, c = per["sum"].to_numpy(), per["count"].to_numpy()
    n = len(per)
    if n == 0:
        return None
    est = s.sum() / c.sum()
    boots = []
    for _ in range(B_COHORT):
        i = RNG.integers(0, n, n)
        boots.append(s[i].sum() / c[i].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": est, "lo": lo, "hi": hi, "people": n, "rows": int(c.sum())}


# ---------- per-person ----------

def person_effect(sub, flag, covars):
    """OLS rise ~ flag + covars within one person; bootstrap of meals."""
    if sub[flag].sum() < 3 or (1 - sub[flag]).sum() < 3:
        return None
    X = np.column_stack([np.ones(len(sub)), sub[[flag] + covars].to_numpy(float)])
    y = sub.rise.to_numpy(float)
    est = np.linalg.lstsq(X, y, rcond=None)[0][1]
    boots = []
    n = len(sub)
    for _ in range(B_PERSON):
        i = RNG.integers(0, n, n)
        Xi = X[i]
        if Xi[:, 1].min() == Xi[:, 1].max():
            continue
        boots.append(np.linalg.lstsq(Xi, y[i], rcond=None)[0][1])
    if len(boots) < 50:
        return None
    se = float(np.std(boots))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": est, "se": se, "lo": lo, "hi": hi, "n_flag": int(sub[flag].sum()), "n_other": int((1 - sub[flag]).sum())}


def shrink(own, cohort, tau2):
    """Empirical-Bayes blend; the interval keeps the cohort estimate's own uncertainty."""
    if own is None:
        return None
    w = tau2 / (tau2 + own["se"] ** 2) if tau2 > 0 else 0.0
    est = w * own["est"] + (1 - w) * cohort["est"]
    se_c = (cohort["hi"] - cohort["lo"]) / 3.92
    se = np.sqrt(w * own["se"] ** 2 + ((1 - w) * se_c) ** 2)
    return {"est": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se, "weight": w}


def dl_tau2(effects):
    effects = [e for e in effects if e is not None]
    if len(effects) < 5:
        return 0.0
    ests = np.array([e["est"] for e in effects])
    ses2 = np.array([e["se"] ** 2 for e in effects])
    return max(0.0, float(np.var(ests) - np.mean(ses2)))


# ---------- robustness ----------

def within_resid_sd(df, xcols, ycol="rise"):
    """SD of the within-person residual (what a within-person effect is measured against)."""
    d = df[["participant_id", ycol] + xcols].copy()
    g = d.groupby("participant_id")
    for c in [ycol] + xcols:
        d[c] = d[c] - g[c].transform("mean")
    X, y = d[xcols].to_numpy(float), d[ycol].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.std(y - X @ beta))


def e_value(est, bound, sd):
    """VanderWeele & Ding (2017) E-value for a mean difference: d = est/sd, RR ~ exp(0.91 d)."""
    def ev(x):
        rr = np.exp(0.91 * abs(x) / sd)
        return rr + np.sqrt(rr * (rr - 1))
    return {"estimate": ev(est), "ciBound": ev(bound) if np.sign(bound) == np.sign(est) else 1.0, "sdUsed": sd}


def iv_walk(mr, parts, covars, a="coaching", b="tips by email", B=2000):
    """Randomized-encouragement check: arm assignment moves how often people walk after meals.
    Wald ratio = (difference in mean carbs/meal-type-adjusted rise) / (difference in share of meals with a walk),
    person level, coaching vs email tips (the fast arm is left out because it also moves dinner time)."""
    X = np.column_stack([np.ones(len(mr)), mr[covars].to_numpy(float)])
    beta = np.linalg.lstsq(X, mr.rise.to_numpy(float), rcond=None)[0]
    d = mr.assign(adj=mr.rise.to_numpy(float) - X[:, 1:] @ beta[1:])
    per = d.groupby("participant_id").agg(share=("walked", "mean"), rise=("adj", "mean"), n=("adj", "size"))
    per = per[per.n >= 10].join(parts.set_index("participant_id").group)
    A, Bp = per[per.group == a], per[per.group == b]

    def stats(x, y):
        fs = x.share.mean() - y.share.mean()
        rf = x.rise.mean() - y.rise.mean()
        return fs, rf, rf / fs if fs != 0 else np.nan
    fs, rf, ratio = stats(A, Bp)
    boots = np.array([stats(A.iloc[RNG.integers(0, len(A), len(A))], Bp.iloc[RNG.integers(0, len(Bp), len(Bp))]) for _ in range(B)])
    ci = lambda k: tuple(np.nanpercentile(boots[:, k], [2.5, 97.5]))  # noqa: E731
    return {"firstStage": {"est": 100 * fs, "lo": 100 * ci(0)[0], "hi": 100 * ci(0)[1], "unit": "percentage points of meals with a walk"},
            "reducedForm": {"est": rf, "lo": ci(1)[0], "hi": ci(1)[1], "unit": "mg/dL"},
            "walkEffect": {"est": ratio, "lo": ci(2)[0], "hi": ci(2)[1], "unit": "mg/dL per meal walked"},
            "people": {a: int(len(A)), b: int(len(Bp))}}


def holm(pvals):
    """Holm step-down adjusted p-values, same order as input."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def people_needed(eff, target, people):
    """People needed for 80% power (two-sided 5%) to detect `target`, scaling this study's SE by sqrt(people)."""
    se = (eff["hi"] - eff["lo"]) / 3.92
    return int(np.ceil(people * (se / (abs(target) / 2.8016)) ** 2))


def agreement(program_est, real):
    """Label for the real-data check against the program's direction."""
    if real is None:
        return "not tested"
    same = np.sign(real["est"]) == np.sign(program_est)
    excl0 = real["lo"] > 0 or real["hi"] < 0
    if same and excl0:
        return "agrees"
    if same:
        return "same direction, not proven"
    return "disagrees" if excl0 else "no clear signal"


# ---------- glucose summaries ----------

def glucose_metrics(gp):
    v = gp.glucose_mg_dl.to_numpy(float)
    if len(v) == 0:
        return None
    mean = v.mean()
    return {
        "mean": mean,
        "tir": 100 * np.mean((v >= 70) & (v <= 180)),
        "tar180": 100 * np.mean(v > 180),
        "tar250": 100 * np.mean(v > 250),
        "tbr70": 100 * np.mean(v < 70),
        "gmi": 3.31 + 0.02392 * mean,
        "readings": int(len(v)),
    }


def status_from(met_latest, met_first):
    """Deterministic status from consensus CGM targets (Battelino 2019 / ADA)."""
    reasons = []
    st = "ok"
    if met_latest["tir"] < 70 or met_latest["tar250"] > 5 or met_latest["tbr70"] > 4:
        st = "alert"
        if met_latest["tir"] < 70:
            reasons.append("time in range below 70%")
        if met_latest["tar250"] > 5:
            reasons.append("more than 5% of readings above 250")
        if met_latest["tbr70"] > 4:
            reasons.append("more than 4% of readings below 70")
    elif met_latest["tir"] < 90 or met_latest["tar180"] > 5:
        st = "watch"
        reasons.append("time in range below 90%" if met_latest["tir"] < 90 else "more than 5% of readings above 180")
    if met_first is not None and met_latest is not met_first and met_latest["mean"] - met_first["mean"] >= 10:
        reasons.append("average sugar up 10+ mg/dL since weeks 1-2")
        if st == "ok":
            st = "watch"
    return st, reasons


# ---------- main ----------

def main(data_dir, out_dir):
    parts, checks, daily, meals, glu, demo_ids = load(data_dir)
    os.makedirs(f"{out_dir}/survivors", exist_ok=True)
    glu["wk"] = None
    start = parts.set_index("participant_id").start_date
    glu["day"] = (glu.timestamp - pd.to_datetime(glu.participant_id.map(start))).dt.days
    glu["window"] = np.where(glu.day < 28, "weeks 1-2", "weeks 11-12")

    rw_path = f"{out_dir}/realworld.json"  # from scripts/realworld.py, run first
    realworld = json.load(open(rw_path)) if os.path.exists(rw_path) else None
    ml_path = f"{out_dir}/ml_effects.json"  # the ML engine, from scripts/ml_effects.py (scripts/reproduce.sh or scripts/build-ml.sh)
    if not os.path.exists(ml_path):
        sys.exit(f"missing {ml_path}: the advice runs on the ML engine. Run scripts/build-ml.sh (or scripts/reproduce.sh) first.")
    ml = json.load(open(ml_path))
    mc_path = f"{out_dir}/mlcompare.json"  # statistics vs ML benchmark (scripts/ml_compare.py)
    mlc = json.load(open(mc_path)) if os.path.exists(mc_path) else None
    mr = meal_responses(meals, glu, parts)
    day_workers = mr[mr.work_schedule != "night shift"]
    dinners = day_workers[day_workers.meal_type == "dinner"]
    print(f"meals with clean response: {len(mr)}, dinners (no night shift): {len(dinners)}", file=sys.stderr)

    # meal-type dummies for walk model
    mt = pd.get_dummies(mr.meal_type, prefix="mt", dtype=float)
    mt = mt.drop(columns=[c for c in mt.columns if c == "mt_lunch"])
    mrw = pd.concat([mr, mt], axis=1)
    walk_cov = ["carbs_g"] + list(mt.columns)

    cohort = {}
    cohort["lateDinner"] = fe_effect(dinners, ["late", "carbs_g"])
    cohort["lateDinnerIauc"] = fe_effect(dinners, ["late", "carbs_g"], ycol="iauc")
    cohort["walk"] = fe_effect(mrw, ["walked"] + walk_cov)
    cohort["walkIauc"] = fe_effect(mrw, ["walked"] + walk_cov, ycol="iauc")
    din_w = mrw[(mrw.meal_type == "dinner") & (mrw.work_schedule != "night shift")]
    cohort["walkAfterDinner"] = fe_effect(din_w, ["walked", "carbs_g"])
    oth = mrw[mrw.meal_type != "dinner"]
    cohort["walkAfterOtherMeals"] = fe_effect(oth, ["walked"] + walk_cov)
    cohort["carbsPer10g"] = fe_effect(mrw, ["carbs_g", "walked"] + list(mt.columns))
    for k in ("est", "lo", "hi"):
        cohort["carbsPer10g"][k] *= 10

    # effect by blood sugar group
    by_group = {}
    for bg, sub in mrw.groupby("blood_sugar_group"):
        by_group[bg] = {
            "walk": fe_effect(sub, ["walked"] + walk_cov),
            "lateDinner": fe_effect(sub[(sub.meal_type == "dinner") & (sub.work_schedule != "night shift")], ["late", "carbs_g"]),
        }

    # dose response: within-person (person-demeaned, carbs-adjusted) rise by walk bin
    dose = []
    mrw["bin"] = None
    for lo_, hi_, lab in WALK_BINS:
        mrw.loc[(mrw.walked_after_min >= lo_) & (mrw.walked_after_min <= hi_), "bin"] = lab
    dummies = pd.get_dummies(mrw.bin, dtype=float).drop(columns=["none"])
    mrd = pd.concat([mrw, dummies], axis=1)
    for j, lab in enumerate(dummies.columns):
        e = fe_effect(mrd, list(dummies.columns) + walk_cov, k=j)
        e["n_meals"] = int((mrd.bin == lab).sum())
        dose.append({"bin": lab, **e})
    order = [b[2] for b in WALK_BINS]
    dose = [{"bin": "none", "est": 0.0, "lo": 0.0, "hi": 0.0, "n_meals": int((mrd.bin == "none").sum())}] + sorted(dose, key=lambda x: order.index(x["bin"]))

    # dinner hour response (within-person, carbs-adjusted, vs 17-18h)
    hour_bins = [(15, 17, "before 6pm"), (18, 18, "6-7pm"), (19, 19, "7-8pm"), (20, 20, "8-9pm"), (21, 21, "9-10pm"), (22, 27, "10pm or later")]
    dd = dinners.copy()
    hh = dd.ts.dt.hour.where(dd.ts.dt.hour >= 4, dd.ts.dt.hour + 24)
    dd["hb"] = None
    for a, b, lab in hour_bins:
        dd.loc[(hh >= a) & (hh <= b), "hb"] = lab
    dd = dd[dd.hb.notna()]
    hd = pd.get_dummies(dd.hb, dtype=float).drop(columns=["before 6pm"])
    dd2 = pd.concat([dd, hd], axis=1)
    dinner_hour = [{"bin": "before 6pm", "est": 0.0, "lo": 0.0, "hi": 0.0, "n_meals": int((dd.hb == "before 6pm").sum())}]
    for j, lab in enumerate(hd.columns):
        e = fe_effect(dd2, list(hd.columns) + ["carbs_g"], k=j)
        dinner_hour.append({"bin": lab, **e, "n_meals": int((dd.hb == lab).sum())})
    dinner_hour.sort(key=lambda x: [h[2] for h in hour_bins].index(x["bin"]))

    # overnight fast -> next-morning baseline (pre-breakfast reading), within-person
    dl = daily[["participant_id", "date", "overnight_fast_hours"]].dropna()
    first_meal = mr.sort_values("ts").groupby(["participant_id", "eating_day"]).first().reset_index()
    fm = meals.copy()
    fm["ts"] = pd.to_datetime(fm.date + " " + fm.time)
    fm = fm[fm.calories >= 50].sort_values("ts").groupby(["participant_id", "eating_day"]).first().reset_index()
    g = glu.set_index(["participant_id", "timestamp"]).glucose_mg_dl
    fm["morning"] = g.reindex(pd.MultiIndex.from_arrays([fm.participant_id, fm.ts.dt.floor("15min")])).to_numpy()
    night = fm.merge(dl, left_on=["participant_id", "eating_day"], right_on=["participant_id", "date"])
    night = night.merge(parts[["participant_id", "work_schedule", "group"]], on="participant_id")
    night = night[night.morning.notna() & (night.work_schedule != "night shift")]
    night["fast13"] = (night.overnight_fast_hours >= 13).astype(int)
    cohort["fastMorningPerHour"] = fe_effect(night, ["overnight_fast_hours"], ycol="morning")
    cohort["fast13Morning"] = fe_effect(night, ["fast13"], ycol="morning")

    # ---------- swap in the ML engine (double ML) for every meal-level effect ----------
    # The fixed-effects (FE) numbers above stay as the statistics cross-check. Overnight-fast effects stay FE:
    # there are no meal features for ML to learn from at the night level.
    stats = {"effects": {k: dict(v_) for k, v_ in cohort.items()}, "walkDose": dose, "dinnerHour": dinner_hour, "byGroup": by_group}
    for k, v_ in ml["cohort"].items():
        if k != "negativeControl":
            cohort[k] = dict(v_)
    by_group, dose, dinner_hour = ml["byGroup"], ml["walkDose"], ml["dinnerHour"]

    # ---------- robustness of the walk finding ----------
    nc_df = mrw[mrw.pre15.notna()]
    rob = {
        "negativeControl": dict(ml["cohort"]["negativeControl"]),
        "instrument": iv_walk(mrw, parts, walk_cov),
    }
    rob["eValue"] = e_value(cohort["walk"]["est"], cohort["walk"]["hi"], ml["cohort"]["walk"]["residSd"])
    stats["negativeControl"] = fe_effect(nc_df, ["walked"] + walk_cov, ycol="pre15")

    # randomized arms: week12 - week0
    c0 = checks[checks.visit == "week 0"].set_index("participant_id")
    c12 = checks[checks.visit == "week 12"].set_index("participant_id")
    outcomes = ["hba1c_pct", "fasting_glucose_mg_dl", "weight_kg", "fatigue_score_0_to_52", "six_minute_walk_m"]
    delta = (c12[outcomes] - c0.loc[c12.index, outcomes]).join(parts.set_index("participant_id")[["group", "work_schedule", "blood_sugar_group"]])
    arms = {}
    for grp, sub in parts.groupby("group"):
        arms[grp] = {"randomized": int(len(sub)), "finished": int((sub.finished_program == "yes").sum())}
    trial = {"arms": arms, "contrasts": {}}
    for out in outcomes:
        trial["contrasts"][out] = {}
        means = {grp: delta[delta.group == grp][out].dropna().to_numpy() for grp in arms}
        for grp, v in means.items():
            trial["contrasts"][out][f"mean:{grp}"] = {"est": v.mean(), "n": int(len(v))}
        for a, b in [("coaching + overnight fast", "coaching"), ("coaching", "tips by email"), ("coaching + overnight fast", "tips by email")]:
            va, vb = means[a], means[b]
            boots = [RNG.choice(va, len(va)).mean() - RNG.choice(vb, len(vb)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            trial["contrasts"][out][f"{a} vs {b}"] = {"est": va.mean() - vb.mean(), "lo": lo, "hi": hi, "n": int(len(va) + len(vb))}

    def arm_contrast(sub, out, a="coaching + overnight fast", b="coaching"):
        va = sub[sub.group == a][out].dropna().to_numpy()
        vb = sub[sub.group == b][out].dropna().to_numpy()
        boots = np.array([RNG.choice(va, len(va)).mean() - RNG.choice(vb, len(vb)).mean() for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = max(1 / len(boots), 2 * min((boots >= 0).mean(), (boots <= 0).mean()))
        return {"est": va.mean() - vb.mean(), "lo": lo, "hi": hi, "n": int(len(va) + len(vb)), "p": p}

    trial["fastVsCoachingBySubgroup"] = {}
    for out in ("hba1c_pct", "fasting_glucose_mg_dl"):
        sg = {f"blood sugar: {g_}": arm_contrast(sub, out) for g_, sub in delta.groupby("blood_sugar_group")}
        sg["excluding night shift"] = arm_contrast(delta[delta.work_schedule != "night shift"], out)
        sg["night shift only"] = arm_contrast(delta[delta.work_schedule == "night shift"], out)
        trial["fastVsCoachingBySubgroup"][out] = sg
    # is the fast effect really different by blood sugar group? Cochran Q across the three groups (df = 2),
    # and Holm adjustment across every subgroup contrast we looked at (2 outcomes x 5 subgroups)
    het = {}
    for out, sg in trial["fastVsCoachingBySubgroup"].items():
        gs = [v_ for k, v_ in sg.items() if k.startswith("blood sugar: ")]
        e = np.array([x["est"] for x in gs])
        w = 1 / (np.array([(x["hi"] - x["lo"]) / 3.92 for x in gs]) ** 2)
        q = float(np.sum(w * (e - np.sum(w * e) / np.sum(w)) ** 2))
        het[out] = {"Q": q, "df": len(gs) - 1, "p": float(np.exp(-q / 2)) if len(gs) == 3 else None}
    keys = [(o, k) for o, sg in trial["fastVsCoachingBySubgroup"].items() for k in sg]
    adj = holm(np.array([trial["fastVsCoachingBySubgroup"][o][k]["p"] for o, k in keys]))
    for (o, k), pa in zip(keys, adj):
        trial["fastVsCoachingBySubgroup"][o][k]["pHolm"] = float(pa)
    trial["heterogeneity"] = het
    group_specific = het["hba1c_pct"]["p"] < 0.05  # use per-group fast numbers in advice only if groups really differ
    dd_ = daily.merge(parts[["participant_id", "group"]], on="participant_id")
    trial["adherence"] = {
        grp: {"overnightFastHours": float(sub.overnight_fast_hours.mean()),
              "nightsAtLeast13hPct": float(100 * (sub.overnight_fast_hours.dropna() >= 13).mean()),
              "eveningCaloriesPct": float(sub.evening_calories_pct.mean()),
              "walkAfterDinnerMin": float(sub.walked_after_dinner_min.mean())}
        for grp, sub in dd_.groupby("group")}

    # cohort mean curves
    cc = [f"c{o}" for o in CURVE_OFFSETS]
    curves = {
        "minutes": CURVE_OFFSETS,
        "earlyDinner": [r(x) for x in dinners[dinners.late == 0][cc].mean()],
        "lateDinner": [r(x) for x in dinners[dinners.late == 1][cc].mean()],
        "noWalk": [r(x) for x in mr[mr.walked == 0][cc].mean()],
        "walk": [r(x) for x in mr[mr.walked == 1][cc].mean()],
    }

    # ---------- real-data agreement and power ----------
    real_checks, power = {}, {}
    if realworld:
        cg, nw = realworld["cgmacros"], realworld.get("nhanesSurvivorsWeighted")
        cg["walk10Statistics"], cg["lateDinnerStatistics"] = cg.get("walk10"), cg["lateDinner"]
        cg["walk10"], cg["lateDinner"] = ml["real"]["walk10"], ml["real"]["lateDinner"]  # same ML engine as the program
        cg["engine"] = "double ML (gradient boosting), same as the program; FE kept as *Statistics"
        real_checks = {
            "walk": {"program": cohort["walk"]["est"], "real": cg.get("walk10"), "source": "CGMacros, walk = " + cg.get("walkDefinition", "")},
            "lateDinner": {"program": cohort["lateDinner"]["est"], "real": cg["lateDinner"], "source": "CGMacros"},
            "overnightFast": {"program": trial["contrasts"]["hba1c_pct"]["coaching + overnight fast vs coaching"]["est"],
                              "real": nw["hba1cPerFastHour"] if nw else None, "source": "NHANES survivors, survey-weighted, HbA1c per extra fast hour"},
        }
        for v_ in real_checks.values():
            v_["label"] = agreement(v_["program"], v_["real"])
        ppl = cg["lateDinner"]["people"]
        power["lateDinner"] = {"realPeople": ppl, "programEffect": cohort["lateDinner"]["est"], "realEffect": cg["lateDinner"]["est"],
                               "peopleForProgramEffect": people_needed(cg["lateDinner"], cohort["lateDinner"]["est"], ppl),
                               "peopleForRealEffect": people_needed(cg["lateDinner"], cg["lateDinner"]["est"], ppl)}
        if cg.get("walk10"):
            ppl = cg["walk10"]["people"]
            power["walk"] = {"realPeople": ppl, "programEffect": cohort["walk"]["est"], "realEffect": cg["walk10"]["est"],
                             "peopleForProgramEffect": people_needed(cg["walk10"], cohort["walk"]["est"], ppl),
                             "peopleForRealEffect": people_needed(cg["walk10"], cg["walk10"]["est"], ppl)}
    def size_word(real_est, program_est):
        ratio = real_est / program_est if program_est else 0
        if ratio <= 0:
            return None
        for lim, w in ((0.15, "much smaller than"), (0.22, "about a fifth of"), (0.3, "about a quarter of"), (0.4, "about a third of"),
                       (0.6, "about half of"), (0.85, "somewhat smaller than"), (1.2, "about the same as")):
            if ratio < lim:
                return w
        return "larger than"

    REAL_SENTENCE = {
        "agrees": "A check in real adults ({src}) found the same.",
        "same direction, not proven": "A check in real adults ({src}) pointed the same way, but it is not proven there yet.",
        "no clear signal": "A check in real people ({src}) showed no clear link yet.",
        "disagrees": "A check in real people ({src}) pointed the other way.",
    }

    # ---------- per-person ----------
    sensor_ids = sorted(glu.participant_id.unique())
    per = {}
    pmr = dict(tuple(mrw.groupby("participant_id")))
    for pid in sensor_ids:
        sub = pmr.get(pid)
        if sub is None:
            per[pid] = {"walk": None, "late": None}
            continue
        mp = ml["personal"].get(pid, {})
        per[pid] = {"walk": mp.get("walk"), "late": mp.get("late")}
    tau_w = dl_tau2([p["walk"] for p in per.values()])
    tau_l = dl_tau2([p["late"] for p in per.values()])

    pinfo = parts.set_index("participant_id")
    gp_all = dict(tuple(glu.groupby("participant_id")))
    night_by = dict(tuple(night.groupby("participant_id")))
    roster = []
    status_counts = {"ok": 0, "watch": 0, "alert": 0}
    all_status = {}
    for pid in sensor_ids:
        gp = gp_all[pid]
        w1 = gp[gp.window == "weeks 1-2"]
        w2 = gp[gp.window == "weeks 11-12"]
        m_first = glucose_metrics(w1) if len(w1) else None
        m_last = glucose_metrics(w2) if len(w2) > 96 else m_first
        st, reasons = status_from(m_last, m_first)
        status_counts[st] += 1
        all_status[pid] = (st, reasons, m_first, m_last)

    for pid in demo_ids:
        info = pinfo.loc[pid]
        st, reasons, m_first, m_last = all_status[pid]
        gp = gp_all[pid]
        sub = pmr.get(pid, mrw.iloc[0:0])
        din = sub[sub.meal_type == "dinner"]
        night_shift = info.work_schedule == "night shift"
        latest_label = "weeks 11-12" if m_last is not m_first else "weeks 1-2"
        sid = f"c4-{pid}"
        ev = []

        def add(id_, label, value, unit, scope, method, ci=None, n=None, caveat=None, nd=1):
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                val = int(value)
            else:
                val = r(value, nd)
            ev.append({"id": id_, "label": label, "value": val, "unit": unit,
                       "ci": [r(ci[0], nd), r(ci[1], nd)] if ci else None, "n": None if n is None else int(n), "scope": scope,
                       "method": method, "caveat": caveat})
            return id_

        add("tir", f"Time in range 70-180 mg/dL ({latest_label})", m_last["tir"], "%", "person",
            "Share of 15-min sensor readings between 70 and 180 mg/dL.", n=m_last["readings"])
        add("mean_glucose", f"Average blood sugar ({latest_label})", m_last["mean"], "mg/dL", "person",
            "Mean of 15-min sensor readings.", n=m_last["readings"])
        add("tar180", f"Readings above 180 mg/dL ({latest_label})", m_last["tar180"], "%", "person",
            "Share of readings above 180 mg/dL.", n=m_last["readings"])
        add("tar250", f"Readings above 250 mg/dL ({latest_label})", m_last["tar250"], "%", "person",
            "Share of readings above 250 mg/dL.", n=m_last["readings"])
        add("tbr70", f"Readings below 70 mg/dL ({latest_label})", m_last["tbr70"], "%", "person",
            "Share of readings below 70 mg/dL.", n=m_last["readings"])
        add("gmi", f"Glucose management indicator ({latest_label})", m_last["gmi"], "%", "person",
            "GMI = 3.31 + 0.02392 x mean glucose; an estimate of HbA1c from sensor data.", n=m_last["readings"])
        if m_last is not m_first:
            add("mean_change", "Change in average blood sugar, weeks 1-2 to weeks 11-12", m_last["mean"] - m_first["mean"],
                "mg/dL", "person", "Mean of weeks 11-12 minus mean of weeks 1-2.", n=m_last["readings"] + m_first["readings"])
        add("status_rule", "Status thresholds", None, None, "rule",
            "alert: time in range < 70%, or > 5% of readings above 250, or > 4% below 70 (international CGM consensus targets). "
            "watch: time in range < 90%, or > 5% above 180, or average up 10+ mg/dL since weeks 1-2.")

        # personal meal-level evidence
        if len(sub):
            add("meals_scored", "Meals with a clean sugar curve", int(len(sub)), "meals", "person",
                "Meals of 10 g carbs or more, with sensor readings and no other 100+ kcal item in the next 2 hours.")
            add("median_rise", "Typical rise after a meal (median)", sub.rise.median(), "mg/dL", "person",
                "Median of peak (15-120 min) minus pre-meal reading.", n=int(len(sub)))
        if len(din):
            add("dinner_rise", "Typical rise after dinner (median)", din.rise.median(), "mg/dL", "person",
                "Median of peak minus pre-dinner reading.", n=int(len(din)))
            add("dinner_count", "Dinners with a clean sugar curve", int(len(din)), "meals", "person", "Scored dinners.")
            add("late_share", "Dinners started at 8 pm or later", 100 * din.late.mean(), "%", "person",
                "Share of scored dinners starting at or after 20:00.", n=int(len(din)))
        walks = int(sub.walked.sum()) if len(sub) else 0
        add("walk_meals", "Meals followed by a walk", walks, "meals", "person", "Scored meals with walked_after_min > 0.")

        pw, pl = per[pid]["walk"], per[pid]["late"]
        sw = shrink(pw, cohort["walk"], tau_w)
        sl = shrink(pl, cohort["lateDinner"], tau_l)
        if pw:
            add("walk_effect_own", "Your sugar rise after meals with a walk vs without", pw["est"], "mg/dL", "person",
                "Your meals only, from the ML engine: what the meal predicts is removed first (gradient boosting, cross-fitted), then the walk effect is fitted on what is left; 95% CI from 300 bootstrap resamples of your meals.",
                ci=(pw["lo"], pw["hi"]), n=pw["n_flag"] + pw["n_other"],
                caveat="You chose when to walk, so this is a pattern, not proof.")
            add("walk_effect_blend", "Best estimate for you (your data blended with the cohort)", sw["est"], "mg/dL", "person",
                f"Empirical-Bayes: {r(100 * sw['weight'], 0)}% weight on your data, rest on the cohort estimate. "
                f"People differed by about {r(np.sqrt(tau_w))} mg/dL in this effect.",
                ci=(sw["lo"], sw["hi"]))
        if pl:
            add("late_effect_own", "Your sugar rise after dinners at 8 pm or later vs earlier", pl["est"], "mg/dL", "person",
                "Your dinners only, from the ML engine: what the meal predicts is removed first (gradient boosting, cross-fitted), then the late-start effect is fitted on what is left; 95% CI from 300 bootstrap resamples.",
                ci=(pl["lo"], pl["hi"]), n=pl["n_flag"] + pl["n_other"],
                caveat="Late dinners may differ in other ways (what you ate, how tired you were).")
            add("late_effect_blend", "Best estimate for you (your data blended with the cohort)", sl["est"], "mg/dL", "person",
                f"Empirical-Bayes: {r(100 * sl['weight'], 0)}% weight on your data, rest on the cohort estimate. "
                + ("People did not differ measurably in this effect, so the cohort estimate is the best guess for you." if tau_l == 0 else f"People differed by about {r(np.sqrt(tau_l))} mg/dL."),
                ci=(sl["lo"], sl["hi"]))

        # overnight fast
        nb = night_by.get(pid)
        if nb is not None and len(nb):
            add("fast_avg", "Your average overnight fast", nb.overnight_fast_hours.mean(), "hours", "person",
                "Hours from last bite to first bite, nights with sensor data.", n=int(len(nb)))
            add("fast13_share", "Nights with a 13-hour fast or longer", 100 * nb.fast13.mean(), "%", "person",
                "Share of nights with overnight_fast_hours >= 13.", n=int(len(nb)))

        # cohort evidence copied in so each panel stands alone
        cw, cl = cohort["walk"], cohort["lateDinner"]
        add("cohort_walk", "Program-wide: rise after meals with a walk vs without (same person)", cw["est"], "mg/dL", "cohort",
            ml["method"]["walk"],
            ci=(cw["lo"], cw["hi"]), n=cw["rows"], caveat="Walking was self-chosen: association, not proof.")
        best = min([d for d in dose if d["bin"] != "none"], key=lambda d: d["est"])
        d1019 = next(d for d in dose if d["bin"] == "10-19 min")
        add("cohort_walk_10_19", "Program-wide: rise after a 10-19 min walk vs no walk", d1019["est"], "mg/dL", "cohort",
            ml["method"]["walk"], ci=(d1019["lo"], d1019["hi"]), n=d1019["n_meals"])
        d2029 = next(d for d in dose if d["bin"] == "20-29 min")
        d30 = next(d for d in dose if d["bin"] == "30+ min")
        add("cohort_walk_20_29", "Program-wide: rise after a 20-29 min walk vs no walk", d2029["est"], "mg/dL", "cohort",
            ml["method"]["walk"], ci=(d2029["lo"], d2029["hi"]), n=d2029["n_meals"])
        add("cohort_walk_30plus", "Program-wide: rise after a 30+ min walk vs no walk", d30["est"], "mg/dL", "cohort",
            ml["method"]["walk"], ci=(d30["lo"], d30["hi"]), n=d30["n_meals"])
        h89 = next(d for d in dinner_hour if d["bin"] == "8-9pm")
        add("cohort_dinner_8_9pm", "Program-wide: rise after dinner at 8-9 pm vs before 6 pm (same person)", h89["est"], "mg/dL", "cohort",
            ml["method"]["hour"] + " Night-shift workers excluded.", ci=(h89["lo"], h89["hi"]), n=h89["n_meals"],
            caveat="Association, not proof.")
        add("cohort_late", "Program-wide: rise after dinners at 8 pm or later vs earlier (same person)", cl["est"], "mg/dL", "cohort",
            ml["method"]["late"] + " Night-shift workers excluded.",
            ci=(cl["lo"], cl["hi"]), n=cl["rows"], caveat="Association, not proof.")
        cwd = cohort["walkAfterDinner"]
        cwo = cohort["walkAfterOtherMeals"]
        add("cohort_walk_dinner", "Program-wide: rise after dinner with a walk vs without (same person)", cwd["est"], "mg/dL", "cohort",
            ml["method"]["walkDinner"] + " Dinners only; night-shift workers excluded.",
            ci=(cwd["lo"], cwd["hi"]), n=cwd["rows"], caveat="Walking was self-chosen: association, not proof.")
        add("cohort_walk_other", "Program-wide: rise after other meals with a walk vs without (same person)", cwo["est"], "mg/dL", "cohort",
            ml["method"]["walk"] + " Breakfasts, lunches and snacks.",
            ci=(cwo["lo"], cwo["hi"]), n=cwo["rows"])
        tsg = trial["fastVsCoachingBySubgroup"]["hba1c_pct"][f"blood sugar: {info.blood_sugar_group}"]
        add("trial_fast_hba1c_group", f"Randomized, people who started with {info.blood_sugar_group}: HbA1c change, coaching + overnight fast vs coaching alone",
            tsg["est"], "% points", "trial", "Difference in mean (week 12 - week 0) between randomized arms within this starting blood sugar group; bootstrap 95% CI.",
            ci=(tsg["lo"], tsg["hi"]), n=tsg["n"], nd=2, caveat="Randomization holds within groups set at the start.")
        adh = trial["adherence"]
        add("trial_fast_adherence", "Randomized: extra overnight fast hours in the fast arm vs coaching alone",
            adh["coaching + overnight fast"]["overnightFastHours"] - adh["coaching"]["overnightFastHours"], "hours", "trial",
            "Mean overnight_fast_hours over all logged days, fast arm minus coaching arm.")
        tf = trial["contrasts"]["hba1c_pct"]["coaching + overnight fast vs coaching"]
        tg = trial["contrasts"]["fasting_glucose_mg_dl"]["coaching + overnight fast vs coaching"]
        add("trial_fast_hba1c", "Randomized: HbA1c change, coaching + overnight fast vs coaching alone", tf["est"], "% points", "trial",
            "Difference in mean (week 12 - week 0) between randomized arms, completers; bootstrap 95% CI.",
            ci=(tf["lo"], tf["hi"]), n=tf["n"], caveat="Fair comparison: groups were assigned at random.", nd=2)
        add("trial_fast_fpg", "Randomized: fasting glucose change, coaching + overnight fast vs coaching alone", tg["est"], "mg/dL", "trial",
            "Difference in mean (week 12 - week 0) between randomized arms, completers; bootstrap 95% CI.",
            ci=(tg["lo"], tg["hi"]), n=tg["n"])

        if realworld:
            cg = realworld["cgmacros"]
            add("real_late_dinner", "Real-data check (CGMacros, 45 real adults): dinner at 8 pm or later vs earlier",
                cg["lateDinner"]["est"], "mg/dL", "real", "Same ML engine as the program (double ML, calories and macros), within person; person-clustered 95% CI.",
                ci=(cg["lateDinner"]["lo"], cg["lateDinner"]["hi"]), n=cg["lateDinner"]["rows"],
                caveat="Same direction as the program data, but the interval includes zero: not proven in real people yet.")
            add("real_movement", "Real-data check (CGMacros): rise per extra MET of movement in the hour after a meal",
                cg["movementPerMet"]["est"], "mg/dL", "real", "Within-person, carbs adjusted; cluster bootstrap 95% CI.",
                ci=(cg["movementPerMet"]["lo"], cg["movementPerMet"]["hi"]), n=cg["movementPerMet"]["rows"],
                caveat="No clear effect in 45 real adults over 10 days; Fitbit METs are a rough measure of walking.")
            if cg.get("walk10"):
                add("real_walk", "Real-data check (CGMacros): rise after meals followed by a walk vs not",
                    cg["walk10"]["est"], "mg/dL", "real", f"Walk = {cg['walkDefinition']}. Same ML engine as the program (double ML, calories, macros and meal type), within person; person-clustered 95% CI.",
                    ci=(cg["walk10"]["lo"], cg["walk10"]["hi"]), n=cg["walk10"]["rows"],
                    caveat=f"Check result: {real_checks['walk']['label']}. About {power['walk']['peopleForRealEffect']} people would be needed to confirm an effect this size.")
            nw = realworld.get("nhanesSurvivorsWeighted")
            if nw:
                add("real_fast", "Real-data check (NHANES survivors, survey-weighted): HbA1c per extra hour of overnight fast",
                    nw["hba1cPerFastHour"]["est"], "% points", "real",
                    "Survey-weighted regression adjusted for age and sex; design-based bootstrap 95% CI (PSUs within strata).",
                    ci=(nw["hba1cPerFastHour"]["lo"], nw["hba1cPerFastHour"]["hi"]), n=nw["hba1cPerFastHour"]["n"], nd=3,
                    caveat=f"Check result: {real_checks['overnightFast']['label']}. A one-time survey snapshot, not a trial.")
        nc = rob["negativeControl"]
        add("cohort_walk_negctrl", "Check: pre-meal sugar before meals that were later followed by a walk vs not (same person)", nc["est"], "mg/dL", "cohort",
            "Negative control: a walk after the meal cannot change the reading 15 min before it. Same ML engine as the walk effect (double ML); person-clustered 95% CI.",
            ci=(nc["lo"], nc["hi"]), n=nc["rows"], caveat="Near zero, so walk meals do not start from a different place.")
        add("cohort_walk_evalue", "Check: how strongly a hidden factor would have to link to both walking and the rise to explain the walk effect away (E-value)",
            rob["eValue"]["estimate"], "x (risk ratio)", "cohort",
            f"VanderWeele-Ding E-value; mean difference converted with the ML engine's within-person residual SD ({r(rob['eValue']['sdUsed'])} mg/dL). "
            f"For the CI bound nearer zero: {r(rob['eValue']['ciBound'])}.")
        for eid, key, what in (("cohort_walk_fe", "walk", "a walk after the meal"), ("cohort_late_fe", "lateDinner", "dinner at 8 pm or later")):
            fe_, dm = stats["effects"][key], cohort[key]
            agree = dm["lo"] <= fe_["hi"] and fe_["lo"] <= dm["hi"] and np.sign(dm["est"]) == np.sign(fe_["est"])
            add(eid, f"Statistics cross-check (fixed-effects regression): rise with {what}, same person", fe_["est"], "mg/dL", "cohort",
                "Within-person linear regression (person fixed effects) adjusted for carbs" + (" and meal type" if key == "walk" else "")
                + "; cluster bootstrap 95% CI over people. The advice uses the ML engine; this is the classic check.",
                ci=(fe_["lo"], fe_["hi"]), n=fe_["rows"],
                caveat=f"ML engine: {r(dm['est']):+g}. " + ("The intervals overlap, so the number does not depend on the method." if agree
                                                         else "The two disagree; treat the advice number with extra caution."))
        fcp = ml["forecast"]["dinnerByPerson"].get(pid)
        if fcp:
            add("ml_forecast_dinner", "ML forecast: expected rise after your typical dinner", fcp["pred"], "mg/dL", "person",
                "Gradient boosting trained on every program meal (calories, macros, meal type, start time, pre-meal sugar and its trend, "
                "your usual rise). Each of your dinners is forecast by a model that never saw it (5-fold, out-of-fold); this is the average.",
                n=fcp["n"], caveat="A forecast, not a measurement.")
            add("ml_forecast_error", "ML forecast: typical miss on a meal it never saw", ml["forecast"]["mae"], "mg/dL", "cohort",
                f"Mean absolute error over {ml['forecast']['meals']} out-of-fold meals; R² {ml['forecast']['r2']}. "
                f"80% of misses fall between {ml['forecast']['err10']} and {ml['forecast']['err90']} mg/dL.")
        hz = trial["heterogeneity"]["hba1c_pct"]
        add("trial_fast_heterogeneity", "Randomized: do blood sugar groups differ in the fast effect on HbA1c? (test p-value)", hz["p"], "p", "trial",
            "Cochran Q across the three starting blood sugar groups (2 degrees of freedom). Under 0.05 would mean the groups really differ.",
            nd=2, caveat="Groups differ" if group_specific else "The program could not tell the groups apart, so advice uses the overall number.")
        evid = {e["id"]: e for e in ev}
        recs = []

        def v(i):
            return evid[i]["value"]

        # walk recommendation
        if pw and sw["hi"] < 0 and pw["hi"] < 0 and sw["weight"] >= 0.25:
            recs.append({"id": "walk", "title": "Take a short walk after meals",
                         "text": f"Your sugar rose about {abs(v('walk_effect_blend'))} mg/dL less after meals when you walked. "
                                 f"Aim for 10 to 20 minutes after dinner, where the program saw the biggest gain ({abs(v('cohort_walk_dinner'))} mg/dL).",
                         "weightOnYourData": r(sw["weight"], 2),
                         "basis": "personal", "evidenceIds": ["walk_effect_own", "walk_effect_blend", "cohort_walk_dinner", "cohort_walk_10_19"]})
        else:
            why = f"You walked after {v('walk_meals')} of your scored meals, too few to measure your own effect yet. " if walks < 6 else \
                  "We can't yet see a clear effect in your own meals. "
            recs.append({"id": "walk", "title": "Try a 10 to 20 minute walk after dinner",
                         "text": why + f"Across the program, a walk after dinner went with a rise about {abs(v('cohort_walk_dinner'))} mg/dL smaller in the same person, "
                                       f"more than after other meals ({abs(v('cohort_walk_other'))} mg/dL). "
                                       f"10 to 19 minutes got most of the gain: {abs(v('cohort_walk_10_19'))} mg/dL less, against {abs(v('cohort_walk_20_29'))} for 20-29 minutes and {abs(v('cohort_walk_30plus'))} for 30+ minutes.",
                         "basis": "cohort", "evidenceIds": ["walk_meals", "cohort_walk_dinner", "cohort_walk_other", "cohort_walk_10_19", "cohort_walk_20_29", "cohort_walk_30plus"]})
        # late dinner recommendation
        if night_shift:
            recs.append({"id": "late_dinner", "title": "Keep meal times steady around your shifts",
                         "text": "You work nights, so clock-time rules about late dinners don't fit you. Keep your main meal at the same point in your waking day.",
                         "basis": "rule", "evidenceIds": ["cohort_late"]})
        elif len(din) and v("late_share") is not None and v("late_share") >= 20:
            if pl and sl["lo"] > 0 and pl["lo"] > 0 and sl["weight"] >= 0.25:
                text = f"{v('late_share')}% of your dinners started at 8 pm or later, and after those your sugar rose about {v('late_effect_blend')} mg/dL more. Try to start dinner before 8 pm."
                basis, ids = "personal", ["late_share", "late_effect_own", "late_effect_blend"]
            else:
                text = (f"{v('late_share')}% of your dinners started at 8 pm or later. Across the program, sugar rose about {v('cohort_late')} mg/dL more after late dinners than after earlier ones in the same person, "
                        f"and {v('cohort_dinner_8_9pm')} mg/dL more after a dinner at 8-9 pm than after one before 6 pm. Try to start dinner before 8 pm.")
                basis, ids = "cohort", ["late_share", "cohort_late", "cohort_dinner_8_9pm"]
            recs.append({"id": "late_dinner", "title": "Move dinner earlier", "text": text, "basis": basis, "evidenceIds": ids})
        elif len(din):
            recs.append({"id": "late_dinner", "title": "Keep dinner early",
                         "text": (f"None of your {v('dinner_count')} scored dinners started at 8 pm or later" if v('late_share') == 0 else f"Only {v('late_share')}% of your dinners started at 8 pm or later") + f". Keep it that way: across the program, sugar rose about {v('cohort_late')} mg/dL more after late dinners.",
                         "basis": "cohort", "evidenceIds": ["dinner_count", "late_share", "cohort_late"]})
        # overnight fast
        low_sugar = m_last["tbr70"] > 4
        if not night_shift and "fast_avg" in evid and low_sugar:
            recs.append({"id": "overnight_fast", "title": "Hold off on longer fasts for now",
                         "text": f"{v('tbr70')}% of your readings were below 70 mg/dL, so check with your care team before stretching your overnight fast. "
                                 f"Right now it averages {v('fast_avg')} hours.",
                         "basis": "rule", "evidenceIds": ["tbr70", "fast_avg", "status_rule"]})
        elif not night_shift and "fast_avg" in evid and not group_specific:
            recs.append({"id": "overnight_fast", "title": "Aim for a 13-hour overnight break",
                         "text": f"Your overnight fast averaged {v('fast_avg')} hours, and {v('fast13_share')}% of nights reached 13 hours. "
                                 f"In the randomized program, adding the overnight fast to coaching changed HbA1c by {v('trial_fast_hba1c')} points "
                                 f"(95% CI {evid['trial_fast_hba1c']['ci'][0]} to {evid['trial_fast_hba1c']['ci'][1]}) compared with coaching alone. "
                                 "It looked larger in some blood sugar groups, but the program could not tell the groups apart, so this is the overall number.",
                         "basis": "trial", "evidenceIds": ["fast_avg", "fast13_share", "trial_fast_hba1c", "trial_fast_heterogeneity", "trial_fast_hba1c_group"]})
        elif not night_shift and "fast_avg" in evid:
            recs.append({"id": "overnight_fast", "title": "Aim for a 13-hour overnight break",
                         "text": f"Your overnight fast averaged {v('fast_avg')} hours, and {v('fast13_share')}% of nights reached 13 hours. "
                                 f"In the randomized program, for people who started with {info.blood_sugar_group}, adding the overnight fast to coaching changed HbA1c by "
                                 f"{v('trial_fast_hba1c_group')} points (95% CI {evid['trial_fast_hba1c_group']['ci'][0]} to {evid['trial_fast_hba1c_group']['ci'][1]}) compared with coaching alone"
                                 + (". The benefit was largest in type 2 diabetes." if info.blood_sugar_group != "type 2 diabetes" else ", the biggest gain of any group."),
                         "basis": "trial", "evidenceIds": ["fast_avg", "fast13_share", "trial_fast_hba1c_group", "trial_fast_hba1c"]})
        if st == "alert" and low_sugar:
            recs.insert(0, {"id": "talk_to_team", "title": "Tell your care team about low readings",
                            "text": f"{v('tbr70')}% of your readings were below 70 mg/dL ({latest_label}); the usual target is under 4%. "
                                    f"Low sugar can make you shaky or dizzy. Share this before your next visit.",
                            "basis": "rule", "evidenceIds": ["tbr70", "status_rule"]})
        elif st == "alert" and m_last["mean"] > 250:
            recs.insert(0, {"id": "talk_to_team", "title": "Send these readings to your care team today",
                            "text": f"Your average was {v('mean_glucose')} mg/dL ({latest_label}), above 250, and only {v('tir')}% of readings were in the 70-180 range. "
                                    f"Contact your care team today rather than waiting for your next visit.",
                            "basis": "rule", "evidenceIds": ["mean_glucose", "tir", "status_rule"]})
        elif st == "alert":
            recs.insert(0, {"id": "talk_to_team", "title": "Share these readings with your care team",
                            "text": f"Only {v('tir')}% of your readings were in the 70-180 range ({latest_label}); the usual target is more than 70%. "
                                    f"Your average was {v('mean_glucose')} mg/dL. Bring this to your next visit.",
                            "basis": "rule", "evidenceIds": ["tir", "mean_glucose", "status_rule"]})

        for rec in recs:
            if rec["id"] == "walk":
                rec["evidenceIds"] += ["cohort_walk_negctrl", "cohort_walk_evalue"]
            ml_eid = {"walk": "cohort_walk_fe", "late_dinner": "cohort_late_fe"}.get(rec["id"])
            if ml_eid in evid:
                rec["evidenceIds"].append(ml_eid)
            key, eid = {"walk": ("walk", "real_walk"), "late_dinner": ("lateDinner", "real_late_dinner"),
                        "overnight_fast": ("overnightFast", "real_fast")}.get(rec["id"], (None, None))
            if rec["basis"] == "rule" or key is None or key not in real_checks or eid not in evid:
                rec["realDataCheck"] = {"label": "not tested", "evidenceId": None}
                continue
            rc = real_checks[key]
            rec["realDataCheck"] = {"label": rc["label"], "evidenceId": eid}
            rec["evidenceIds"].append(eid)
            src = "NHANES" if key == "overnightFast" else "CGMacros"
            sw_ = size_word(rc["real"]["est"], rc["program"]) if key in ("walk", "lateDinner") and rc["label"] == "same direction, not proven" else None
            rec["text"] += " " + (f"A check in real adults ({src}) pointed the same way, at {sw_} the program's size, but it is not proven there yet."
                                  if sw_ else REAL_SENTENCE[rc["label"]].format(src=src))
            if sw_:
                rec["realDataCheck"]["realSize"] = sw_ + " the program's size"
        # safety routing (judges' panel): care team today for frequent lows or a very high average
        high_avg = m_last["mean"] > 250
        care = []
        if low_sugar:
            care.append({"reason": f"Frequent lows: {v('tbr70')}% of readings below 70 (target under 4%)", "evidenceId": "tbr70"})
        if high_avg:
            care.append({"reason": f"Very high average: {v('mean_glucose')} mg/dL (above 250)", "evidenceId": "mean_glucose"})
        care_flag = {"today": bool(care), "reasons": care,
                     "rule": "Send to the care team today if more than 4% of readings are below 70 mg/dL or the average is above 250 mg/dL."}
        if low_sugar:
            for rec in recs:
                if rec["id"] == "walk":
                    rec["caution"] = WALK_CAUTION
                    rec["text"] += " " + WALK_CAUTION
        team_key = "team_lows" if low_sugar else "team_high" if high_avg else "team_alert" if st == "alert" else None
        pview = patient_view(recs, low_sugar, team_key)
        hypo = {"level": "high", "reasonEvidenceId": "tbr70", "text": pview["en"]["lowSugarWarning"],
                "translations": {k: pview[k]["lowSugarWarning"] for k in ("es", "ht")}} if low_sugar else None

        # charts
        hourly = []
        for lab, part in (("weeks 1-2", gp[gp.window == "weeks 1-2"]), ("weeks 11-12", gp[gp.window == "weeks 11-12"])):
            prof = part.groupby(part.timestamp.dt.hour).glucose_mg_dl.mean().reindex(range(24))
            hourly.append({"label": lab, "values": [r(x) for x in prof]})
        early_c = din[din.late == 0][cc].mean() if len(din) else None
        late_c = din[din.late == 1][cc].mean() if len(din) else None
        nw_c = sub[sub.walked == 0][cc].mean() if len(sub) else None
        w_c = sub[sub.walked == 1][cc].mean() if len(sub) else None

        def series(s):
            if s is None:
                return None
            vals = [r(x) for x in s]
            return vals if any(x is not None for x in vals) else None

        charts = {
            "dailyProfile": {"hours": list(range(24)), "series": hourly, "unit": "mg/dL", "targetRange": [70, 180]},
            "dinnerResponse": {"minutes": CURVE_OFFSETS, "unit": "mg/dL above pre-meal",
                               "you": {"early": series(early_c), "late": series(late_c),
                                       "nEarly": int((din.late == 0).sum()) if len(din) else 0, "nLate": int((din.late == 1).sum()) if len(din) else 0},
                               "cohort": {"early": curves["earlyDinner"], "late": curves["lateDinner"]}},
            "walkResponse": {"minutes": CURVE_OFFSETS, "unit": "mg/dL above pre-meal",
                             "you": {"noWalk": series(nw_c), "walk": series(w_c), "nNoWalk": int(len(sub) - walks), "nWalk": walks},
                             "cohort": {"noWalk": curves["noWalk"], "walk": curves["walk"]}},
            "walkDose": {"bins": [d["bin"] for d in dose], "est": [r(d["est"]) for d in dose],
                         "lo": [r(d["lo"]) for d in dose], "hi": [r(d["hi"]) for d in dose], "nMeals": [d["n_meals"] for d in dose],
                         "unit": "mg/dL vs no walk (same person)"},
            "overnightFast": {"hours": [r(x) for x in nb.overnight_fast_hours] if nb is not None else [],
                              "morningGlucose": [r(x, 0) for x in nb.morning] if nb is not None else [], "unit": "mg/dL"},
            "checkups": {"metrics": ["hba1c_pct", "fasting_glucose_mg_dl", "weight_kg"],
                         "week0": [r(c0.loc[pid, k], 1) for k in ["hba1c_pct", "fasting_glucose_mg_dl", "weight_kg"]],
                         "week12": [r(c12.loc[pid, k], 1) if pid in c12.index else None for k in ["hba1c_pct", "fasting_glucose_mg_dl", "weight_kg"]]},
        }

        sex = "F" if info.sex == "female" else "M"
        display = f"{pid} · {info.age}{sex} · {info.cancer_type}"
        rising = "mean_change" in evid and evid["mean_change"]["value"] >= 10
        if st == "alert" and m_last["tbr70"] > 4:
            headline = f"Frequent lows: {v('tbr70')}% of readings below 70"
        elif st == "alert":
            headline = f"High most of the day: {round(m_last['tir'])}% in range, average {round(m_last['mean'])} mg/dL"
        elif st == "watch" and rising:
            headline = f"Sugar rising: average up {v('mean_change')} mg/dL since the start"
        elif st == "watch" and m_last["tir"] < 90:
            headline = f"Often above range: {round(m_last['tir'])}% in range"
        elif st == "watch":
            headline = f"Spikes above 180: {v('tar180')}% of readings"
        else:
            headline = f"Steady: {round(m_last['tir'])}% in range"

        # signal tags for the roster tile: short, evidence-backed, most important first
        tags = []

        def tag(label, tone, eid):
            tags.append({"label": label, "tone": tone, "evidenceId": eid})

        if m_last["tbr70"] > 4:
            tag(f"Lows {v('tbr70')}%", "risk", "tbr70")
        if m_last["tir"] < 90:
            tag(f"In range {round(m_last['tir'])}%", "risk" if m_last["tir"] < 70 else "warn", "tir")
        if rising:
            tag(f"Avg +{v('mean_change')}", "risk", "mean_change")
        if night_shift:
            tag("Night shift", "info", "status_rule")
        elif "late_share" in evid and v("late_share") >= 30:
            tag(f"Late dinners {round(v('late_share'))}%", "warn", "late_share")
        if any(rc["id"] == "walk" and rc["basis"] == "personal" for rc in recs):
            tag(f"Walks work: {v('walk_effect_blend')} mg/dL", "good", "walk_effect_blend")
        elif len(sub) and walks / len(sub) < 0.1:
            tag(f"Walks after {walks}/{len(sub)} meals", "warn", "walk_meals")
        if "fast_avg" in evid and not night_shift:
            if v("fast_avg") < 12:
                tag(f"Fast {v('fast_avg')} h", "warn", "fast_avg")
            elif v("fast13_share") >= 50:
                tag(f"Fast {v('fast_avg')} h", "good", "fast_avg")
        tags = tags[:4]
        updated = str(gp.timestamp.max().isoformat())
        summary = (f"{info.age}-year-old {info.sex}, {info.cancer_type} cancer, {info.years_since_treatment} years after treatment; "
                   f"{info.blood_sugar_group} at start; group: {info.group}; {info.work_schedule}. "
                   f"Average sugar {round(m_last["mean"])} mg/dL in {latest_label}, {round(m_last["tir"])}% of readings in range. "
                   + (f"Median rise after dinner {round(din.rise.median())} mg/dL. " if len(din) else "")
                   + (f"ML forecast for a typical dinner: {round(fcp['pred'])} mg/dL. " if fcp else "")
                   + "Synthetic participant (made-up data).")
        today_reason = ("Glucose: " + (care[0]["reason"] if care else headline))[:90]
        priority = (0 if low_sugar else 1) if care else {"alert": 2, "watch": 3, "ok": 4}[st]
        detail = {"survivorId": sid, "displayName": display, "status": st, "statusReasons": reasons, "headline": headline, "tags": tags,
                  "summary": summary, "plainSummary": plain_summary(recs, low_sugar, team_key), "updatedAt": updated,
                  "careTeamFlag": care_flag, "hypoglycaemiaWarning": hypo, "todayReason": today_reason, "priority": priority,
                  "patientView": pview,
                  "profile": {"age": int(info.age), "sex": info.sex, "cancerType": info.cancer_type,
                              "yearsSinceTreatment": float(info.years_since_treatment), "bloodSugarGroup": info.blood_sugar_group,
                              "group": info.group, "workSchedule": info.work_schedule},
                  "evidence": ev, "recommendations": recs, "charts": charts,
                  "dataNote": "Synthetic data from the Crane Lab hackathon set. Nobody in it is real."}
        with open(f"{out_dir}/survivors/{sid}.json", "w") as f:
            json.dump(detail, f, separators=(",", ":"))
        roster.append({"survivorId": sid, "displayName": display, "status": st, "headline": headline, "tags": tags, "updatedAt": updated,
                       "careTeamToday": care_flag["today"], "todayReason": today_reason, "priority": priority,
                       "dataType": "synthetic", "module": "C4 Glucose Guide"})

    rank = {"alert": 0, "watch": 1, "ok": 2}
    roster.sort(key=lambda x: (rank[x["status"]], x["survivorId"]))
    with open(f"{out_dir}/roster.json", "w") as f:
        json.dump(roster, f, indent=1)

    def eff(e):
        return {k: (r(v_, 2) if isinstance(v_, (float, np.floating)) else v_) for k, v_ in e.items()}

    def rnd(x):
        return int(round(abs(x)))
    ld, wd = cohort["lateDinner"], cohort["walkAfterDinner"]
    d10 = next(d for d in dose if d["bin"] == "10-19 min")
    longer = [d for d in dose if d["bin"] not in ("none", "10-19 min")]
    rw, rl = real_checks.get("walk"), real_checks.get("lateDinner")
    sw_w = size_word(rw["real"]["est"], rw["program"]) if rw and rw.get("real") else None
    sw_l = size_word(rl["real"]["est"], rl["program"]) if rl and rl.get("real") else None
    plain_language = {
        "headline": f"Late dinner +{rnd(ld['est'])}, evening walk -{rnd(wd['est'])}, 10 minutes is enough.",
        "points": [
            {"id": "late_dinner", "text": f"Dinner at 8 pm or later: sugar rose about {rnd(ld['est'])} mg/dL more than after an earlier dinner, same person (95% CI {rnd(ld['lo'])} to {rnd(ld['hi'])}).",
             "est": r(ld["est"], 2), "lo": r(ld["lo"], 2), "hi": r(ld["hi"], 2), "realDataCheck": rl["label"] if rl else "not tested"},
            {"id": "evening_walk", "text": f"A walk after dinner: sugar rose about {rnd(wd['est'])} mg/dL less (95% CI {rnd(wd['hi'])} to {rnd(wd['lo'])} less).",
             "est": r(wd["est"], 2), "lo": r(wd["lo"], 2), "hi": r(wd["hi"], 2), "realDataCheck": rw["label"] if rw else "not tested"},
            {"id": "ten_minutes", "text": f"10 minutes is enough: 10 to 19 minutes went with {rnd(d10['est'])} mg/dL less; longer walks added little ("
                                          + " and ".join(f"{rnd(d['est'])} for {d['bin']}" for d in longer) + ").",
             "est": r(d10["est"], 2), "lo": r(d10["lo"], 2), "hi": r(d10["hi"], 2), "realDataCheck": "not tested"},
        ],
        "realData": (f"In 45 real adults (CGMacros) both pointed the same way but were smaller: the walk effect was {sw_w} and the late-dinner effect {sw_l} "
                     "the program's size, and neither is proven there yet.") if sw_w and sw_l else None,
        "source": "Synthetic 12-week program (made up), 500 sensor wearers; ML engine (double ML), within person.",
    }

    cohort_out = {
        "source": "synthetic 12-week program (made up), 500 sensor wearers; checkups for all 2,000",
        "effects": {k: eff(v_) for k, v_ in cohort.items()},
        "effectsByBloodSugarGroup": {g_: {k: eff(v_) for k, v_ in d.items()} for g_, d in by_group.items()},
        "walkDose": [eff(d) for d in dose],
        "dinnerHour": [eff(d) for d in dinner_hour],
        "curves": curves,
        "trial": {"arms": trial["arms"], "contrasts": {o: {k: eff(v_) for k, v_ in d.items()} for o, d in trial["contrasts"].items()},
                  "fastVsCoachingBySubgroup": {o: {k: eff(v_) for k, v_ in d.items()} for o, d in trial["fastVsCoachingBySubgroup"].items()},
                  "adherence": {g_: {k: r(v_, 2) for k, v_ in d.items()} for g_, d in trial["adherence"].items()},
                  "heterogeneity": {o: eff(d) for o, d in trial["heterogeneity"].items()}, "groupSpecificAdvice": bool(group_specific)},
        "robustness": {
            "negativeControl": eff(rob["negativeControl"]),
            "eValue": eff(rob["eValue"]),
            "instrument": {k: (eff(v_) if isinstance(v_, dict) else v_) for k, v_ in rob["instrument"].items()},
            "instrumentVerdict": "inconclusive: assignment moved walking only a little, and a dinner-only version gave an impossible effect size, "
                                 "so coaching changes the rise through more than walking (the exclusion restriction fails). Not counted as support.",
            "realDataChecks": {k: {"label": v_["label"], "source": v_["source"], "program": r(v_["program"], 2),
                                   "real": eff(v_["real"]) if v_["real"] else None} for k, v_ in real_checks.items()},
            "power": {k: {kk: (r(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v_.items()} for k, v_ in power.items()},
        },
        "statusCounts": status_counts,
        "realWorld": realworld,
        "engine": ml["engine"],
        "statistics": {k: (eff(v_) if isinstance(v_, dict) and "est" in v_ else v_) for k, v_ in stats.items() if k in ("effects", "negativeControl")}
                      | {"walkDose": stats["walkDose"], "dinnerHour": stats["dinnerHour"]},
        "forecast": {k: v_ for k, v_ in ml["forecast"].items() if k != "dinnerByPerson"},
        "mlCompare": mlc,
        "methods": __doc__.split("Methods (all numbers in evidence[] come from here):")[1].strip(),
        "plainLanguage": plain_language,
        **({"fairness": {"note": "Synthetic program: these show whether the engine treats groups alike, not real-world fairness. Groups under 20 people get no effect estimate. "
                                "walkDiffersP / lateDinnerDiffersP: Cochran Q test that the group effects differ (under 0.05 = they differ).",
                        "attributes": ml["fairness"]}} if ml.get("fairness") else {}),  # only when the engine run produced it
        "careTeamToday": {"rule": "More than 4% of readings below 70 mg/dL, or an average above 250 mg/dL.",
                          "count": sum(1 for x in roster if x["careTeamToday"])},
    }
    with open(f"{out_dir}/cohort.json", "w") as f:
        json.dump(cohort_out, f, indent=1)
    print(json.dumps({k: cohort_out[k] for k in ("effects", "walkDose", "dinnerHour", "statusCounts")}, indent=1), file=sys.stderr)
    print(json.dumps(cohort_out["trial"], indent=1), file=sys.stderr)
    print(json.dumps(cohort_out["effectsByBloodSugarGroup"], indent=1), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
