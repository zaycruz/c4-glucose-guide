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
"""

import json
import os
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261017)
B_COHORT = 400
B_PERSON = 300
LATE_HOUR = 20
CURVE_OFFSETS = list(range(0, 195, 15))  # minutes 0..180
WALK_BINS = [(0, 0, "none"), (1, 9, "1-9 min"), (10, 19, "10-19 min"), (20, 29, "20-29 min"), (30, 999, "30+ min")]


def r(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


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

    # cohort mean curves
    cc = [f"c{o}" for o in CURVE_OFFSETS]
    curves = {
        "minutes": CURVE_OFFSETS,
        "earlyDinner": [r(x) for x in dinners[dinners.late == 0][cc].mean()],
        "lateDinner": [r(x) for x in dinners[dinners.late == 1][cc].mean()],
        "noWalk": [r(x) for x in mr[mr.walked == 0][cc].mean()],
        "walk": [r(x) for x in mr[mr.walked == 1][cc].mean()],
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
        din = sub[sub.meal_type == "dinner"]
        per[pid] = {
            "walk": person_effect(sub, "walked", walk_cov),
            "late": person_effect(din, "late", ["carbs_g"]) if parts.set_index("participant_id").loc[pid, "work_schedule"] != "night shift" else None,
        }
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
                "Your meals only: regression of rise on walk, carbs and meal type; 95% CI from 300 bootstrap resamples of your meals.",
                ci=(pw["lo"], pw["hi"]), n=pw["n_flag"] + pw["n_other"],
                caveat="You chose when to walk, so this is a pattern, not proof.")
            add("walk_effect_blend", "Best estimate for you (your data blended with the cohort)", sw["est"], "mg/dL", "person",
                f"Empirical-Bayes: {r(100 * sw['weight'], 0)}% weight on your data, rest on the cohort estimate. "
                f"People differed by about {r(np.sqrt(tau_w))} mg/dL in this effect.",
                ci=(sw["lo"], sw["hi"]))
        if pl:
            add("late_effect_own", "Your sugar rise after dinners at 8 pm or later vs earlier", pl["est"], "mg/dL", "person",
                "Your dinners only: regression of rise on late start and carbs; 95% CI from 300 bootstrap resamples.",
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
            "Within-person regression (person fixed effects) adjusted for carbs and meal type; 95% CI from cluster bootstrap over people.",
            ci=(cw["lo"], cw["hi"]), n=cw["rows"], caveat="Walking was self-chosen: association, not proof.")
        best = min([d for d in dose if d["bin"] != "none"], key=lambda d: d["est"])
        d1019 = next(d for d in dose if d["bin"] == "10-19 min")
        add("cohort_walk_10_19", "Program-wide: rise after a 10-19 min walk vs no walk", d1019["est"], "mg/dL", "cohort",
            "Within-person, carbs and meal type adjusted; cluster bootstrap 95% CI.", ci=(d1019["lo"], d1019["hi"]), n=d1019["n_meals"])
        d2029 = next(d for d in dose if d["bin"] == "20-29 min")
        d30 = next(d for d in dose if d["bin"] == "30+ min")
        add("cohort_walk_20_29", "Program-wide: rise after a 20-29 min walk vs no walk", d2029["est"], "mg/dL", "cohort",
            "Within-person, carbs and meal type adjusted; cluster bootstrap 95% CI.", ci=(d2029["lo"], d2029["hi"]), n=d2029["n_meals"])
        add("cohort_walk_30plus", "Program-wide: rise after a 30+ min walk vs no walk", d30["est"], "mg/dL", "cohort",
            "Within-person, carbs and meal type adjusted; cluster bootstrap 95% CI.", ci=(d30["lo"], d30["hi"]), n=d30["n_meals"])
        h89 = next(d for d in dinner_hour if d["bin"] == "8-9pm")
        add("cohort_dinner_8_9pm", "Program-wide: rise after dinner at 8-9 pm vs before 6 pm (same person)", h89["est"], "mg/dL", "cohort",
            "Within-person, carbs adjusted; night-shift workers excluded; cluster bootstrap 95% CI.", ci=(h89["lo"], h89["hi"]), n=h89["n_meals"],
            caveat="Association, not proof.")
        add("cohort_late", "Program-wide: rise after dinners at 8 pm or later vs earlier (same person)", cl["est"], "mg/dL", "cohort",
            "Within-person regression adjusted for carbs; night-shift workers excluded; cluster bootstrap 95% CI.",
            ci=(cl["lo"], cl["hi"]), n=cl["rows"], caveat="Association, not proof.")
        tf = trial["contrasts"]["hba1c_pct"]["coaching + overnight fast vs coaching"]
        tg = trial["contrasts"]["fasting_glucose_mg_dl"]["coaching + overnight fast vs coaching"]
        add("trial_fast_hba1c", "Randomized: HbA1c change, coaching + overnight fast vs coaching alone", tf["est"], "% points", "trial",
            "Difference in mean (week 12 - week 0) between randomized arms, completers; bootstrap 95% CI.",
            ci=(tf["lo"], tf["hi"]), n=tf["n"], caveat="Fair comparison: groups were assigned at random.", nd=2)
        add("trial_fast_fpg", "Randomized: fasting glucose change, coaching + overnight fast vs coaching alone", tg["est"], "mg/dL", "trial",
            "Difference in mean (week 12 - week 0) between randomized arms, completers; bootstrap 95% CI.",
            ci=(tg["lo"], tg["hi"]), n=tg["n"])

        evid = {e["id"]: e for e in ev}
        recs = []

        def v(i):
            return evid[i]["value"]

        # walk recommendation
        if pw and sw["hi"] < 0 and pw["hi"] < 0 and sw["weight"] >= 0.25:
            recs.append({"id": "walk", "title": "Take a short walk after meals",
                         "text": f"Your sugar rose about {abs(v('walk_effect_blend'))} mg/dL less after meals when you walked. "
                                 f"Aim for 10 to 20 minutes after your biggest meal.",
                         "weightOnYourData": r(sw["weight"], 2),
                         "basis": "personal", "evidenceIds": ["walk_effect_own", "walk_effect_blend", "cohort_walk_10_19"]})
        else:
            why = f"You walked after {v('walk_meals')} of your scored meals, too few to measure your own effect yet. " if walks < 6 else \
                  "We can't yet see a clear effect in your own meals. "
            recs.append({"id": "walk", "title": "Try a 10 to 20 minute walk after dinner",
                         "text": why + f"Across the program, a 10 to 19 minute walk went with a rise about {abs(v('cohort_walk_10_19'))} mg/dL smaller in the same person. "
                                       f"Longer walks added little: 20-29 minutes went with {abs(v('cohort_walk_20_29'))} mg/dL less and 30+ minutes with {abs(v('cohort_walk_30plus'))} mg/dL less.",
                         "basis": "cohort", "evidenceIds": ["walk_meals", "cohort_walk_10_19", "cohort_walk_20_29", "cohort_walk_30plus"]})
        # late dinner recommendation
        if night_shift:
            recs.append({"id": "late_dinner", "title": "Keep meal times steady around your shifts",
                         "text": "You work nights, so clock-time rules about late dinners don't fit you. Keep your main meal at the same point in your waking day.",
                         "basis": "rule", "evidenceIds": ["cohort_late"]})
        elif len(din) and v("late_share") is not None and v("late_share") >= 20:
            if pl and sl["lo"] > 0 and pl["lo"] > 0 and sl["weight"] >= 0.25:
                text = f"{v('late_share')}% of your dinners started at 8 pm or later, and those raised your sugar about {v('late_effect_blend')} mg/dL more. Try to start dinner before 8 pm."
                basis, ids = "personal", ["late_share", "late_effect_own", "late_effect_blend"]
            else:
                text = (f"{v('late_share')}% of your dinners started at 8 pm or later. Across the program, late dinners raised sugar about {v('cohort_late')} mg/dL more than earlier ones in the same person, "
                        f"and a dinner at 8-9 pm raised it {v('cohort_dinner_8_9pm')} mg/dL more than one before 6 pm. Try to start dinner before 8 pm.")
                basis, ids = "cohort", ["late_share", "cohort_late", "cohort_dinner_8_9pm"]
            recs.append({"id": "late_dinner", "title": "Move dinner earlier", "text": text, "basis": basis, "evidenceIds": ids})
        elif len(din):
            recs.append({"id": "late_dinner", "title": "Keep dinner early",
                         "text": (f"None of your {v('dinner_count')} scored dinners started at 8 pm or later" if v('late_share') == 0 else f"Only {v('late_share')}% of your dinners started at 8 pm or later") + f". Keep it that way: across the program, late dinners raised sugar about {v('cohort_late')} mg/dL more.",
                         "basis": "cohort", "evidenceIds": ["dinner_count", "late_share", "cohort_late"]})
        # overnight fast
        low_sugar = m_last["tbr70"] > 4
        if not night_shift and "fast_avg" in evid and low_sugar:
            recs.append({"id": "overnight_fast", "title": "Hold off on longer fasts for now",
                         "text": f"{v('tbr70')}% of your readings were below 70 mg/dL, so check with your care team before stretching your overnight fast. "
                                 f"Right now it averages {v('fast_avg')} hours.",
                         "basis": "rule", "evidenceIds": ["tbr70", "fast_avg", "status_rule"]})
        elif not night_shift and "fast_avg" in evid:
            recs.append({"id": "overnight_fast", "title": "Aim for a 13-hour overnight break",
                         "text": f"Your overnight fast averaged {v('fast_avg')} hours, and {v('fast13_share')}% of nights reached 13 hours. "
                                 f"In the randomized program, adding a 13-hour overnight fast to coaching changed HbA1c by {v('trial_fast_hba1c')} points "
                                 f"(95% CI {evid['trial_fast_hba1c']['ci'][0]} to {evid['trial_fast_hba1c']['ci'][1]}) compared with coaching alone.",
                         "basis": "trial", "evidenceIds": ["fast_avg", "fast13_share", "trial_fast_hba1c"]})
        if st == "alert" and low_sugar:
            recs.insert(0, {"id": "talk_to_team", "title": "Tell your care team about low readings",
                            "text": f"{v('tbr70')}% of your readings were below 70 mg/dL ({latest_label}); the usual target is under 4%. "
                                    f"Low sugar can make you shaky or dizzy. Share this before your next visit.",
                            "basis": "rule", "evidenceIds": ["tbr70", "status_rule"]})
        elif st == "alert":
            recs.insert(0, {"id": "talk_to_team", "title": "Share these readings with your care team",
                            "text": f"Only {v('tir')}% of your readings were in the 70-180 range ({latest_label}); the usual target is more than 70%. "
                                    f"Your average was {v('mean_glucose')} mg/dL. Bring this to your next visit.",
                            "basis": "rule", "evidenceIds": ["tir", "mean_glucose", "status_rule"]})

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
        if st == "alert":
            headline = f"{round(m_last["tir"])}% time in range ({latest_label}); " + "; ".join(reasons[:1])
        elif st == "watch":
            headline = f"{round(m_last["tir"])}% time in range; " + "; ".join(reasons[:1])
        else:
            headline = f"{round(m_last["tir"])}% time in range, steady"
        updated = str(gp.timestamp.max().isoformat())
        summary = (f"{info.age}-year-old {info.sex}, {info.cancer_type} cancer, {info.years_since_treatment} years after treatment; "
                   f"{info.blood_sugar_group} at start; group: {info.group}; {info.work_schedule}. "
                   f"Average sugar {round(m_last["mean"])} mg/dL in {latest_label}, {round(m_last["tir"])}% of readings in range. "
                   + (f"Median rise after dinner {round(din.rise.median())} mg/dL. " if len(din) else "")
                   + "Synthetic participant (made-up data).")
        detail = {"survivorId": sid, "displayName": display, "status": st, "statusReasons": reasons, "headline": headline,
                  "summary": summary, "updatedAt": updated,
                  "profile": {"age": int(info.age), "sex": info.sex, "cancerType": info.cancer_type,
                              "yearsSinceTreatment": float(info.years_since_treatment), "bloodSugarGroup": info.blood_sugar_group,
                              "group": info.group, "workSchedule": info.work_schedule},
                  "evidence": ev, "recommendations": recs, "charts": charts,
                  "dataNote": "Synthetic data from the Crane Lab hackathon set. Nobody in it is real."}
        with open(f"{out_dir}/survivors/{sid}.json", "w") as f:
            json.dump(detail, f, separators=(",", ":"))
        roster.append({"survivorId": sid, "displayName": display, "status": st, "headline": headline, "updatedAt": updated})

    rank = {"alert": 0, "watch": 1, "ok": 2}
    roster.sort(key=lambda x: (rank[x["status"]], x["survivorId"]))
    with open(f"{out_dir}/roster.json", "w") as f:
        json.dump(roster, f, indent=1)

    def eff(e):
        return {k: (r(v_, 2) if isinstance(v_, (float, np.floating)) else v_) for k, v_ in e.items()}

    cohort_out = {
        "source": "synthetic 12-week program (made up), 500 sensor wearers; checkups for all 2,000",
        "effects": {k: eff(v_) for k, v_ in cohort.items()},
        "effectsByBloodSugarGroup": {g_: {k: eff(v_) for k, v_ in d.items()} for g_, d in by_group.items()},
        "walkDose": [eff(d) for d in dose],
        "dinnerHour": [eff(d) for d in dinner_hour],
        "curves": curves,
        "trial": {"arms": trial["arms"], "contrasts": {o: {k: eff(v_) for k, v_ in d.items()} for o, d in trial["contrasts"].items()}},
        "statusCounts": status_counts,
        "methods": __doc__.split("Methods (all numbers in evidence[] come from here):")[1].strip(),
    }
    with open(f"{out_dir}/cohort.json", "w") as f:
        json.dump(cohort_out, f, indent=1)
    print(json.dumps({k: cohort_out[k] for k in ("effects", "walkDose", "dinnerHour", "statusCounts")}, indent=1), file=sys.stderr)
    print(json.dumps(cohort_out["trial"], indent=1), file=sys.stderr)
    print(json.dumps(cohort_out["effectsByBloodSugarGroup"], indent=1), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
