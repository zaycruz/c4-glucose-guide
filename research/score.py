"""Frozen scorer for the prediction autoresearch loop. Do not edit during the loop (see research/LEAKAGE.md).

Usage (on Proxmox, not the Mac):
  python -I research/score.py <dataset_dir> <name> [--compare <name>] [--shuffle] [--lookahead] [--final]

The loop edits only research/features.py, which must expose:
  build(raw, lookahead=False) -> (prog_X, cg_X)   feature frames row-aligned to raw["prog"] / raw["cg"]
  fold_encode(Xtr, ytr, Xte) -> (Xtr2, Xte2)      fold-aware encodings; sees training outcomes only
  model() -> estimator with fit/predict
Columns prefixed hist_ use the person's own earlier outcomes; they are dropped for the *_cold splits.
"""
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
from analyze import load, meal_responses  # noqa: E402

OUT = os.path.join(HERE, "runs")
SEED = 20261017
B = 1000
OUTCOME = re.compile(r"^(peak|iauc|rise|c\d+|n_post|clean|next_ts|ts_key)$")
POST = re.compile(r"(walked_after|walked$|finished_program|days_in_program|week ?12|wk12|post_)")
DAILY_ONLY = ["steps", "active_minutes", "walked_after_dinner_min", "bedtime", "sleep_minutes", "sleep_efficiency_pct",
              "resting_heart_rate", "food_logged", "first_ate", "last_ate", "eating_window_hours",
              "overnight_fast_hours", "evening_calories_pct", "healthy_eating_score", "alcohol_drinks", "energy_1_to_5",
              "hunger_1_to_5"]


def held_out(pid):
    return int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 5 == 0


def cg_frame(d):
    """CGMacros meals and their 2-hour rise (same rules as scripts/ml_compare.py), keeping timestamps."""
    mm = pd.read_csv(f"{d}/CGMacros/cgmacros_minute_by_minute.csv", parse_dates=["timestamp"])
    meals = pd.read_csv(f"{d}/CGMacros/cgmacros_meals.csv", parse_dates=["timestamp"])
    meals["meal_type"] = meals.meal_type.str.lower().str.replace(r"snacks?( 1)?", "snack", regex=True)
    meals = meals.sort_values(["participant_id", "timestamp"])
    meals["next"] = meals.groupby("participant_id").timestamp.shift(-1)
    keep = (meals.next.isna()) | ((meals.next - meals.timestamp) > pd.Timedelta(minutes=120))
    meals = meals[keep & (meals.meal_carbs_g >= 10)]
    by = {pid: g.set_index("timestamp").glucose_libre_mg_dl for pid, g in mm.groupby("participant_id")}
    rows = []
    for m in meals.itertuples():
        g = by.get(m.participant_id)
        if g is None:
            continue
        win = g.loc[m.timestamp: m.timestamp + pd.Timedelta(minutes=120)]
        if len(win) < 100 or pd.isna(win.iloc[0]):
            continue
        rows.append({"participant_id": m.participant_id, "ts": m.timestamp, "meal_type": m.meal_type,
                     "calories": m.meal_calories, "carbs_g": m.meal_carbs_g, "protein_g": m.meal_protein_g,
                     "fat_g": m.meal_fat_g, "fiber_g": m.meal_fiber_g, "amount_eaten_pct": m.meal_amount_eaten_pct,
                     "baseline": win.iloc[0], "rise": win.max() - win.iloc[0]})
    cg = pd.DataFrame(rows).dropna(subset=["rise", "carbs_g", "protein_g", "fat_g", "fiber_g", "calories"])
    return cg.reset_index(drop=True), mm


def load_raw(d):
    cache = os.path.join(OUT, "raw.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    parts, checks, daily, meals, glu, _ = load(d)
    prog = meal_responses(meals, glu, parts).reset_index(drop=True)
    cg, mm = cg_frame(d)
    c = f"{d}/CGMacros"
    raw = {"prog": prog, "parts": parts, "checks": checks, "daily": daily, "meals": meals, "glu": glu,
           "cg": cg, "cg_mm": mm, "cg_parts": pd.read_csv(f"{c}/cgmacros_participants.csv"),
           "cg_gut": pd.read_csv(f"{c}/cgmacros_gut_health.csv"), "cg_microbes": pd.read_csv(f"{c}/cgmacros_gut_microbes.csv")}
    os.makedirs(OUT, exist_ok=True)
    pd.to_pickle(raw, cache)
    return raw


def check_columns(X, y, label):
    bad = [c for c in X.columns if OUTCOME.match(c)]
    bad += [c for c in X.columns if POST.search(c) and not c.startswith("prev_")]
    bad += [c for c in X.columns if any(k in c for k in DAILY_ONLY) and not c.startswith(("prev_", "pre_"))]
    assert not bad, f"{label}: forbidden feature columns {bad}"
    nonnum = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    assert not nonnum, f"{label}: non-numeric columns {nonnum}"
    for c in X.columns:
        v = X[c].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(y)
        if ok.sum() > 100 and np.std(v[ok]) > 0:
            assert abs(np.corrcoef(v[ok], y[ok])[0, 1]) < 0.98, f"{label}: {c} is near-identical to the outcome"


def run_split(F, X, y, g, folds):
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        Xtr, Xte = F.fold_encode(X.iloc[tr].copy(), y[tr].copy(), X.iloc[te].copy())
        pred[te] = F.model().fit(Xtr, y[tr]).predict(Xte)
    return pred


def metrics(y, p):
    e = y - p
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), "n": int(len(y))}


def paired(y, p_new, p_old, g, rng):
    """Paired person-bootstrap 95% CI of MAE(new) - MAE(old). Negative is better."""
    d = np.abs(y - p_new) - np.abs(y - p_old)
    people = np.unique(g)
    s = pd.Series(d).groupby(g).agg(["sum", "count"]).loc[people]
    boots = []
    for _ in range(B):
        k = rng.choice(len(people), len(people))
        boots.append(s["sum"].to_numpy()[k].sum() / s["count"].to_numpy()[k].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def dev_mask(df, final):
    return np.ones(len(df), bool) if final else ~np.array([held_out(p) for p in df.participant_id])


def splits_for(df, X, final):
    """Yield (split name, row mask, folds over masked rows, drop hist_)."""
    pid = df.participant_id.to_numpy()
    hold = np.array([held_out(p) for p in pid])
    order = df.groupby("participant_id").ts.rank(pct=True, method="first").to_numpy()
    if not final:
        m = ~hold
        gk = list(GroupKFold(5).split(X[m], groups=pid[m]))
        later = order[m] > 0.75
        known = [(np.where(~later)[0], np.where(later)[0])]
    else:  # train on dev people (and holdout people's earlier days for the known split), test on holdout people
        m = np.ones(len(pid), bool)
        gk = [(np.where(~hold)[0], np.where(hold)[0])]
        known = [(np.where(~(hold & (order > 0.75)))[0], np.where(hold & (order > 0.75))[0])]
    return m, {"new_cold": (gk, True), "new_online": (gk, False), "known_later": (known, False)}


def main():
    a = sys.argv[1:]
    d, name = a[0], a[1]
    flags = set(x for x in a[2:] if x.startswith("--"))
    cmp = a[a.index("--compare") + 1] if "--compare" in a else None
    t0 = time.time()
    import features as F
    raw = load_raw(d)
    prog_X, cg_X = F.build(raw, lookahead="--lookahead" in flags)
    rng = np.random.default_rng(SEED)
    res = {"name": name, "flags": sorted(flags), "splits": {}}
    preds = {}
    for ds, df, X in (("program", raw["prog"], prog_X), ("cgmacros", raw["cg"], cg_X)):
        X = X.reset_index(drop=True)
        assert len(X) == len(df), f"{ds}: features not row-aligned"
        y_all = df.rise.to_numpy(float)
        check_columns(X, y_all, ds)
        m, sp = splits_for(df, X, "--final" in flags)
        Xm, ym, gm = X[m].reset_index(drop=True), y_all[m], df.participant_id.to_numpy()[m]
        if "--shuffle" in flags:
            ym = rng.permutation(ym)
        for sname, (folds, cold) in sp.items():
            if ds == "cgmacros" and sname == "known_later":
                continue
            Xs = Xm.drop(columns=[c for c in Xm.columns if c.startswith("hist_")]) if cold else Xm
            p = run_split(F, Xs, ym, gm, folds)
            ok = np.isfinite(p)
            key = f"{ds}_{sname}"
            res["splits"][key] = metrics(ym[ok], p[ok])
            preds[key] = (np.where(ok)[0], p[ok])
            print(f"{key}: {json.dumps(res['splits'][key])}  {time.time() - t0:.0f}s", file=sys.stderr)
        res.setdefault("features", {})[ds] = list(X.columns)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{name}.npz"), **{k + "_idx": v[0] for k, v in preds.items()},
             **{k + "_p": v[1] for k, v in preds.items()})
    if cmp:
        old = np.load(os.path.join(OUT, f"{cmp}.npz"))
        res["vs"] = {"name": cmp}
        for key, (idx, p) in preds.items():
            if key + "_p" not in old.files:
                continue
            ds = key.split("_")[0]
            df = raw["prog"] if ds == "program" else raw["cg"]
            m = dev_mask(df, "--final" in flags)
            y = df.rise.to_numpy(float)[m]
            g = df.participant_id.to_numpy()[m]
            po = pd.Series(old[key + "_p"], index=old[key + "_idx"])
            pn = pd.Series(p, index=idx)
            both = po.index.intersection(pn.index)
            res["vs"][key] = paired(y[both], pn[both].to_numpy(), po[both].to_numpy(), g[both], rng)
    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    row = [name] + [f"{res['splits'][k]['mae']:.2f}/{res['splits'][k]['r2']:.3f}" for k in sorted(res["splits"])]
    with open(os.path.join(OUT, "results.tsv"), "a") as f:
        f.write("\t".join(row) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
