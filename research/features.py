"""The one file the autoresearch loop edits. Interface and rules: research/score.py and research/LEAKAGE.md."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MACROS = ["calories", "carbs_g", "protein_g", "fat_g", "fiber_g", "sugar_g"]
MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]
HGB = dict(max_iter=200, learning_rate=0.05, min_samples_leaf=40, random_state=0)


def base(df, macros):
    X = df[macros + ["baseline", "hour"]].astype(float).copy()
    X["pre_slope"] = (df.baseline - df.pre15).fillna(0.0) if "pre15" in df else 0.0
    for t in MEAL_TYPES:
        X[f"mt_{t}"] = (df.meal_type == t).astype(float)
    if "mediterranean_style" in df:
        X["med"] = (df.mediterranean_style == "yes").astype(float)
    return X


def build(raw, lookahead=False):
    prog = raw["prog"]
    cg = raw["cg"].copy()
    cg["hour"] = cg.ts.dt.hour + cg.ts.dt.minute / 60
    return base(prog, MACROS), base(cg, ["calories", "carbs_g", "protein_g", "fat_g", "fiber_g"])


def fold_encode(Xtr, ytr, Xte):
    return Xtr, Xte


def model():
    return HistGradientBoostingRegressor(**HGB)
