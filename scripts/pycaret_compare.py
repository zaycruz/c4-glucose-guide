"""pycaret leaderboard on the same pre-meal features as scripts/ml_compare.py (new-people split).

Every model pycaret ships for regression, 5-fold cross-validation grouped by person, default
settings. Reads <features_dir>/{program,cgmacros}_features.csv (written by ml_compare.py with
FEATURES_DIR set). pycaret needs Python 3.11:

  uv run --python 3.11 --with pycaret python -I scripts/pycaret_compare.py <features_dir> <out_json>
"""
import json
import sys

import pandas as pd
from pycaret.regression import compare_models, pull, setup


def board(path):
    df = pd.read_csv(path)
    groups = df.pop("participant_id")
    setup(data=df, target="rise", fold_strategy="groupkfold", fold_groups=groups, fold=5,
          data_split_shuffle=False, train_size=0.999, session_id=20261017, verbose=False, html=False, n_jobs=-1)
    compare_models(sort="MAE", n_select=1, verbose=False, exclude=["catboost"])
    lb = pull()
    return {"meals": int(len(df)), "people": int(groups.nunique()),
            "board": [{"model": row["Model"], "mae": round(float(row["MAE"]), 2), "rmse": round(float(row["RMSE"]), 2),
                       "r2": round(float(row["R2"]), 3)} for _, row in lb.iterrows()]}


if __name__ == "__main__":
    d, out = sys.argv[1], sys.argv[2]
    res = {"tool": "pycaret compare_models, grouped 5-fold by person, default settings",
           "program": board(f"{d}/program_features.csv"), "cgmacros": board(f"{d}/cgmacros_features.csv")}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
