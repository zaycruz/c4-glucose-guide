#!/bin/sh
# Run the ML engine and the statistics-vs-ML benchmark on a remote Linux host over ssh (minutes of CPU),
# with the same locked environment as scripts/reproduce.sh, then copy the JSON into public/data.
# Afterwards run scripts/build-data.sh. For a full check of every README number use scripts/reproduce.sh.
# Usage: scripts/build-ml.sh <ssh host> <Challenge4_Dose_and_Timing dir>
set -e
H="${1:?ssh host}"; D="${2:?dataset dir}"; R=/root/c4-ml
ssh -o BatchMode=yes "$H" "mkdir -p $R/data $R/out"
rsync -a "$D/synthetic_data" "$D/CGMacros" "$H:$R/data/"
rsync -a --delete scripts pyproject.toml uv.lock .python-version "$H:$R/"
ssh -o BatchMode=yes "$H" "cd $R && { test -x bin/uv || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$R/bin UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1; } \
  && export UV_CACHE_DIR=$R/cache UV_PYTHON_INSTALL_DIR=$R/py \
  && bin/uv run --frozen python -I scripts/ml_effects.py data out > out/effects.log 2>&1 \
  && bin/uv run --frozen python -I scripts/ml_compare.py data out/mlcompare.json > out/compare.log 2>&1"
scp -q "$H:$R/out/ml_effects.json" "$H:$R/out/mlcompare.json" public/data/
[ -n "$CLEAN" ] && ssh -o BatchMode=yes "$H" "rm -rf $R"  # CLEAN=1 removes the remote copy (data is kept between runs otherwise)
echo "ML outputs in public/data (ml_effects.json, mlcompare.json); now run scripts/build-data.sh"
