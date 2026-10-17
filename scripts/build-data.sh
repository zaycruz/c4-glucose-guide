#!/bin/sh
# Rebuild everything the Worker serves (offline, about 30 s). Usage: scripts/build-data.sh <Challenge4_Dose_and_Timing dir>
# Needs public/data/ml_effects.json from scripts/build-ml.sh (or copied from a scripts/reproduce.sh out dir).
set -e
D="${1:?dataset dir}"
uv run --frozen python -I scripts/realworld.py "$D" public/data > /dev/null
uv run --frozen python -I scripts/analyze.py "$D" public/data 2> /dev/null
echo "built public/data from $D"
