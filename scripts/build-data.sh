#!/bin/sh
# Rebuild everything the Worker serves. Usage: scripts/build-data.sh <Challenge4_Dose_and_Timing dir>
set -e
D="${1:?dataset dir}"
uv run -q --with pandas --with numpy python -I scripts/realworld.py "$D" public/data > /dev/null
uv run -q --with pandas --with numpy python -I scripts/analyze.py "$D" public/data 2> /dev/null
echo "built public/data from $D"
