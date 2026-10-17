#!/usr/bin/env bash
# One command from the raw Challenge 4 data to every headline number in README.md. See REPRODUCE.md.
#
#   scripts/reproduce.sh <Challenge4_Dose_and_Timing dir> [out dir, default repro_out]
#
# Steps: check the data against data/SHA256SUMS -> real-data checks -> ML engine (double ML, forecast)
# -> statistics-vs-ML benchmark -> evidence and advice -> diff every README number (scripts/headline.py).
# Needs uv (https://docs.astral.sh/uv/); Python 3.12 and the pinned packages come from uv.lock.
# Seeds are fixed in the scripts (20261017; realworld 7; models random_state=0), so a rerun gives the same numbers.
# Runtime on 4 vCPU: about 10 min (ML engine ~4 min, benchmark ~4 min, the rest ~1 min).
# It writes only to the out dir; public/data (what the Worker serves) is not touched.
set -uo pipefail
cd "$(dirname "$0")/.."
D="$(cd "${1:?usage: scripts/reproduce.sh <Challenge4_Dose_and_Timing dir> [out dir]}" && pwd)"
OUT="${2:-repro_out}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export PYTHONHASHSEED=0
py() { uv run --frozen python -I "$@"; }
step() {
  local name="$1"; shift
  local t0=$SECONDS
  echo "== $name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$OUT/steps.tsv"
  if [ $rc -ne 0 ]; then tail -20 "$OUT/$name.log"; echo "FAILED at $name (log: $OUT/$name.log)"; exit $rc; fi
}
: > "$OUT/steps.tsv"
if command -v sha256sum >/dev/null; then SUM="sha256sum"; else SUM="shasum -a 256"; fi
step 0-data-check sh -c "cd '$D' && $SUM -c '$PWD/data/SHA256SUMS'"
step 1-env uv sync --frozen
step 2-realworld py scripts/realworld.py "$D" "$OUT"
step 3-ml-engine py scripts/ml_effects.py "$D" "$OUT"
step 4-ml-benchmark py scripts/ml_compare.py "$D" "$OUT/mlcompare.json"
step 5-evidence py scripts/analyze.py "$D" "$OUT"
echo "== 6-headline-diff"
py scripts/headline.py "$OUT" --expected public/data --readme README.md --md "$OUT/headline.md" > "$OUT/6-headline-diff.log" 2>&1
rc=$?
tail -1 "$OUT/headline.md"
printf '6-headline-diff\t%s\t0s\n' "$rc" >> "$OUT/steps.tsv"
{ echo "python: $(uv run --frozen python -c 'import sys;print(sys.version.split()[0])')"
  uv run --frozen python -c 'import numpy,pandas,sklearn,scipy;print("numpy",numpy.__version__,"pandas",pandas.__version__,"scikit-learn",sklearn.__version__,"scipy",scipy.__version__)'
  echo "uv: $(uv --version)"; echo "commit: $(git rev-parse HEAD 2>/dev/null)"; } > "$OUT/versions.txt"
exit $rc
