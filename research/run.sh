#!/bin/sh
# Run one scoring experiment on a Proxmox node; results land in research/runs/ locally.
# Usage: research/run.sh <host> <dataset_dir> <name> [score.py flags...]
set -e
H="$1"; D="$2"; N="$3"; shift 3; R=/root/c4-ar
ssh -o BatchMode=yes "$H" "mkdir -p $R/data $R/scripts $R/research/runs"
rsync -a "$D/synthetic_data" "$D/CGMacros" "$H:$R/data/"
rsync -a scripts/analyze.py "$H:$R/scripts/"
rsync -a research/score.py research/features.py "$H:$R/research/"
rsync -a --ignore-existing research/runs/ "$H:$R/research/runs/" 2>/dev/null || true
ssh -o BatchMode=yes "$H" "cd $R && { test -x bin/uv || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$R/bin UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1; } \
  && export UV_CACHE_DIR=$R/cache UV_PYTHON_INSTALL_DIR=$R/py \
  && bin/uv run -q --python 3.12 --with pandas --with numpy --with scikit-learn --with lightgbm python -I research/score.py data $N $* > research/runs/$N.log 2>&1; echo exit=\$?"
mkdir -p research/runs
rsync -a --exclude raw.pkl "$H:$R/research/runs/" research/runs/
tail -3 research/runs/$N.log | head -1 >/dev/null; grep -E "^(program|cgmacros)_" research/runs/$N.log || tail -20 research/runs/$N.log
