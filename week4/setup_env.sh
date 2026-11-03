#!/usr/bin/env bash
# Builds week4/.venv on a fresh clone (Linux, or Git Bash on Windows). Run from the repo root:
#   bash week4/setup_env.sh
# Needs uv (https://docs.astral.sh/uv/) and internet access. uv fetches Python 3.13 if missing.
# On Linux, torch==2.12.1 from PyPI is a CUDA build. If its CUDA version is too new for the
# machine's driver, set TORCH_INDEX_URL to a matching PyTorch wheel index before running, e.g.
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash week4/setup_env.sh
set -euo pipefail

command -v uv >/dev/null || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

uv venv week4/.venv --python 3.13
if [ -x week4/.venv/bin/python ]; then PY=week4/.venv/bin/python; else PY=week4/.venv/Scripts/python.exe; fi

uv pip install --python "$PY" -r week4/requirements.txt
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  uv pip install --python "$PY" --reinstall-package torch "torch==2.12.1" --index-url "$TORCH_INDEX_URL"
fi
uv pip install --python "$PY" --no-deps -e .

# Downloads the TabPFN v2 weights once (no login needed, Phase 1) and reports what torch can see.
"$PY" - <<'EOF'
import sys
sys.path.insert(0, "week4/src")
import numpy as np, torch
from tabpfn_tools import make_v2
model = make_v2(device="cpu", seed=0, n_estimators=1)
model.fit(np.random.default_rng(0).normal(size=(50, 3)), np.arange(50) % 2)
print("torch", torch.__version__, "| CUDA visible:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
EOF
echo "Environment ready: $PY"
