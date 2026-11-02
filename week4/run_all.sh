#!/usr/bin/env bash
# Reproduces Week 4 outputs, phase by phase. Run from the repo root in Git Bash.
# Environment setup is at the top of week4/requirements.txt.
set -euo pipefail

PY=week4/.venv/Scripts/python.exe

# Phase 1: data audit (seconds) and TabPFN feasibility (about 1 hour on CPU, mostly the runtime steps).
"$PY" week4/src/data_audit.py
"$PY" week4/src/tabpfn_feasibility.py versions limits schema embeddings embeddings_null context finetune
"$PY" week4/src/tabpfn_feasibility.py runtime runtime_clean runtime_planned
