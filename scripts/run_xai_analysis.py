"""Regenerates the whole Week 3 explainability analysis, in order, from the raw data.

    .venv-xai/Scripts/python scripts/run_xai_analysis.py            # reuse slow intermediate results
    .venv-xai/Scripts/python scripts/run_xai_analysis.py --force    # recompute everything (hours; see D8.4)

Steps (each is a script that can also be run on its own):

    1. run_xai_models.py       verify both models, retrain the MLP, fix the samples
    2. run_xai_shap.py         Task 1: SHAP, checked against exact Shapley values
    3. run_xai_captum.py       Task 2: Integrated Gradients, Feature Ablation, layers
    4. run_xai_internals.py    Task 3: internals and figures
    5. run_xai_comparison.py   Task 4: SHAP vs Captum, faithfulness, sanity checks
    6. notebooks/03_xai_analysis.ipynb, executed in place
    7. reports/xai_comparison_report.pdf, built from the Markdown report

Must be run with the XAI environment's interpreter (`requirements-xai.txt`):
it is passed to every step, and the notebook kernel is pointed at it too.
Any failing step stops the run with its exit code.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
NOTEBOOK = ROOT / "notebooks" / "03_xai_analysis.ipynb"

# (script, accepts --force)
STEPS = [
    ("run_xai_models.py", False),
    ("run_xai_shap.py", True),
    ("run_xai_captum.py", False),
    ("run_xai_internals.py", True),
    ("run_xai_comparison.py", True),
]


def check_environment() -> None:
    missing = []
    for module in ("captum", "shap", "streamlit", "torch", "xgboost"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    import pydantic

    if missing or not pydantic.VERSION.startswith("2"):
        sys.exit(f"Run this with the XAI environment (requirements-xai.txt): missing {missing}, "
                 f"pydantic {pydantic.VERSION}. Interpreter: {sys.executable}")


def run(command: list[str], label: str, env: dict | None = None) -> None:
    print(f"\n######## {label} ########", flush=True)
    start = time.time()
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"{label} failed with exit code {result.returncode}; later steps were not run")
    print(f"######## {label}: done in {time.time() - start:.0f} s ########", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="recompute cached attributions and interactions")
    args = parser.parse_args()
    check_environment()
    total = time.time()

    for script, accepts_force in STEPS:
        run([sys.executable, "-u", str(SCRIPTS / script), *(["--force"] if args.force and accepts_force else [])],
            script)

    # The default kernel spec launches `python` from PATH, which need not be this
    # interpreter. Putting this interpreter's directory first makes the notebook
    # run in the same environment as the scripts (the notebook also checks).
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=1800", str(NOTEBOOK)], "notebook 03_xai_analysis.ipynb", env=env)
    run([sys.executable, str(SCRIPTS / "build_xai_report_pdf.py")], "report PDF")
    print(f"\nEverything regenerated in {(time.time() - total) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
