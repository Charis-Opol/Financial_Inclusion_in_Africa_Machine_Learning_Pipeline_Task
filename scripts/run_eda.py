"""Re-executes the Phase 1 EDA notebook non-interactively (reproducibility chain, Phase 5.6).

The actual EDA logic lives in `notebooks/01_eda.ipynb` (Phase 1's deliverable
is explicitly notebook-based, not script-based -- see `Implementation_plan.md`
Phase 1) -- this just re-runs it in place so `reports/eda_summary.md`'s
findings stay reproducible from a single command.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "01_eda.ipynb"


def main() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace", str(NOTEBOOK_PATH),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
