# Status: Checkpoint 2 (Phase 2)

Date: 2026-11-03. Base commit: `c84b96c`. Nothing committed or pushed. Nothing outside `week4/` changed.

**Phase 2 is complete.** The Phase 3 code you asked me to write during the run is also ready. It gets its own review at Checkpoint 3 before any experiment uses it.

## Done

| File | Contents |
|---|---|
| `week4/src/e1_reproduce_baseline.py` | E1: the baseline's own arm functions, run as reported and with in-fold preprocessing |
| `week4/src/e1_summary.py` | Compares everything with the original saved values, by script |
| `week4/results/baseline_reproduction.csv` | Every fold, country and metric |
| `week4/results/baseline_reproduction_summary.md` | **The mismatch note** |
| `week4/results/runs/e1_*.json` | Run records |

**Phase 3 code**, for Checkpoint 3: `week4/src/group_dro.py`, `tabpfn_tools.py`, `harness.py`, `arms.py`, `hybrids.py`, `week4/run_all.py`, `week4/tests/`, `week4/CODE_WALKTHROUGH.md`.

## Findings

From `week4/results/baseline_reproduction_summary.md`:

- **The baseline reproduces exactly.**
  - Every fold PR-AUC and threshold matches the original run to full precision. The maximum difference is 0.000000 for both arms.
  - Tuned hyperparameters match in all 5 folds of both arms.
  - Every reported mean ± std matches to three decimals: XGBoost 0.590 ± 0.019, MLP 0.584 ± 0.019, and XGBoost per country Kenya 0.649, Rwanda 0.473, Tanzania 0.627, Uganda 0.553.
- **Fitting preprocessing before CV barely mattered.**

  | Arm | Change in mean PR-AUC | Largest change in any fold |
  |---|---|---|
  | XGBoost | 0.000000 | 0.000000 |
  | MLP | −0.000919 | 0.002724 |

  Both are well below δ = 0.01. XGBoost doesn't change because one-hot encoding learns nothing beyond which levels exist.
- **Run time:** XGBoost about 142 s and MLP about 678 s per 5-fold run.

## Decisions and assumptions

- **D2.1:** E1 imports and calls the baseline's own functions, never its `main()`, which writes to `reports/`.
- **D2.2:** E1 covers the two baseline arms your brief names, not the SMOTE or 3-layer arms.
- **D2.4:** all Week 4 results use fold-fitted preprocessing, and the reported baseline stays as reported. The effect measured above shows the two are comparable.

## Problems and risks

- **My error:** my first comparison wrongly showed the per-country SDs as mismatched, because I used the sample SD and the original code uses the population SD. Fixed and logged before this report; the reproduction itself had no mismatch.
- **Smaller fixes:**
  - **Earlier timings:** the Phase 3 tests and smoke runs shared the CPU with part of E1, limited to 2 threads, so E1's run times are slightly pessimistic. Its results are deterministic and unaffected.
  - **Dependencies:** `imbalanced-learn` and `seaborn` were missing from `week4/.venv`, so I added them at the baseline's pinned versions.
- **Still open:** the Group DRO source (the Sagawa PDF) and the GPU choice.

## Review checklist

1. `week4/results/baseline_reproduction_summary.md`.
2. `week4/src/e1_reproduce_baseline.py`, especially `nested_cv_in_fold`, which mirrors `cv_runner.run_nested_cv` except for the preprocessing line.

To reproduce (Git Bash, about 30 minutes):

```bash
week4/.venv/Scripts/python.exe week4/src/e1_reproduce_baseline.py
week4/.venv/Scripts/python.exe week4/src/e1_summary.py
```

## Suggested commit message

```text
Add Week 4 Phase 2: exact reproduction of the baseline (E1)
```

The Phase 3 files are also in the working tree. Commit them separately after Checkpoint 3, or now if you prefer, since no result depends on them yet.

## Next

**Checkpoint 3 needs your review of the Phase 3 code before any experiment uses it.** In this order:
1. `week4/CODE_WALKTHROUGH.md`.
2. `week4/src/group_dro.py`.
3. `week4/src/tabpfn_tools.py`.
4. Section 11 of the walkthrough: four design decisions for you.

**What I need from you:**

1. **Reply "continue to Phase 3"** to approve Phase 2. I'll then write the Checkpoint 3 report.
2. **The four design points in section 11 of the walkthrough.** The first matters most: Group DRO may push towards Kenya rather than the country with the lowest PR-AUC, because class weights are pooled across countries. Should I add an arm whose class weights are computed within each country?
3. **The Sagawa et al. (2020) PDF:** add it, or I describe the update as our own implementation. This must be settled before the paper's methods section.
4. **Your GPU setup:** what kind it is and where it runs. Only TabPFN uses it.
