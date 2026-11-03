# Status: Checkpoint 3, ready to run on the GPU machine

Date: 2026-11-03. Base commit: `95c91da`. Nothing committed or pushed. Nothing outside `week4/` changed, except the root `.gitignore`, so the two raw data files are tracked.

**Everything is prepared to run in one go.** Clone, run `bash week4/setup_env.sh`, then `bash week4/run_everything.sh`. Steps are in `week4/GPU_RUN.md`. No experiment has run on real labels.

## Done

| File | Contents |
|---|---|
| `week4/run_everything.sh` | Preflight, tests, smoke, GPU and CPU lanes in parallel, E6, analysis, completion report |
| `week4/setup_env.sh` | Builds `week4/.venv` on a fresh clone, downloads TabPFN v2, reports the GPU |
| `week4/GPU_RUN.md` | Step by step: push, clone, set up, run, bring results back |
| `week4/src/preflight.py` | Stops early on a dirty tree, missing data or no GPU. Measures GPU against CPU agreement and times one E2 fit |
| `week4/src/analysis.py` | Every pre-registered metric, 2,000-replicate paired bootstrap, Holm, and 14 figures (PNG and PDF) |
| `week4/src/e6_shap.py` | E6 (RQ6): SHAP rankings per country against the size-matched noise baseline, and Group DRO against ERM |
| `week4/src/completion.py` | Finished and missing folds per arm |
| `data/raw/*.csv` | Now tracked, so the clone has the data |

Phase 3 code from the previous report is unchanged apart from: the D3.3 arms, the estimator count in the H2 cache path, and fold metadata now recording budget, threads and host.

## Findings

- **Checked here:**
  - 41 tests pass.
  - Every stage, E6 and the analysis ran end to end on a smoke subset.
  - E6's refitted models reproduce their saved scores to 3e-8.
  - The preflight passes on CPU.
- **Not checked:** anything on a GPU. The preflight, tests and smoke run on that machine are the first test, and the run stops there if any fail.
- **TabPFN's default on a GPU is fp16 mixed precision**, while Phase 1 used float32 on CPU. The preflight records how far GPU scores move from CPU scores (D3.7).
- **Measured E6 cost:** 260 ms per row on this CPU, so MLP explanations run on the GPU there.

## Decisions and assumptions

New ones are D3.3 to D3.11 in `DECISIONS.md`. The ones to check:

- **D3.3, my call:** I added an ERM and Group DRO pair with class weights balanced within each country. That was your open question from section 11. I added the arms rather than leave E3 waiting, before any result exists. `mlp_dro` stays primary. To drop them, delete the two lines in `build_arms()`.
- **D3.4:** H2 now uses 8 estimators, matching TabPFN, instead of 2.
- **D3.5:** every hybrid runs under leave-one-country-out, so none is picked after seeing E5.
- **D3.8:** all stages run unattended, at your request. Checkpoints 4A to 4C become reviews of finished results.

## Problems and risks

- **The repository is public.** Pushing publishes `Train_v2.csv` and `Test_v2.csv`. If the dataset's terms don't allow redistribution, make the repo private before pushing (`gh repo edit --visibility private`), then push and clone.
- **GPU-side unknowns:** the driver's CUDA version (see `TORCH_INDEX_URL` in `GPU_RUN.md`), memory for 10,000-row TabPFN contexts, and run time. The preflight reports each before the long jobs start.
- **The Sagawa et al. (2020) PDF is still missing.** It doesn't block the run, but the paper's methods section needs a decision.

## Review checklist

1. `week4/GPU_RUN.md`.
2. `week4/run_everything.sh`: what runs, in what order, and what stops it.
3. `DECISIONS.md` D3.3 to D3.11.

## Suggested commit message

```text
Prepare Week 4 Phase 4 for a single GPU run: runner, preflight, analysis, E6, tracked data
```

## Next

1. Decide whether the repo should be private, then commit and push.
2. Follow `week4/GPU_RUN.md` on the GPU machine.
3. When results are pushed back, `git pull` here. I'll then check completeness and the preflight's agreement numbers, and write Checkpoint 4 from the tables, with verdicts applied by the pre-registered rules.
