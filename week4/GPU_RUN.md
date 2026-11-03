# Running Phase 4 on the GPU machine

One script runs everything: checks, every experiment (E2 to E7), E6 (SHAP), then all metrics, tables and figures. Finished work is skipped on a rerun, so if anything stops, run the same command again.

## 1. On this computer: commit and push

Commit the working tree, then push. The raw data (`data/raw/Train_v2.csv`, `Test_v2.csv`) is now tracked, so the clone has everything the runs read.

## 2. On the GPU machine: clone and set up

```bash
git clone https://github.com/Charis-Opol/Financial_Inclusion_in_Africa_Machine_Learning_Pipeline_Task.git
cd Financial_Inclusion_in_Africa_Machine_Learning_Pipeline_Task
curl -LsSf https://astral.sh/uv/install.sh | sh     # only if uv is missing
bash week4/setup_env.sh
```

The last line of the setup should say `CUDA visible: True` and name the GPU. If it says `False`, the driver is older than the CUDA build of torch 2.12.1 on PyPI. Check the driver's CUDA version with `nvidia-smi`, then rerun the setup with a matching PyTorch wheel index, for example:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash week4/setup_env.sh
```

## 3. Run everything

```bash
mkdir -p week4/results/logs
nohup bash week4/run_everything.sh > week4/results/logs/run_everything.out 2>&1 &
tail -f week4/results/logs/run_everything.out
```

`nohup` keeps it running after you close the terminal. Stop following the log with Ctrl+C; the run carries on.

**What happens, in order:**

| Step | What | Stops everything if it fails? |
|---|---|---|
| Preflight | Clean commit, data present, GPU visible, TabPFN v2 loads; GPU against CPU agreement; one timed E2 fit and a projection of the TabPFN hours | Yes |
| Tests | All 41 tests, including the real-TabPFN leakage test | Yes |
| Smoke | Every stage, E6 and the analysis on a tiny subset, on the GPU | Yes |
| GPU lane | TabPFN in E2 and E4, H1, H2 (8 estimators), hybrids under leave-one-country-out | No; other lanes continue |
| CPU lanes | XGBoost and MLP arms of E2, E3, E4, E7, one lane per seed on 12+ cores | No |
| E6 | SHAP for XGBoost, the ERM MLP and the Group DRO MLP, seed 0 | No |
| Analysis | Metrics, bootstrap intervals, paired comparisons, all figures | No |

**Logs**, in `week4/results/logs/<date_time>/`: `main.log` (the summary), `preflight.log`, `tests.log`, `smoke.log`, one `lane_*.log` per lane, `e6.log`, `analysis.log`.

**How long:** not known until the preflight runs. Look for `one planned E2 fit on cuda: ...s` in `preflight.log`. On this laptop's CPU the CPU work alone was estimated at about 11 hours of single-process compute. Split over 5 seed lanes on a many-core machine, that should be a few hours, but this is an estimate.

**Settings**, as environment variables before `bash`: `DEVICE` (default `cuda`), `H2_ESTIMATORS` (default 8), `PARALLEL_SEEDS` (default 1 on 12+ cores).

## 4. Bring the results back

When `main.log` ends with "All Phase 4 runs finished":

```bash
git add week4/results week4/figures
git commit -m "Week 4 Phase 4: all experiment results and figures"
git push
```

Then `git pull` on this computer. The embeddings cache (`week4/results/cache/`) and the smoke outputs are gitignored and not needed: every metric is recomputed from `week4/results/oof/`.

**If it ends with "Some runs failed or are missing":** read the failing lane's log, fix the cause, and rerun the same `nohup` command. Finished folds are skipped. `week4/results/phase4_completion.md` lists what is done.
