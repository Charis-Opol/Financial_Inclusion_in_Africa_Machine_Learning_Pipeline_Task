# Status: Checkpoint 1 (Phase 1)

Date: 2026-11-02. Base commit: `4929a9b`. Nothing committed or pushed. No study model was trained, and no TabPFN score was computed on the real labels.

**Please decide before Phase 2:** the questions under Next.

## Done

| File | Contents |
|---|---|
| `week4/results/data_audit.md`, `per_country_counts.csv`, `fold_sizes.csv` | Per-country rows, missingness, class balance, prevalence, fold sizes |
| `week4/results/feasibility.md` | TabPFN version, licence, limits, hybrid checks, runtime |
| `week4/results/runs/phase1_*.json` | Run records with git commit and raw values |
| `week4/src/data_audit.py`, `tabpfn_feasibility.py`, `run_record.py` | Code for all of the above |
| `week4/requirements.txt`, `week4/run_all.sh`, `week4/.gitignore` | Frozen environment and reproduction script |
| `week4/DECISIONS.md`, `week4/RESEARCH_LOG.md` | TabPFN verification record, D1.1 to D1.14, dated log |

## Findings

All numbers are from `week4/results/`.

**Data**

| Country | Survey year | Rows | Positives | Prevalence | Positives per outer test fold |
|---|---|---|---|---|---|
| Kenya | 2018 | 6,068 | 1,521 | 25.1% | 304 to 305 |
| Rwanda | 2016 | 8,735 | 1,003 | 11.5% | 200 to 201 |
| Tanzania | 2017 | 6,620 | 607 | 9.2% | 121 to 122 |
| Uganda | 2018 | 2,101 | 181 | 8.6% | 36 to 37 |

- No truly missing cells. Placeholder answers are rare, at most 3.2% (`job_type`, Uganda).
- **Tanzania's `location_type` runs the opposite way.** Positive rate is 0.207 rural and 0.048 urban. The other three countries are higher in urban areas. This could be swapped coding, but the data can't prove it.
- **`job_type` is coded very differently by country.** Uganda has no farming or informal-employment rows at all, for example. Distance between countries reaches 0.866 on a 0 to 1 scale. Models can partly infer country from it even with `country` dropped, which matters for leave-one-country-out (LOCO).

**TabPFN**

- **`TabPFNClassifier()` is not v2.** The package (9.2.0) defaults to v3.5, which is gated and non-commercial. v2 must be requested explicitly, and needs no login.
- **Every training set is over v2's 10,000-row limit:** 18,819 in outer folds, up to 21,423 in LOCO.
- **The obvious embedding calls leak labels.** On pure-noise labels, where a correct method should score about 0.5:

  | Embedding | AUC on noise |
  |---|---|
  | Train tokens | 1.000 |
  | Test tokens, with the row also in the context | 0.955 |
  | Out-of-fold | 0.520 mean (SD 0.030, 8 draws) |

  Out-of-fold is close to chance, but this does not prove leakage is exactly zero, so Phase 3's unit test checks it again.
- **Fine-tuning runs on CPU, but a Group DRO loss isn't clean.** It needs a private method, and batches carry no row groups.
- **Runtime:** 786 s per fold for the planned setting on this CPU. That gives an estimated 5.5 h for E2 and 4.6 h for E4 on TabPFN alone. H1 is about 3.2 h. H2 is about 8 h with 2 estimators, or 32 h with 8.

## Decisions and assumptions

To confirm or overrule (details in `DECISIONS.md`):

- **D1.2 Row limit:** each of the 8 estimators gets its own context of about 10,000 rows, stratified by country and label. All training rows get used across the ensemble. This is the only departure from "out of the box", and the paper reports it.
- **D1.6 H2:** 2 estimators for embeddings, to keep it to about 8 h. It would be compared with a parent using 8, and the paper says so.
- **D1.8 H3:** marked as future work, so E5c won't run.
- **D1.12 Compute:** CPU overnight runs, saving each fold as it finishes.
- **D1.13 Data:** Tanzania's `location_type` is left uncorrected and reported.

`HYPOTHESES.md` has not changed.

## Problems and risks

- **My error:** a stray process from a Phase 0 command competed for CPU during most of the first runtime ladder. I stopped it and reran the sizes that matter. Effect at 5,000 rows: 311.7 s against 306.6 s clean. Only timing was affected.
- **My error:** I earlier told you E2 would take about 7 hours. The estimate is 5.5 hours.
- **Runtime estimates are extrapolated** from single runs and have not been measured end to end. MLP and XGBoost time is extra. Memory use was not measured, and free RAM was low (1.3 GB of 15 GB) during the ladder, though nothing failed.
- **Long runs need the laptop awake and plugged in** for about 10 h or more.
- **Label definition and provenance are still open.** The repo calls the data "Findex-derived" but has no variable definitions, and the Findex report in the folder never mentions Rwanda.
- **Still open from Checkpoint 0:** the Group DRO source (below).

## Review checklist

1. `week4/results/feasibility.md`, section 2.
2. `week4/results/data_audit.md`, the last three sections.
3. `week4/DECISIONS.md`, Phase 1 table.
4. `week4/src/tabpfn_feasibility.py`: `section_embeddings`, `stratified_context_lists`.

To reproduce (Git Bash, about 1 hour):

```bash
uv venv week4/.venv --python 3.13
uv pip install --python week4/.venv/Scripts/python.exe -r week4/requirements.txt
uv pip install --python week4/.venv/Scripts/python.exe --no-deps -e .
bash week4/run_all.sh
```

## Suggested commit message

```text
Add Week 4 Phase 1: data audit and TabPFN feasibility checks
```

## Next

**Phase 2** reruns the baseline's XGBoost and MLP with its existing nested CV (E1, seed 42), writing to `week4/results/baseline_reproduction.csv` with a note on any mismatch, and without touching `reports/`. It also measures the effect of fitting preprocessing before CV. The original run took about 68 minutes in total `[VERIFY: reports/ablation_study_report.md]`.

**What I need from you:**

1. **Group DRO source:** add the Sagawa et al. (2020) PDF to `applied_research/literature/` (my recommendation), or tell me to describe the update as our own implementation. Needed by Phase 3.
2. **H3:** is future work acceptable (D1.8)? I recommend yes.
3. **Tanzania `location_type`:** do you know of any documentation of the coding? If not, I report it uncorrected (D1.13).
4. **Reply "continue to Phase 2"** when ready. Unless you say otherwise, that confirms D1.2, D1.6, D1.8, D1.12 and D1.13.
