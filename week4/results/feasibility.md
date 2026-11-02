# TabPFN feasibility

Phase 1 toy checks, run on this machine. Every number comes from `week4/results/runs/phase1_tabpfn_feasibility.json`, produced by `week4/src/tabpfn_feasibility.py`, except the budget estimates in section 3, which are derived from those numbers and labelled as estimates.

**No check measures TabPFN's performance on the study's labels.** Behaviour checks use synthetic data. Runtime checks use real feature rows with randomly permuted labels and record only seconds.

## 1. Version, licence, limits

| Item | Finding |
|---|---|
| Package | `tabpfn` 9.2.0, code under Apache-2.0 |
| Package default model | **v3.5, not v2.** A plain `TabPFNClassifier()` would not be TabPFN v2. |
| How to get v2 | `TabPFNClassifier.create_default_for_version(ModelVersion.V2)` |
| v2 checkpoint | `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, 29.0 MB |
| Login or licence acceptance for v2 | **None.** It downloaded without a prompt, and the Hugging Face repo is not gated. |
| v2 weights licence | Prior Labs License 1.1: Apache-2.0 plus an attribution paragraph. Distributing the weights, or a product containing them, requires displaying "Built with PriorLabs-TabPFN". Internal benchmarking is exempt. |
| Later versions | v2.5, v2.6, v3 and v3.5 go through browser licence acceptance in the code. The v3.5 terms bar commercial and production use. Not used here. |
| Hardware | 8 torch threads, no CUDA, about 15 GB RAM |

Limits enforced by the installed code for v2:

| Check | Result |
|---|---|
| CPU, 1,000 rows, default settings | Accepted |
| CPU, 1,001 rows, default settings | Refused: more than 1,000 samples on CPU |
| CPU allowed, 10,000 rows | Accepted |
| CPU allowed, 10,001 rows | Refused: above the supported 10,000 samples |
| CPU allowed, 501 features | Refused: above the supported 500 features |
| 11 classes, limits ignored | Refused: above 10 classes, and this one cannot be overridden |
| 10,001 rows, `ignore_pretraining_limits=True` | Accepted |

Against the study (`fold_sizes.csv`):

| Design | Training rows | Over the 10,000 limit? |
|---|---|---|
| Outer folds | 18,819 to 18,820 | Yes, 1.88 times |
| Inner folds | 12,546 to 12,547 | Yes, 1.25 times |
| Leave-one-country-out | 14,789 to 21,423 | Yes, up to 2.14 times |

Features are not a constraint: 11 columns reach TabPFN, or 40 for the XGBoost one-hot encoding. There are 2 classes.

## 2. Hybrid checks

### Embeddings (H2): feasible, but only out-of-fold

`model.get_embeddings(X, data_source=...)` exists and returns `(n_estimators, n_rows, 192)`.

**The leakage test.** Labels are pure noise, so any skill means leakage. The table shows the AUC of a logistic head on embeddings under 5-fold CV, on 600 rows with 1 estimator.

| How the embedding was made | AUC on noise labels |
|---|---|
| A. `data_source="train"` (the row's own label is in its context) | **1.000** |
| B. `data_source="test"`, with the row also in the context | **0.955** |
| C. Out-of-fold: `data_source="test"`, row excluded from context | 0.550 |

Both A and B leak badly. Option B is the easy mistake to make, because it looks like the safe "test" setting.

Repeating C over 8 further noise draws gives a mean AUC of **0.520 (SD 0.030)**. The same head on raw features gives 0.486 (SD 0.040). This is far from A and B and close to chance, but it does not prove the leakage is exactly zero: a mean of 0.520 with that SD is about 1.9 standard errors above 0.5. Phase 3's unit test checks this again.

Other findings:

- **Label reaches its own row.** Flipping one label moved that row's train embedding by L2 1.756. Other rows moved a median of 0.094, and test rows 0.074.
- **`data_source="train"` ignores `X`.** Passing 50 new rows still returned all 600 context rows. It would misalign silently if used carelessly.
- **The embeddings carry signal.** On synthetic data with real signal, a head on out-of-fold embeddings reached AUC 0.949, against 0.968 for TabPFN's own predictions on held-out rows. These use different row sets, so read them only as a sanity check.

### Context control (H1 and the row limit): feasible

- Passing a subset of rows to `fit` sets the context exactly, and changes predictions compared with the full data.
- `inference_config={"SUBSAMPLE_SAMPLES": [...]}` accepts one explicit row list per estimator. With a single list it matched subsetting before `fit` exactly (maximum difference 0.0).
- Built-in methods exist too (`balanced`, `stratified`, `majority_downsample`), but `stratified` preserves class only, not country.

### Fine-tuning with a custom loss (H3): runs, but not clean

- `tabpfn.finetuning.FinetunedTabPFNClassifier` runs on CPU with v2: 37.0 s for one epoch on 250 synthetic rows, 60.3 s on 2,500 rows, with 1 estimator and one training step in each.
- The loss can be replaced only by overriding a private method, `_forward_with_loss`. The override was called as expected.
- **A batch carries no row IDs or groups.** Its fields are `X_context`, `X_query`, `y_context`, `y_query`, `cat_indices`, `configs`. Group DRO needs the group of each query row, so this would mean changing the package's data pipeline, not just the loss.
- Library defaults assume a GPU and many steps: `device="cuda"`, 30 epochs, 50,000 samples per dataset.

## 3. CPU runtime

One fit and predict, 8 estimators (the default), on real feature rows with permuted labels.

| Context rows | Rows predicted | Seconds | Note |
|---|---|---|---|
| 1,000 | 4,705 | 77.4 | Clean rerun |
| 2,000 | 4,705 | 117.6 | Clean rerun |
| 5,000 | 4,705 | 306.6 | Clean rerun |
| 10,000 | 4,705 | 745.4 | Clean |
| **18,819 rows, 8 stratified contexts of 9,998** | **4,705** | **786.2** | **Planned E2 setting. All 18,819 rows used by at least one estimator.** |

Other steps from the first ladder are in the JSON. A stray process of mine competed for CPU during steps 1 to 6 and most of step 7 of that ladder. The effect looks small: 311.7 s against 306.6 s clean at 5,000 rows. Memory was not measured; the planned run finished without error.

**Estimated TabPFN compute** (estimates, derived from the times above, not measured end to end):

| Experiment | Fits | Basis | Estimate |
|---|---|---|---|
| E2 TabPFN | 25 | 786 s per fit | 5.5 h |
| E4 TabPFN | 20 | 786 s adjusted for each held-out country's size | 4.6 h |
| **Priority A total** | | | **about 10 h** |
| E5a H1 | 25 | About 6,700 context rows, interpolated to 458 s | 3.2 h |
| E5b H2, 2 estimators | 25 | 5 out-of-fold passes plus test rows | 8 h |
| E5b H2, 8 estimators | 25 | Same | 32 h |

The MLP and XGBoost runs are additional and are timed in Phase 2. So CPU is workable for priority A as overnight runs, if the machine stays awake and each fold's output is saved as it finishes. H2 at 8 estimators is not practical on this machine.
