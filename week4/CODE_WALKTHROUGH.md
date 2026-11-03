# Code walkthrough

This explains the Group DRO implementation and the hybrid constructions in plain language, so you can defend them. Read it next to the code. It follows the order in which things happen.

| File | What it holds |
|---|---|
| `src/group_dro.py` | Group DRO: the weights, the batches, the training loop, the MLP |
| `src/tabpfn_tools.py` | TabPFN v2, context selection, leak-free embeddings |
| `src/hybrids.py` | H2: saved embeddings and the small head |
| `src/harness.py` | Folds, fold-only preprocessing, tuning, saving results |
| `src/arms.py` | Every model variant, described for the harness |
| `run_all.py` | The command you run |

## 1. The idea behind Group DRO

Ordinary training (ERM) minimises the average loss over all rows. A group that is small, or simply hard, can stay badly served, because improving it barely moves the average.

Group DRO instead aims to minimise the loss of the worst group:

```text
minimise over model θ:   max over groups g of   loss_g(θ) + C / sqrt(n_g)
```

The objective and the C/√n_g term are in `ghosal2023distributionally` (p.2-3) and `stahlschmidt2024domain` (p.5, Eq. 3 and 4).

Taking a hard `max` is unstable, because the worst group can flip from batch to batch. So the code keeps a weight per group, `q`, moves it gradually towards the groups with high loss, and trains on the weighted loss.

**Sourcing caveat.** The specific update rule below (exponentiated gradient on `q`) is what your brief asks for from Sagawa et al. (2020), but that paper is not in `applied_research/literature/`, and no paper in the folder gives the update. Until the PDF is added, the Methodology must present it as our own implementation statement (`paper/LITERATURE_GAPS.md`).

## 2. `GroupWeights`: the weights q

**`__init__`.**
- `q` starts uniform: with 4 countries, each gets 0.25.
- `adjustments` is `C / sqrt(n_g)`, one fixed number per group. It is larger for smaller groups, because a small group's measured loss is less certain. For Uganda's roughly 1,680 training rows and C = 2, it is about 0.05.

**`update(group_losses)`** moves `q`. In formula form:

```text
q_g  ←  q_g × exp(step_size × (loss_g + adjustment_g)),   then rescale so q sums to 1
```

- A group with higher loss gets a bigger multiplier, so its share of `q` grows.
- `step_size` says how fast `q` moves. At 0 it never moves. Large values make it snap to the current worst group.
- **It runs in log space.** My first version multiplied directly, and a unit test caught it producing NaN at high step size and loss, because `exp` overflowed. `log_softmax` gives the same result without overflow, and also keeps a nearly-zero group able to recover.
- `.detach()` means `q` is treated as a fixed number when the model's gradients are computed. The model is not rewarded for manipulating `q`.

**`weighted_loss(group_losses)`** is what the model minimises: the sum over groups of `q_g × (loss_g + adjustment_g)`.

## 3. `GroupBalancedBatches`: every group in every batch

The update needs a loss for every group on every step, so each batch takes the same number of rows from each group. With batch size 256 and 4 countries, that's 64 rows each.

- Each group's rows are shuffled and served in order, and reshuffled when used up. Uganda's rows therefore come round far more often than Rwanda's, and no row repeats before its group's rows have all been used.
- `steps_per_epoch` is `ceil(rows / batch_size)`, the same as an ordinary shuffled epoch. So Group DRO and ERM take the same number of updates per epoch.

## 4. `group_mean_losses`

Adds up the row losses per group, and divides by each group's row count in the batch. If a group is missing, it **raises an error** instead of returning 0, because a silent 0 would make that group lose weight for the wrong reason.

## 5. `train_group_dro`: one training step

```python
logits = call(*[tensor[index] for tensor in inputs])
row_losses = loss_fn(logits, y[index]) * sample_weight[index]
group_losses = group_mean_losses(row_losses, groups[index], n_groups)
weights.update(group_losses)
weights.weighted_loss(group_losses).backward()
optimizer.step()
```

1. Score the batch.
2. Compute each row's loss, times its class-imbalance weight, the same weights the ERM MLP uses.
3. Reduce to one loss per group.
4. Update `q` first, so the model steps on the newest weights.
5. Take a gradient step on the weighted loss.

It also saves `q` and each group's loss for every epoch. These go into each fold's results, so you can see which countries it actually focused on, rather than assuming.

## 6. `GroupDROMLP`

A subclass of the baseline `PyTorchMLP`. It inherits the network, input handling and prediction, and replaces only `fit`. So a difference from the ERM MLP comes from training, not from the model.

Inputs are the same embedding-index and scaled-numeric columns. Hyperparameters are the baseline's search space plus `step_size` and `adjustment_c`.

## 7. Reweighting (the simple comparison)

`country_weights` gives each row `N / (K × n_country)`, so every country has equal total weight and the mean weight is 1. The reweighted arms multiply this by the class weights. Reweighting is fixed before training; Group DRO's weights respond to losses during training. That's what RQ3 compares.

## 8. TabPFN and contexts

TabPFN doesn't train on your data. It reads a **context** (rows with labels), then predicts new rows from it. So choosing the context is the closest thing it has to choosing training data.

**`make_v2`** requests v2 explicitly, because the package default is v3.5. Everything else stays at default.

**`stratified_context_lists` (D1.2).** Training folds have about 18,800 rows, and v2 supports 10,000. So each of the 8 estimators gets its own list of about 10,000 rows, keeping each country-and-label group's share. `_deal` hands rows out from a shuffled group in turn, so lists differ and every row gets used somewhere in the ensemble. Phase 1 confirmed that an explicit list behaves exactly like subsetting before `fit`.

**H1, `balanced_context_lists` (D1.7).** Every country contributes the same number of rows to each context, set by the smallest country: about 1,680 per country in an outer fold, so about 6,700 in total. Within a country, its own positive rate is kept. So the only thing H1 changes is the mix of countries the model reads from. There's no training and no loss; it's "fairer data in the prompt".

## 9. H2: embeddings without leakage

An embedding is TabPFN's internal 192-number summary of a row. H2 trains a small head on these, three ways.

**The danger.** TabPFN builds a row's embedding with the context's labels in view. If a row is in its own context, its embedding contains its own label. Phase 1 measured this on pure-noise labels, where a correct method should score about 0.5:

| Method | AUC on noise |
|---|---|
| Train tokens | 1.000 |
| Test tokens, with the row also in the context | 0.955 |
| Out-of-fold | 0.520 mean |

A head trained on leaky embeddings would look excellent in training and fail on real rows.

**The fix, `out_of_fold_embeddings`.** Within an outer training fold:

1. Split the rows into 5 parts.
2. For each part, fit TabPFN on the other 4 and embed that part with `data_source="test"`.
3. Each row therefore gets an embedding from a model that never saw it.

**Validation rows** (`embed_new_rows`) use a context drawn from the whole outer training fold. They were never in it, so they're clean too. All contexts use the same row limit, so training and validation embeddings come from contexts of the same size.

**`fold_embeddings`** saves them per outer fold, so the three heads use exactly the same embeddings, and tuning doesn't rerun TabPFN.

**The head.** `EmbeddingHead` has one hidden layer. `fit_head` then either runs the same plain loop as the baseline MLP (ERM, or reweighted with different row weights), or calls the same `train_group_dro` from section 5.

**Known limitation.** During inner-loop tuning, a training row's embedding was made with a context that may include that inner split's validation rows. Their labels can therefore reach the head indirectly and flatter the inner score slightly. It can't touch the outer test rows, so reported results aren't affected, but tuning could be mildly affected.

**H3 is not implemented** (D1.8). `FinetunedTabPFNClassifier` batches carry no row groups, so Group DRO would need changes to the package's data pipeline.

## 10. The harness

Every arm goes through `run_arm`:

1. **Folds.** `outer_folds` gives every arm the same folds for a seed, or `loco_folds` holds out one country.
2. **Context.** For H2, make or load the embeddings.
3. **Tune (`tune`).** An Optuna search scored on inner out-of-fold predictions, using the outer training rows only. Preprocessing is refitted inside each inner fold.
4. **Fit and predict.** The arm gets validation **features with the label column removed**, so it can't reach those labels. A test checks this.
5. **Save.** Scores go to `results/oof/<design>/<arm>/seed<n>/<fold>.csv.gz`, and parameters, timing and `q` history to a `.json` beside it.

The JSON is written last, and its existence marks a fold as finished. So if a session drops, rerunning the same command resumes.

**Selection.** Arms used in E2 select on pooled inner PR-AUC. The E3 MLP arms all select on worst-country inner PR-AUC (D0.4), so selection can't favour Group DRO.

**Sweeps.** Arms such as `mlp_dro_step_0.1` reuse `mlp_dro`'s tuned parameters for that fold and change one setting. Nothing is tuned, and they're reported as ablations, not as candidates.

**Devices.** Only TabPFN uses `--device`. XGBoost and the MLPs stay on CPU, because the baseline `PyTorchMLP` is CPU-only, and one device per model keeps seeds and arms comparable.

## 11. Design points for you to decide at Checkpoint 3

1. **Group DRO may focus on the wrong country.** Class weights are pooled, so positives weigh about 3.55 and negatives about 0.58 `[VERIFY: derived from 3,312 positives in 23,524 rows, results/per_country_counts.csv]`. Kenya has the most positives, so its weighted loss may be the highest even though Rwanda has the lowest PR-AUC. `q` could then push towards Kenya. The saved `q` history will show this. E7's country-by-label groups partly address it. An extra arm with class weights computed within each country would address it directly, but I haven't added it, because that's a design choice you should make before seeing results.
2. **"Identical class-imbalance handling" changes XGBoost.** The baseline's best XGBoost tuned `scale_pos_weight`, while its MLP used balanced weights. To follow your brief, the primary `xgb` uses balanced weights like the MLP, and `xgb_spw` keeps the baseline's method as a reported sensitivity arm. If `xgb_spw` scores higher, RQ1's comparison is against the weaker XGBoost, and the paper will show both. TabPFN takes no row weights, so it stays at defaults.
3. **Tuning budgets are unequal.** I kept the baseline's 15 trials for XGBoost and 8 for MLPs. `mlp_dro` searches 2 more hyperparameters in those same 8 trials, which works slightly against it. The step-size sweep gives some protection.
4. **No early stopping.** `koh2021wilds` (p.67) says methods like Group DRO work only while training loss hasn't vanished, achieved through early stopping. The baseline MLP has none, and I kept 20 epochs for every arm so they stay comparable. Weight decay is tuned, and the saved per-group losses will show whether loss vanishes.

## 12. Added for the GPU run

- **Point 1 of section 11, decided (D3.3).** A second pair of E3 arms uses class weights balanced inside each country: `mlp_erm_within_cw` and `mlp_dro_within_cw`. Each country keeps its total weight, and its two classes weigh the same within it (`within_country_class_weights` in `harness.py`). Both arms use the same weights, so the pair differs only in the objective.
- **`run_everything.sh`** runs the checks first and stops if any fail. It then runs a GPU lane (TabPFN and hybrids) and CPU lanes (XGBoost and MLPs, one per seed) side by side, then E6, then the analysis.
- **`analysis.py`.**
  - `weighted_ap` computes PR-AUC with each bootstrap replicate's draw counts as row weights, so 2,000 replicates need one sort per model, not 2,000. A test checks it against sklearn.
  - `bootstrap_counts` resamples within country-by-label strata, and every model uses the same counts, so every comparison is paired.
  - `paired` gives a − b, or a minus the better of several models, chosen inside each replicate (RQ4).
- **`e6_shap.py`.**
  - Each outer fold's model is refitted from its saved parameters, and the refit's scores are compared with the saved ones.
  - XGBoost is explained with TreeSHAP. Both MLPs use Week 3's exact Shapley values over the raw features.
  - Rankings exclude `country`. The tests compare size-matched country subsamples with random halves of one country.
