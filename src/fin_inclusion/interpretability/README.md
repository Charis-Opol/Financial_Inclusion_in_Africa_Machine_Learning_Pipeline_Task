# `interpretability/` — how the explanation code fits together

This package answers one question about a model: **for this respondent, how
much did each of the 10 survey fields push the score up or down?** It answers
it for two models (the shipped XGBoost and a PyTorch MLP) with several
methods, in a form where the answers can be compared.

Explanations describe what a **model** learned to rely on. They are not
claims about what causes people to have bank accounts.

## The flow

```text
raw survey rows (10 fields)
        │
        ▼
  model adapter ──────────────► logit (the model's raw score)
  model_adapters.py
        │
        ▼
  explainer  ─────────────────► Attribution  (rows × columns of contributions)
  base.py (interface)           attribution.py
        │
        ▼
  grouping  ──────────────────► Attribution over the 10 raw features
  grouping.py
        │
        ▼
  checks  ────────────────────► do the contributions add up to the score?
  checks.py
```

`sampling.py` sits beside this: it fixes **which rows** are explained and
which rows serve as the reference, so every method uses the same ones.

## What each file is for

| File | One-line purpose |
|---|---|
| `model_adapters.py` | Gives both models the same call: raw rows in, logit out. Hides that XGBoost wants 40 one-hot columns and the MLP wants embedding indices. |
| `base.py` | The `Explainer` interface every method implements: `explain(X_raw) -> Attribution`. |
| `attribution.py` | The `Attribution` result: the numbers, plus which model, method and reference they belong to. Saves to and loads from `.npz`. |
| `grouping.py` | Adds encoded columns back together into the 10 raw features, and turns importances into shares. |
| `checks.py` | Verifies that contributions + base value = the model's score. |
| `sampling.py` | Picks the background (reference) rows and the rows to explain. |
| `shap_analysis.py` | The original Phase 4 SHAP plots. Left as it was; not used by the newer code. |

## Words used throughout

- **Logit** — the model's score before it is squashed into 0–1. All
  explanations are in logits, because contributions add up there and do not
  add up in probabilities.
- **Score** — the sigmoid of the logit, between 0 and 1. It ranks
  respondents; it is *not* a calibrated probability (both models were
  trained with class weighting).
- **Attribution** — one feature's contribution to one respondent's logit.
  Positive pushes towards "has an account", negative pushes away.
- **Baseline / reference / background** — what a respondent is compared
  with. "This field added +0.4" always means "+0.4 compared with the
  reference". Change the reference and the numbers change.
- **Base value** — the model's logit at the reference. Base value + all
  contributions = the respondent's logit.
- **Efficiency / completeness** — the property in the line above. The first
  name is used for SHAP, the second for Integrated Gradients; `checks.py`
  tests it for both.
- **Raw feature** — one of the 10 survey fields. An **encoded column** is
  what a model actually receives (for example `country_Uganda`, or one
  dimension of an embedding).
- **Pooled / focus** — all four countries together / Uganda only.

## Reading order

1. `attribution.py` — the result everything else produces or consumes.
2. `model_adapters.py` — how a model is asked for a score.
3. `grouping.py`, then `checks.py`.
4. `sampling.py`.
5. `scripts/run_xai_models.py` — the first script that uses them.

## Scripts and what they write

| Script | Writes |
|---|---|
| `scripts/run_xai_models.py` | `reports/xai/model_verification.json` (do the models perform as reported?), `reports/xai/sample_sets.json` (which rows), `models/xai/` (the retrained MLP) |

Run them from the XAI environment (see `requirements-xai.txt`):

```bash
.venv-xai/Scripts/python scripts/run_xai_models.py
```

Every choice that was not obvious, and why it was made, is in
`reports/xai_decisions_log.md`.
