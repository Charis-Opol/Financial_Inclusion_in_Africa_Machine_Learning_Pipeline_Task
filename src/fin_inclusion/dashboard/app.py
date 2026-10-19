"""Streamlit dashboard: explain either model's score for one respondent.

Run from the repo root, in the XAI environment:

    .venv-xai/Scripts/streamlit run src/fin_inclusion/dashboard/app.py

This file only lays out the page. Validation, scoring and explaining live in
`service.py`; charts in `charts.py`. If anything the dashboard depends on is
missing (data, a model, an analysis artifact) it stops with the reason
instead of showing partial results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "config"))
from settings import load_settings  # noqa: E402

from fin_inclusion.dashboard import charts  # noqa: E402
from fin_inclusion.dashboard.service import (  # noqa: E402
    BASELINE_LABELS,
    LIMITATIONS,
    METHODS,
    MODEL_LABELS,
    ExplanationService,
    InvalidRecord,
    SavedResults,
    allowed_values,
    held_out_examples,
)

FIELDS = ["country", "location_type", "cellphone_access", "household_size", "age_of_respondent",
          "gender_of_respondent", "relationship_with_head", "marital_status", "education_level", "job_type"]
NUMERIC_RANGES = {"household_size": (1, 30), "age_of_respondent": (16, 100)}  # the API schema's bounds
N_HELD_OUT = 30
SOURCES = ["Held-out respondent (unlabelled test set)", "Rule-chosen training example", "Enter manually"]

st.set_page_config(page_title="Explaining account-ownership scores", layout="wide")


@st.cache_resource(show_spinner="Loading the models, data and saved results...")
def load():
    from fin_inclusion.interpretability.workspace import load_workspace

    settings = load_settings()
    ws = load_workspace(settings.paths.raw_train, settings.paths.raw_test, ROOT / "models" / "production",
                        settings.xai.model_dir, settings.xai.artifacts_dir)
    captum = json.loads((settings.xai.artifacts_dir / "captum_summary.json").read_text(encoding="utf-8"))
    service = ExplanationService.from_workspace(ws, captum["integrated_gradients"]["n_steps"]["chosen"])
    saved = SavedResults(settings.xai.artifacts_dir, settings.xai.figures_dir)
    return service, saved, held_out_examples(settings.paths.raw_test, N_HELD_OUT, settings.seed)


try:
    service, saved, held_out = load()
except Exception as exc:  # fail loudly: no page is better than a page built on missing pieces
    st.error(f"The dashboard cannot start: {type(exc).__name__}: {exc}")
    st.stop()

choices = allowed_values()
training_examples = saved.local_examples()


def load_example(record: dict) -> None:
    for name in FIELDS:
        st.session_state[name] = record[name]


# --- Sidebar: respondent, model, method, baseline -------------------------------------

with st.sidebar:
    st.header("Respondent")
    source = st.radio("Start from", SOURCES, key="source")
    if source == SOURCES[0]:
        index = st.selectbox(
            "Held-out respondent", range(len(held_out)), key="held_out_index",
            format_func=lambda i: f"#{i + 1}: {held_out[i]['country']}, {held_out[i]['location_type']}, "
                                  f"{held_out[i]['education_level']}",
            help="From Test_v2.csv: never seen in training, and its true answer is not available.",
        )
        chosen = held_out[index]
    elif source == SOURCES[1]:
        name = st.selectbox("Training example", list(training_examples), key="training_example",
                            format_func=lambda n: n.replace("_", " "))
        st.caption(f"Rule: {training_examples[name]['rule']}. Actual: "
                   f"{'has an account' if training_examples[name]['has_account'] else 'no account'}. "
                   "A training row, so the models have seen it.")
        chosen = training_examples[name]["features"]
    else:
        chosen = None
    if chosen is not None and st.session_state.get("loaded") != (source, str(chosen)):
        load_example(chosen)
        st.session_state["loaded"] = (source, str(chosen))

    for name in FIELDS:
        if name in NUMERIC_RANGES:
            low, high = NUMERIC_RANGES[name]
            st.number_input(name.replace("_", " "), min_value=low, max_value=high, step=1, key=name)
        else:
            st.selectbox(name.replace("_", " "), choices[name], key=name)

    st.header("Explanation")
    model_key = st.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get, key="model")
    method = st.selectbox("Method", service.methods_for(model_key), format_func=lambda m: METHODS[m].label,
                          key=f"method_{model_key}")
    st.caption(METHODS[method].description)
    baseline = st.radio("Baseline (IG and Ablation only)", list(BASELINE_LABELS), format_func=BASELINE_LABELS.get,
                        key="baseline", disabled=not METHODS[method].uses_baseline,
                        help="SHAP methods compare with 200 reference respondents instead of one baseline point.")

record = {name: st.session_state[name] for name in FIELDS}

# --- Main page ---------------------------------------------------------------------------

st.title("Why does the model score this respondent the way it does?")
st.warning(LIMITATIONS)

try:
    explanation = service.explain(record, model_key, method, baseline)
except InvalidRecord as exc:
    st.error("This input is rejected by the prediction API's schema:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
    st.stop()

reference = (f"compared with {len(service.background)} reference respondents" if not METHODS[method].uses_baseline
             else f"compared with one baseline respondent ({BASELINE_LABELS[baseline].split(':')[0].lower()})")
score_column, threshold_column, prediction_column = st.columns(3)
score_column.metric("Score", f"{explanation.score:.3f}", help="A ranking score, not a calibrated probability.")
threshold_column.metric("Decision threshold", f"{explanation.threshold:.4f}",
                        help="0.4551 is the shipped XGBoost threshold; the MLP uses its own (0.6824).")
prediction_column.metric("Prediction", "Has an account" if explanation.predicted_yes else "No account")
for warning in explanation.warnings:
    st.warning(warning)

tab_local, tab_what_if, tab_methods, tab_global, tab_agreement = st.tabs(
    ["This respondent", "What if?", "Compare methods", "Global importance", "Method agreement"]
)

with tab_local:
    table = explanation.table()
    st.altair_chart(charts.contribution_chart(table, f"{MODEL_LABELS[model_key]}: {METHODS[method].label}"),
                    width="stretch")
    note = ("Contributions add up to the respondent's logit minus the reference's." if method != "feature_ablation"
            else "Feature Ablation contributions are measured one field at a time and do not add up.")
    st.caption(f"Contributions in logits, {reference}. Reference logit {explanation.attribution.base_value:+.3f}; this "
               f"respondent {explanation.logit:+.3f}. {note}")
    st.dataframe(table.style.format({"contribution": "{:+.3f}"}), hide_index=True, width="stretch")

with tab_what_if:
    st.write("Change one answer and see how the score and the explanation move.")
    left, right = st.columns(2)
    field_name = left.selectbox("Answer to change", FIELDS, format_func=lambda f: f.replace("_", " "), key="what_if_field")
    if field_name in NUMERIC_RANGES:
        low, high = NUMERIC_RANGES[field_name]
        new_value = right.number_input("New value", min_value=low, max_value=high, step=1,
                                       value=int(record[field_name]), key=f"what_if_{field_name}")
    else:
        options = list(choices[field_name])
        new_value = right.selectbox("New value", options, index=options.index(record[field_name]),
                                    key=f"what_if_{field_name}")
    before, after = service.what_if(record, field_name, new_value, model_key, method, baseline)
    first, second, third = st.columns(3)
    first.metric("Score before", f"{before.score:.3f}")
    second.metric("Score after", f"{after.score:.3f}", delta=f"{after.score - before.score:+.3f}")
    third.metric("Prediction after", "Has an account" if after.predicted_yes else "No account")
    st.altair_chart(charts.what_if_chart(before.table(), after.table(),
                                         f"{field_name.replace('_', ' ')}: {record[field_name]} -> {new_value}"),
                    width="stretch")
    st.caption("A what-if shows how the MODEL responds to a changed answer. It does not say what would happen to "
               "the person if their circumstances changed.")

with tab_methods:
    comparison = service.compare_methods(record, model_key, baseline)
    st.altair_chart(charts.method_heatmap(comparison, f"This respondent under every method ({MODEL_LABELS[model_key]})"),
                    width="stretch")
    st.caption(f"SHAP methods: {len(service.background)} reference respondents. Integrated Gradients and Ablation: "
               f"the {baseline} baseline. Different references are the main reason methods disagree on one person "
               "(see the 'Method agreement' tab).")

with tab_global:
    st.altair_chart(charts.importance_chart(saved.importance_with_ci(model_key),
                                            f"How much each field matters on average ({MODEL_LABELS[model_key]})"),
                    width="stretch")
    st.caption("Mean |contribution| over the pooled sample (1,000 respondents) and all Uganda respondents (2,101), "
               "TreeSHAP for XGBoost and exact SHAP for the MLP, with 95% bootstrap intervals. Uganda is compared "
               "with the pooled reference, not a Uganda-only one.")
    st.altair_chart(charts.share_chart(saved.importance_shares(model_key), "Share of importance under every method"),
                    width="stretch")

with tab_agreement:
    st.write("How closely the methods agree, measured on the saved explanations (Phase 6 of the analysis).")
    table = saved.comparison_table()
    sample = st.radio("Sample", ["pooled", "uganda"], horizontal=True, key="agreement_sample")
    columns = ["id", "comparison", "global_spearman", "instance_spearman_median_excluding_shared_zeros",
               "same_top_feature_share", "sign_agreement_share"]
    st.dataframe(table[table["sample"] == sample][columns].rename(columns={
        "global_spearman": "global rank corr.",
        "instance_spearman_median_excluding_shared_zeros": "per-respondent rank corr. (median)",
        "same_top_feature_share": "same top field", "sign_agreement_share": "same sign"}),
        hide_index=True, width="stretch")
    figure = saved.figure("comparison_forest__instance_no_shared_zeros_rank_correlation.png")
    if figure is not None:
        st.image(str(figure))
    st.caption("Agreement is not correctness. A, R and M rows compare methods on the MLP (R changes only the "
               "reference, M only the method); B rows compare the two models.")
