"""Dashboard tests: the service on synthetic models, the charts, and a headless
run of the real app (which needs the real models, data and artifacts)."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic", minversion="2", reason="the dashboard validates with the pydantic v2 serving schema")
pytest.importorskip("altair", reason="the dashboard draws with altair")
pytest.importorskip("captum", reason="the dashboard offers the Captum methods")

from fin_inclusion.dashboard import charts  # noqa: E402
from fin_inclusion.dashboard.service import (  # noqa: E402
    METHODS,
    ExplanationService,
    InvalidRecord,
    allowed_values,
    validate_record,
)
from fin_inclusion.interpretability.captum_explainers import BASELINES, baseline_codes  # noqa: E402
from fin_inclusion.interpretability.checks import check_efficiency  # noqa: E402
from fin_inclusion.interpretability.model_adapters import XGBoostRawModel  # noqa: E402
from fin_inclusion.preprocessing.pipeline_factory import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS  # noqa: E402

RAW_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
ROOT = Path(__file__).resolve().parent.parent
# Levels present in the synthetic survey (conftest.py), so the tiny models know them.
RECORD = {
    "country": "Uganda", "location_type": "Rural", "cellphone_access": "Yes", "household_size": 5,
    "age_of_respondent": 34, "gender_of_respondent": "Female", "relationship_with_head": "Spouse",
    "marital_status": "Married/Living together", "education_level": "Primary education",
    "job_type": "Self employed",
}


@pytest.fixture(scope="module")
def service(xgb_artifact, mlp_fitted):
    model_dir, survey, _, _ = xgb_artifact
    models = {"xgboost": XGBoostRawModel.load(model_dir), "mlp": mlp_fitted[0]}
    training_codes = models["mlp"].to_codes(survey[RAW_FEATURES])
    baselines = {key: {kind: baseline_codes(m, training_codes, kind) for kind in BASELINES} for key, m in models.items()}
    return ExplanationService(models, survey[RAW_FEATURES].iloc[:40], baselines, ig_steps=64,
                              training_max={"household_size": 18.0, "age_of_respondent": 89.0})


# --- Validation (same rules as the API) ---------------------------------------------------


def test_validation_accepts_a_valid_record_and_returns_it_clean():
    assert validate_record(RECORD) == RECORD


@pytest.mark.parametrize("change, field", [
    ({"country": "uganda"}, "country"),            # unseen spelling: rejected, never encoded as "unknown"
    ({"household_size": 3.5}, "household_size"),   # strict int
    ({"household_size": 0}, "household_size"),    # below the schema's range
    ({"age_of_respondent": "34"}, "age_of_respondent"),
    ({"favourite_colour": "blue"}, "favourite_colour"),  # unknown field
])
def test_validation_rejects_what_the_api_rejects_and_names_the_field(change, field):
    with pytest.raises(InvalidRecord) as error:
        validate_record({**RECORD, **change})
    assert any(e.startswith(field) for e in error.value.errors)


def test_allowed_values_are_the_api_vocabulary():
    values = allowed_values()

    assert set(values) == set(CATEGORICAL_COLUMNS)
    assert "Dont know" in values["marital_status"]


# --- Explanations ------------------------------------------------------------------------


def test_methods_offered_per_model():
    assert set(METHODS) == {"tree_shap", "exact_shap", "integrated_gradients", "feature_ablation"}


@pytest.mark.parametrize("model_key", ["xgboost", "mlp"])
def test_every_offered_method_explains_one_respondent(service, model_key):
    for method in service.methods_for(model_key):
        explanation = service.explain(RECORD, model_key, method, "modal")

        assert explanation.attribution.values.shape == (1, 10)
        assert 0 < explanation.score < 1
        assert explanation.predicted_yes == (explanation.score >= explanation.threshold)
        table = explanation.table()
        assert list(table.columns) == ["field", "value", "contribution"]
        assert (table["contribution"].abs().diff().dropna() <= 1e-12).all()  # largest first
        if method != "feature_ablation":  # ablation is not additive, by design
            assert check_efficiency(explanation.attribution, np.array([explanation.logit]), atol=1e-2).passed


def test_methods_not_available_for_a_model_are_refused(service):
    assert "integrated_gradients" not in service.methods_for("xgboost")
    assert "tree_shap" not in service.methods_for("mlp")
    with pytest.raises(ValueError, match="not available"):
        service.explain(RECORD, "xgboost", "integrated_gradients")
    with pytest.raises(KeyError, match="Unknown model"):
        service.explain(RECORD, "random_forest", "exact_shap")


def test_invalid_input_is_refused_before_any_model_is_called(service):
    with pytest.raises(InvalidRecord):
        service.explain({**RECORD, "country": "Narnia"}, "xgboost", "tree_shap")


def test_baseline_matters_only_for_baseline_methods(service):
    ablation_modal = service.explain(RECORD, "mlp", "feature_ablation", "modal")
    ablation_unknown = service.explain(RECORD, "mlp", "feature_ablation", "unknown")
    shap_modal = service.explain(RECORD, "mlp", "exact_shap", "modal")
    shap_unknown = service.explain(RECORD, "mlp", "exact_shap", "unknown")

    assert not np.allclose(ablation_modal.attribution.values, ablation_unknown.attribution.values)
    np.testing.assert_array_equal(shap_modal.attribution.values, shap_unknown.attribution.values)
    assert shap_modal.baseline is None and ablation_modal.baseline == "modal"


def test_out_of_range_household_size_carries_a_warning(service):
    explanation = service.explain({**RECORD, "household_size": 25}, "xgboost", "tree_shap")

    assert any("household_size" in w and "extrapolation" in w for w in explanation.warnings)


def test_what_if_changes_one_field_only(service):
    before, after = service.what_if(RECORD, "cellphone_access", "No", "xgboost", "exact_shap")

    assert before.record["cellphone_access"] == "Yes" and after.record["cellphone_access"] == "No"
    assert {k: v for k, v in after.record.items() if k != "cellphone_access"} == \
           {k: v for k, v in before.record.items() if k != "cellphone_access"}


def test_compare_methods_has_one_column_per_available_method(service):
    frame = service.compare_methods(RECORD, "mlp")

    assert list(frame.columns) == [METHODS[m].label for m in service.methods_for("mlp")]
    assert list(frame.index) == RAW_FEATURES


# --- Charts --------------------------------------------------------------------------------


def test_charts_build_valid_specifications(service):
    import pandas as pd

    explanation = service.explain(RECORD, "mlp", "exact_shap")
    before, after = service.what_if(RECORD, "country", "Kenya", "mlp", "exact_shap")
    importance = pd.DataFrame({"sample": ["pooled", "uganda"] * 2, "field": ["a", "a", "b", "b"],
                               "mean_abs": [0.5, 0.4, 0.2, 0.3], "ci_low": [0.4, 0.3, 0.1, 0.2],
                               "ci_high": [0.6, 0.5, 0.3, 0.4], "n": [1000, 2101, 1000, 2101]})
    shares = importance.rename(columns={"mean_abs": "share"}).assign(method="m")

    for chart in (charts.contribution_chart(explanation.table(), "t"), charts.importance_chart(importance, "t"),
                  charts.share_chart(shares, "t"), charts.what_if_chart(before.table(), after.table(), "t"),
                  charts.method_heatmap(service.compare_methods(RECORD, "mlp"), "t")):
        spec = chart.to_dict()
        assert "$schema" in spec


# --- The real app, headless -------------------------------------------------------------------


def _real_artifacts_present() -> bool:
    needed = [ROOT / "models" / "production" / "model.json", ROOT / "models" / "xai" / "mlp.pt",
              ROOT / "data" / "raw" / "Train_v2.csv", ROOT / "data" / "raw" / "Test_v2.csv",
              ROOT / "reports" / "xai" / "comparison_table.csv"]
    return all(path.exists() for path in needed)


def test_the_app_runs_headlessly_for_every_model_and_method():
    pytest.importorskip("streamlit", exc_type=ImportError, reason="streamlit is needed to run the app")
    if not _real_artifacts_present():
        pytest.skip("needs the real models, data/raw/ and the XAI artifacts (run scripts/run_xai_*.py)")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src" / "fin_inclusion" / "dashboard" / "app.py"), default_timeout=900)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert [t.label for t in app.tabs] == ["This respondent", "What if?", "Compare methods", "Global importance",
                                           "Method agreement"]
    assert any("not causes" in w.value or "not what causes" in w.value for w in app.warning)  # limitations box

    for model_key in ("xgboost", "mlp"):
        app.selectbox(key="model").set_value(model_key).run()
        selector = app.selectbox(key=f"method_{model_key}")
        for index in range(len(selector.options)):
            app.selectbox(key=f"method_{model_key}").select_index(index).run()
            assert not app.exception, (model_key, index, [e.value for e in app.exception])
            assert app.metric[0].label == "Score"

    app.radio(key="source").set_value("Enter manually").run()
    app.number_input(key="household_size").set_value(30).run()
    assert not app.exception
    assert any("extrapolation" in w.value for w in app.warning)
