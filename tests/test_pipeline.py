import pytest

from src.data_processing import load_diabetes_data
from src.evaluation import subgroup_performance, threshold_analysis
from src.external_validation import evaluate_external_dataset, load_external_validation_data
from src.modeling import compare_models
from src.prediction import patient_to_frame, predict_patient
from src.preprocessing import NUMERIC_FEATURES, replace_invalid_zeros
from src.validation import validate_diabetes_data


def test_validation_reports_invalid_zero_counts():
    result = validate_diabetes_data(load_diabetes_data())
    assert result.row_count == 768
    assert result.duplicate_rows == 0
    assert result.invalid_zero_counts["Insulin"] > 0


def test_model_comparison_selects_best_model_without_tuning():
    df = replace_invalid_zeros(load_diabetes_data())
    result = compare_models(df, tune=False)
    assert not result.metrics.empty
    assert result.best_model_name in result.fitted_models
    assert result.metrics.iloc[0]["roc_auc"] >= result.metrics.iloc[-1]["roc_auc"]


def test_patient_prediction_returns_probability():
    df = replace_invalid_zeros(load_diabetes_data())
    result = compare_models(df, tune=False)
    patient = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 72,
        "SkinThickness": 23,
        "Insulin": 80,
        "BMI": 31.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 35,
    }
    prediction = predict_patient(result.best_estimator, patient)
    assert 0 <= prediction["probability"] <= 1
    assert prediction["risk_label"] in {"Lower Risk", "Higher Risk"}


def test_patient_to_frame_requires_all_features():
    with pytest.raises(ValueError):
        patient_to_frame({feature: 1 for feature in NUMERIC_FEATURES[:-1]})


def test_threshold_and_subgroup_reports_are_generated():
    df = replace_invalid_zeros(load_diabetes_data())
    result = compare_models(df, tune=False)
    thresholds = threshold_analysis(result.best_estimator, result.x_test, result.y_test)
    subgroups = subgroup_performance(result.best_estimator, result.x_test, result.y_test)

    assert {"threshold", "recall", "precision"}.issubset(thresholds.columns)
    assert {"subgroup_type", "subgroup", "recall"}.issubset(subgroups.columns)
    assert thresholds["recall"].between(0, 1).all()


def test_external_validation_requires_expected_schema(tmp_path):
    invalid_path = tmp_path / "external.csv"
    invalid_path.write_text("Glucose,Outcome\n120,1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_external_validation_data(invalid_path)


def test_external_validation_metrics_on_matching_schema():
    df = replace_invalid_zeros(load_diabetes_data())
    result = compare_models(df, tune=False)
    external_metrics = evaluate_external_dataset(
        result.best_estimator,
        df[NUMERIC_FEATURES + ["Outcome"]].head(50),
    )

    assert external_metrics["model"] == "External Validation"
    assert 0 <= external_metrics["roc_auc"] <= 1
