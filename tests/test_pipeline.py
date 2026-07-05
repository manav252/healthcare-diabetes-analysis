import pytest

from src.data_processing import load_diabetes_data
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

