from src.data_processing import clean_diabetes_data, load_diabetes_data
from src.feature_engineering import add_risk_features


def test_diabetes_data_loads():
    df = load_diabetes_data()
    assert len(df) > 0
    assert "Outcome" in df.columns


def test_cleaning_replaces_invalid_values():
    df = clean_diabetes_data(load_diabetes_data())
    assert (df[["Glucose", "BloodPressure", "BMI"]] > 0).all().all()


def test_risk_features_created():
    df = add_risk_features(clean_diabetes_data(load_diabetes_data()))
    assert "glucose_risk" in df.columns
    assert "bmi_group" in df.columns
