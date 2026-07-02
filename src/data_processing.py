from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"


def load_diabetes_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes dataset."""
    return pd.read_csv(path)


def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid zero values in medical measurements with medians."""
    cleaned_df = df.copy()
    invalid_zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for column in invalid_zero_columns:
        cleaned_df[column] = cleaned_df[column].replace(0, cleaned_df[column].median())
    return cleaned_df
