from pathlib import Path

import pandas as pd

from src.preprocessing import replace_invalid_zeros


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"


def load_diabetes_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes dataset."""
    return pd.read_csv(path)


def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid zero values in medical measurements with column medians."""
    cleaned_df = replace_invalid_zeros(df)
    for column in cleaned_df.columns:
        if cleaned_df[column].isna().any() and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
    return cleaned_df
