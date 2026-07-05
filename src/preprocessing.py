"""Preprocessing utilities and sklearn pipelines."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.validation import INVALID_ZERO_COLUMNS, TARGET_COLUMN


NUMERIC_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def replace_invalid_zeros(
    df: pd.DataFrame,
    columns: Iterable[str] = INVALID_ZERO_COLUMNS,
) -> pd.DataFrame:
    """Replace clinically invalid zeros with missing values for imputation."""
    cleaned = df.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Build a reusable numeric preprocessing transformer."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES)],
        remainder="drop",
    )


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into model features and target."""
    return df[NUMERIC_FEATURES].copy(), df[target_column].copy()


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split for reproducible evaluation."""
    features, target = split_features_target(df)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
