"""External validation helpers for newer diabetes datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import evaluate_classifier
from src.preprocessing import NUMERIC_FEATURES
from src.validation import TARGET_COLUMN


def load_external_validation_data(path: Path) -> pd.DataFrame:
    """Load an external validation CSV with the expected diabetes schema."""
    df = pd.read_csv(path)
    required_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"External validation data is missing columns: {', '.join(missing_columns)}")
    return df[required_columns]


def evaluate_external_dataset(model: Any, external_df: pd.DataFrame) -> dict[str, float | str]:
    """Evaluate a fitted model on a held-out external diabetes dataset."""
    return evaluate_classifier(
        "External Validation",
        model,
        external_df[NUMERIC_FEATURES],
        external_df[TARGET_COLUMN],
    )

