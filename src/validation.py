"""Data validation checks for the diabetes analytics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_COLUMNS = {
    "Pregnancies": "int64",
    "Glucose": "int64",
    "BloodPressure": "int64",
    "SkinThickness": "int64",
    "Insulin": "int64",
    "BMI": "float64",
    "DiabetesPedigreeFunction": "float64",
    "Age": "int64",
    "Outcome": "int64",
}

INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_COLUMN = "Outcome"


@dataclass(frozen=True)
class ValidationResult:
    """Container for core data quality checks."""

    row_count: int
    column_count: int
    missing_values: dict[str, int]
    duplicate_rows: int
    invalid_zero_counts: dict[str, int]
    incorrect_types: dict[str, str]
    outlier_counts: dict[str, int]

    @property
    def passed(self) -> bool:
        """Return True when no structural data quality issues are present."""
        return (
            not any(self.missing_values.values())
            and self.duplicate_rows == 0
            and not self.incorrect_types
        )


def find_outliers_iqr(df: pd.DataFrame, columns: Iterable[str] | None = None) -> dict[str, int]:
    """Count numeric outliers using the 1.5 IQR rule."""
    numeric_columns = list(columns) if columns is not None else list(df.select_dtypes("number").columns)
    outliers: dict[str, int] = {}
    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[column] = int(((df[column] < lower) | (df[column] > upper)).sum())
    return outliers


def validate_diabetes_data(df: pd.DataFrame) -> ValidationResult:
    """Validate schema, missingness, duplicates, invalid values, and outliers."""
    missing_values = df.isna().sum().astype(int).to_dict()
    duplicate_rows = int(df.duplicated().sum())
    invalid_zero_counts = {
        column: int((df[column] == 0).sum())
        for column in INVALID_ZERO_COLUMNS
        if column in df.columns
    }

    incorrect_types: dict[str, str] = {}
    for column, expected_type in EXPECTED_COLUMNS.items():
        if column not in df.columns:
            incorrect_types[column] = "missing column"
        elif str(df[column].dtype) != expected_type:
            incorrect_types[column] = str(df[column].dtype)

    return ValidationResult(
        row_count=len(df),
        column_count=len(df.columns),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        invalid_zero_counts=invalid_zero_counts,
        incorrect_types=incorrect_types,
        outlier_counts=find_outliers_iqr(df),
    )


def validation_to_markdown(result: ValidationResult) -> str:
    """Render a validation result as a Markdown report."""
    lines = [
        "# Data Quality Report",
        "",
        "## Summary",
        "",
        f"- Rows: {result.row_count}",
        f"- Columns: {result.column_count}",
        f"- Duplicate rows: {result.duplicate_rows}",
        f"- Structural checks passed: {'Yes' if result.passed else 'No'}",
        "",
        "## Missing Values",
        "",
        "| Column | Missing Values |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {column} | {count} |" for column, count in result.missing_values.items())

    lines.extend(["", "## Invalid Zero Values", "", "| Column | Zero Count |", "| --- | ---: |"])
    lines.extend(f"| {column} | {count} |" for column, count in result.invalid_zero_counts.items())

    lines.extend(["", "## Incorrect Data Types", "", "| Column | Observed Type |", "| --- | --- |"])
    if result.incorrect_types:
        lines.extend(f"| {column} | {observed} |" for column, observed in result.incorrect_types.items())
    else:
        lines.append("| None | All expected types present |")

    lines.extend(["", "## IQR Outlier Counts", "", "| Column | Outliers |", "| --- | ---: |"])
    lines.extend(f"| {column} | {count} |" for column, count in result.outlier_counts.items())
    lines.append("")
    return "\n".join(lines)


def write_data_quality_report(df: pd.DataFrame, output_path: Path) -> Path:
    """Validate a dataframe and write a Markdown data quality report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(validation_to_markdown(validate_diabetes_data(df)), encoding="utf-8")
    return output_path

