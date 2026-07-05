"""Prediction API for patient diabetes risk scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing import NUMERIC_FEATURES


def patient_to_frame(patient: dict[str, float | int]) -> pd.DataFrame:
    """Convert patient input into a model-ready dataframe."""
    missing = [feature for feature in NUMERIC_FEATURES if feature not in patient]
    if missing:
        raise ValueError(f"Missing patient features: {', '.join(missing)}")
    return pd.DataFrame([{feature: patient[feature] for feature in NUMERIC_FEATURES}])


def predict_patient(model: Any, patient: dict[str, float | int]) -> dict[str, float | int | str]:
    """Predict diabetes outcome and probability for a single patient."""
    patient_df = patient_to_frame(patient)
    probability = float(model.predict_proba(patient_df)[:, 1][0])
    prediction = int(probability >= 0.5)
    risk_label = "Higher Risk" if prediction == 1 else "Lower Risk"
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_label": risk_label,
    }

