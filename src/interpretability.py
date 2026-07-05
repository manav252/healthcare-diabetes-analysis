"""Model interpretability utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import NUMERIC_FEATURES


def feature_importance_frame(model: Any) -> pd.DataFrame:
    """Extract global feature importance from fitted tree models or coefficients."""
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return pd.DataFrame({"feature": NUMERIC_FEATURES, "importance": values}).sort_values(
        "importance", ascending=False
    )


def save_feature_importance_plot(model: Any, output_path: Path) -> Path:
    """Save a horizontal feature importance plot for the fitted best model."""
    importance = feature_importance_frame(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["feature"], importance["importance"], color="#2563eb")
    ax.invert_yaxis()
    ax.set_title("Global Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def shap_summary_available() -> bool:
    """Return True when SHAP is installed in the current environment."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return True


def individual_prediction_explanation(model: Any, patient: pd.DataFrame) -> pd.DataFrame:
    """Provide a lightweight local explanation from global importances and patient values."""
    importance = feature_importance_frame(model)
    if importance.empty:
        return pd.DataFrame(columns=["feature", "patient_value", "global_importance"])
    values = patient.iloc[0].to_dict()
    explanation = importance.copy()
    explanation["patient_value"] = explanation["feature"].map(values)
    explanation = explanation.rename(columns={"importance": "global_importance"})
    return explanation[["feature", "patient_value", "global_importance"]]

