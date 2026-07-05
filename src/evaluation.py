"""Model evaluation metrics and plot helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve


def get_positive_probabilities(model: Any, x_test: pd.DataFrame) -> pd.Series:
    """Return positive-class probabilities for classifiers that expose them."""
    probabilities = model.predict_proba(x_test)[:, 1]
    return pd.Series(probabilities, index=x_test.index, name="diabetes_probability")


def evaluate_classifier(
    name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    """Calculate standard binary-classification metrics."""
    predictions = model.predict(x_test)
    probabilities = get_positive_probabilities(model, x_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def confusion_matrix_frame(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return confusion matrix as a labeled dataframe."""
    matrix = confusion_matrix(y_test, model.predict(x_test))
    return pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])


def threshold_analysis(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate precision/recall tradeoffs across probability thresholds."""
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    probabilities = get_positive_probabilities(model, x_test)
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )
    return pd.DataFrame(rows)


def subgroup_performance(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare recall and precision across clinically meaningful subgroups."""
    evaluation_df = x_test.copy()
    evaluation_df["Outcome"] = y_test
    evaluation_df["prediction"] = model.predict(x_test)
    evaluation_df["age_group"] = pd.cut(
        evaluation_df["Age"],
        bins=[20, 30, 45, 90],
        labels=["21-30", "31-45", "46+"],
        include_lowest=True,
    )
    evaluation_df["bmi_group"] = pd.cut(
        evaluation_df["BMI"],
        bins=[0, 24.9, 29.9, 80],
        labels=["Normal/Underweight", "Overweight", "Obese"],
        include_lowest=True,
    )

    rows = []
    for group_column in ["age_group", "bmi_group"]:
        for group_name, group in evaluation_df.groupby(group_column, observed=True):
            if group.empty:
                continue
            rows.append(
                {
                    "subgroup_type": group_column,
                    "subgroup": str(group_name),
                    "records": len(group),
                    "outcome_rate": group["Outcome"].mean(),
                    "accuracy": accuracy_score(group["Outcome"], group["prediction"]),
                    "precision": precision_score(group["Outcome"], group["prediction"], zero_division=0),
                    "recall": recall_score(group["Outcome"], group["prediction"], zero_division=0),
                    "f1_score": f1_score(group["Outcome"], group["prediction"], zero_division=0),
                }
            )
    return pd.DataFrame(rows)


def save_evaluation_plots(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, Path]:
    """Save ROC, precision-recall, and confusion matrix plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = get_positive_probabilities(model, x_test)
    predictions = model.predict(x_test)

    roc_path = output_dir / "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    pr_path = output_dir / "precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)

    cm_path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    calibration_path = output_dir / "calibration_curve.png"
    observed_rate, predicted_probability = calibration_curve(
        y_test,
        probabilities,
        n_bins=8,
        strategy="quantile",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(observed_rate, predicted_probability, marker="o", label="Best model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_title(f"Calibration Curve (Brier = {brier_score_loss(y_test, probabilities):.3f})")
    ax.set_xlabel("Observed diabetes outcome rate")
    ax.set_ylabel("Mean predicted probability")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(calibration_path, dpi=160)
    plt.close(fig)

    return {
        "roc_curve": roc_path,
        "precision_recall_curve": pr_path,
        "confusion_matrix": cm_path,
        "calibration_curve": calibration_path,
    }
