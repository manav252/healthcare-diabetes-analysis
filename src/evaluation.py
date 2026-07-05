"""Model evaluation metrics and plot helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
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

    return {"roc_curve": roc_path, "precision_recall_curve": pr_path, "confusion_matrix": cm_path}

