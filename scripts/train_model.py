"""Train diabetes classifiers and save portfolio-ready artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from src.data_processing import DATA_PATH, load_diabetes_data
from src.evaluation import save_evaluation_plots
from src.interpretability import save_feature_importance_plot
from src.modeling import compare_models
from src.preprocessing import replace_invalid_zeros
from src.validation import write_data_quality_report


def main() -> None:
    """Run the full training and reporting pipeline."""
    raw_df = load_diabetes_data(DATA_PATH)
    modeling_df = replace_invalid_zeros(raw_df)

    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(exist_ok=True)

    write_data_quality_report(raw_df, reports_dir / "data_quality_report.md")
    result = compare_models(modeling_df, tune=True)
    result.metrics.to_csv(reports_dir / "model_comparison.csv", index=False)
    joblib.dump(result.best_estimator, models_dir / "best_diabetes_model.joblib")

    save_evaluation_plots(result.best_estimator, result.x_test, result.y_test, figures_dir)
    save_feature_importance_plot(result.best_estimator, figures_dir / "feature_importance.png")

    print(f"Best model: {result.best_model_name}")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
