# Project Audit

## Current Project Structure

The repository already contains a compact data science project scaffold:

- `data/diabetes.csv`: raw diabetes dataset with 768 records and 9 columns.
- `notebooks/health_care_project.py`: original exploratory analysis script.
- `src/`: reusable package skeleton for data processing, feature engineering, utilities, visualization, and modeling.
- `app.py`: Streamlit dashboard with overview, EDA, risk insights, and baseline model tabs.
- `reports/insights.md`: healthcare-oriented findings and limitations.
- `docs/`: architecture, methodology, and findings notes.
- `tests/`: pytest checks for data loading, cleaning, and feature creation.
- `.github/workflows/ci.yml`: CI workflow for installing requirements, compiling Python files, and running tests.
- `screenshots/`: dashboard and analysis screenshots.

## What Already Exists

### Existing Notebooks/Scripts

- A script-style notebook exists at `notebooks/health_care_project.py`.
- It loads the dataset, performs basic cleaning, prints summary statistics, and creates several EDA charts.

### Existing EDA

- Outcome distribution.
- BMI distribution.
- Age vs glucose scatter plot.
- Glucose distribution.
- Correlation heatmap in the dashboard.
- Risk segment comparisons for glucose and BMI.

### Existing Preprocessing

- `src/data_processing.py` loads the CSV.
- Invalid zero values are replaced with medians for key medical measurement columns.
- `src/feature_engineering.py` creates glucose risk and BMI group features.

### Existing Visualizations

- Matplotlib and Seaborn charts in the dashboard and notebook script.
- Screenshots are included for portfolio presentation.

### Existing ML Models

- Logistic Regression and Random Forest are trained inside the Streamlit dashboard.
- Metrics include accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and Random Forest feature importance.

### Existing Documentation

- README with overview, dataset, tools, workflow, insights, dashboard screenshot, and run instructions.
- Supporting documentation in `docs/`.
- Written insights in `reports/insights.md`.
- Contributing, changelog, license, requirements, and CI workflow are present.

### Existing Datasets

- `data/diabetes.csv` is present and tracked.
- The dataset has no missing values and no duplicate rows.
- Clinically invalid zeros are present in glucose, blood pressure, skin thickness, insulin, and BMI.

### Existing Dashboard

- `app.py` is a working Streamlit app with overview, exploratory analysis, risk insights, and model baseline sections.

## What Is Good

- The project already uses a modular folder layout rather than a single notebook only.
- The dashboard is recruiter-friendly and easy to run.
- The README clearly states this is not a diagnostic tool.
- CI and tests already exist, which is stronger than many beginner portfolio projects.
- Cleaning invalid medical zeros is a clinically aware preprocessing step.
- The existing risk bands make the analysis easier for non-technical stakeholders.

## What Is Missing

- No formal data validation module or generated data quality report.
- Model training logic is embedded in the dashboard instead of reusable training modules.
- `src/modeling.py` is a placeholder.
- No sklearn `Pipeline`/`ColumnTransformer` training path.
- No model artifact persistence in `models/`.
- No multi-model comparison beyond two baseline classifiers.
- No hyperparameter tuning.
- No standalone evaluation module for ROC curves, precision-recall curves, and comparison tables.
- No reusable prediction API.
- No SHAP or robust interpretability workflow.
- No dashboard folder despite a dashboard being part of the project goal.
- Tests do not cover prediction or training pipeline behavior.
- README needs a more production-grade architecture and results narrative.

## What Should Be Improved

| Priority | Improvement | Rationale |
| --- | --- | --- |
| High | Add data validation and quality reporting | Healthcare analytics projects need explicit checks for missingness, duplicates, invalid values, schema issues, and outliers. |
| High | Move model training out of Streamlit | Reusable modules make the project look like a real ML solution and reduce dashboard complexity. |
| High | Add sklearn pipelines and train/test split helpers | Pipelines improve reproducibility and prevent preprocessing leakage. |
| High | Compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and optional XGBoost | Recruiters expect structured model benchmarking. |
| High | Add model evaluation artifacts and best-model selection | A portfolio project should clearly explain how the chosen model was selected. |
| High | Add prediction helpers and tests | The dashboard prediction workflow should use the same pipeline as training. |
| Medium | Add feature importance and optional SHAP explanations | Interpretability is especially important for healthcare analytics. |
| Medium | Create `dashboard/` package while preserving root `app.py` | Matches requested production structure without breaking existing run commands. |
| Medium | Rewrite README with architecture, setup, results, and business insights | Improves reviewer confidence and admissions presentation quality. |
| Medium | Add `models/` directory and artifact outputs | Demonstrates end-to-end ML lifecycle thinking. |
| Low | Add richer notebook exports | The codebase should be primary, with notebooks as supporting analysis only. |
| Low | Add deployment documentation | Useful after model and dashboard structure are stable. |

## Improvements Implemented In This Update

- Added `src/validation.py` and generated `reports/data_quality_report.md`.
- Added sklearn preprocessing pipelines in `src/preprocessing.py`.
- Replaced dashboard-embedded model training with reusable modeling, evaluation, interpretability, and prediction modules.
- Added Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, optional XGBoost, and tuned Random Forest comparison.
- Added model comparison export at `reports/model_comparison.csv`.
- Added generated ROC, precision-recall, confusion matrix, and feature importance plots under `reports/figures/`.
- Added a persisted best model artifact at `models/best_diabetes_model.joblib`.
- Added a professional dashboard package at `dashboard/app.py` while preserving `app.py` as the existing Streamlit entry point.
- Added patient prediction and local explanation helpers.
- Added unit tests for validation, model comparison, and patient prediction.
- Rewrote README and expanded methodology/architecture documentation.
