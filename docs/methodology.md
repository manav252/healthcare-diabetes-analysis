# Methodology

The project validates the raw dataset, treats clinically invalid zero values as missing measurements, and trains reproducible sklearn pipelines for diabetes outcome prediction. It is designed as an educational healthcare analytics solution, not a clinical diagnostic system.

Core columns:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- Age
- Outcome

## Modeling Workflow

1. Validate missing values, duplicates, expected data types, invalid zeros, and IQR outliers.
2. Replace invalid zeros in medical measurement fields with missing values.
3. Impute missing values with the training median inside an sklearn `Pipeline`.
4. Scale numeric inputs for Logistic Regression and leave tree models unscaled.
5. Compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost when installed.
6. Tune a Random Forest with cross-validated ROC-AUC.
7. Select the best model by test ROC-AUC.
8. Generate calibration and threshold tables for recall-sensitive screening decisions.
9. Compare performance across age and BMI subgroups.
10. Save model comparison metrics, plots, feature importance, and the best model artifact.

## Interpretability

Global feature importance is generated for tree-based models and absolute coefficients are used for linear models. Optional SHAP support is documented, surfaced in the dashboard, and exported by the training pipeline when the dependency supports the fitted estimator.

## External Validation

`src/external_validation.py` provides a schema-checked path for evaluating the trained model against a newer external diabetes dataset. A public external dataset is not bundled because clinical provenance and licensing should be reviewed before inclusion.
