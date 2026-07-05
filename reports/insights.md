# Healthcare Diabetes Analysis Insights

## Executive Summary

This project analyzes diabetes risk indicators from a structured healthcare dataset. The goal is not to diagnose patients, but to identify which clinical measurements are most associated with the recorded diabetes outcome and present the findings in a recruiter-friendly analytics case study.

## Data Quality Notes

- The dataset contains 768 patient records and 9 columns.
- Several medical fields contain zero values that are not clinically valid for this dataset: glucose, blood pressure, skin thickness, insulin, and BMI.
- Invalid zero values are replaced with median values to avoid dropping a large number of records while keeping the analysis interpretable.

## Key Insights

- Glucose is the strongest practical risk signal in the dataset. Patients in higher glucose bands show a noticeably higher diabetes outcome rate.
- BMI adds useful supporting context. Obese BMI groups generally show higher diabetes outcome rates than normal BMI groups.
- Age and pregnancies help explain differences between patient groups, but they are less directly actionable than glucose and BMI screening.
- Insulin and skin thickness contain many invalid zero values in the raw data, so cleaning is important before interpreting these fields.

## Business / Healthcare Impact

- A dashboard can help healthcare analysts quickly compare high-risk patient groups and prioritize deeper clinical review.
- Risk bands make the analysis easier for non-technical stakeholders than raw numeric distributions alone.
- The model baseline demonstrates how cleaned clinical features can support early screening workflows, while still requiring clinical validation before any real-world use.

## Model Baseline

The project now compares Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, optional XGBoost, and a tuned Random Forest pipeline. Metrics include accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve, precision-recall curve, and global feature importance.

## Practical Recommendations

- Prioritize glucose screening and follow-up review for patients in the highest glucose band.
- Interpret BMI, age, pregnancies, and diabetes pedigree function as supporting risk context rather than standalone diagnostic signals.
- Treat invalid zeros in insulin, skin thickness, BMI, blood pressure, and glucose as data quality issues before analysis.
- Use model probabilities as a triage aid for education and portfolio demonstration only; clinical deployment would require external validation, calibration, bias review, and medical governance.

## Limitations

- This is a small educational dataset, not a production clinical system.
- The model should not be used for diagnosis.
- Additional features such as lifestyle, family history details, lab trends, and medical history would improve real-world usefulness.

## Future Improvements

- Add cross-validation and hyperparameter tuning.
- Add SHAP explanations for model interpretability.
- Compare imputation strategies for invalid medical zero values.
- Deploy the Streamlit dashboard for easier portfolio review.
