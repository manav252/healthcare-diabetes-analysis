# Healthcare Diabetes Analysis

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

Data analyst case study exploring diabetes risk indicators with Python, Pandas, Seaborn, Streamlit, and baseline machine learning.

## Problem Statement

Healthcare teams need clear, interpretable analysis of patient risk indicators such as glucose, BMI, blood pressure, insulin, age, and diabetes outcome. This project cleans a diabetes dataset, explores important clinical patterns, and presents the findings in a Streamlit dashboard suitable for portfolio review.

## Dataset

The dataset is stored in `data/diabetes.csv` and contains 768 patient records with the following fields:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

## Tools

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-Learn
- Streamlit
- Pytest

## Workflow

1. Load the raw diabetes dataset.
2. Replace invalid zero values in medical columns with median values.
3. Create interpretable risk bands for glucose and BMI.
4. Perform exploratory data analysis on outcome, glucose, BMI, age, and correlations.
5. Train baseline Logistic Regression and Random Forest models.
6. Present insights through a Streamlit dashboard and written report.

## Key Insights

- Glucose is the strongest practical risk signal in the dataset.
- Higher BMI groups show higher diabetes outcome rates, especially when combined with high glucose.
- Cleaning invalid zero values is necessary before interpreting insulin, BMI, blood pressure, and skin thickness.
- Random Forest feature importance helps explain which variables contribute most to the baseline prediction model.

More details are available in [`reports/insights.md`](reports/insights.md).

## Business / Healthcare Impact

This dashboard helps a non-technical reviewer quickly identify high-risk patient groups, compare clinical indicators, and understand why data cleaning matters before healthcare analysis. It is designed as an educational analytics case study, not a diagnostic tool.

## Dashboard Screenshot

![Healthcare Streamlit dashboard](screenshots/streamlit-dashboard-overview.png)

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

## Folder Structure

```text
.
├── app.py
├── data/
├── docs/
├── notebooks/
├── reports/
│   └── insights.md
├── screenshots/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

## Future Scope

- Add cross-validation and hyperparameter tuning.
- Add SHAP or permutation importance for stronger model interpretability.
- Compare different imputation strategies for invalid medical values.
- Deploy the Streamlit dashboard for public access.
