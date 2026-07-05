"""Professional Streamlit dashboard for diabetes analytics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_processing import clean_diabetes_data, load_diabetes_data
from src.evaluation import confusion_matrix_frame
from src.feature_engineering import add_risk_features
from src.interpretability import (
    feature_importance_frame,
    individual_prediction_explanation,
    shap_summary_available,
)
from src.modeling import compare_models
from src.prediction import predict_patient
from src.preprocessing import NUMERIC_FEATURES, replace_invalid_zeros
from src.validation import validate_diabetes_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"


@st.cache_data
def get_raw_data() -> pd.DataFrame:
    """Load raw diabetes data."""
    return load_diabetes_data(DATA_PATH)


@st.cache_data
def get_dashboard_data() -> pd.DataFrame:
    """Load cleaned and enriched data for EDA."""
    return add_risk_features(clean_diabetes_data(get_raw_data()))


@st.cache_resource
def get_model_result():
    """Train and cache model comparison results."""
    return compare_models(replace_invalid_zeros(get_raw_data()), tune=True)


def draw_bar(data: pd.DataFrame, x: str, y: str, title: str) -> None:
    """Render a compact bar chart with consistent styling."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=data, x=x, y=y, ax=ax, color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_overview(df: pd.DataFrame) -> None:
    """Render dataset overview and data quality checks."""
    validation = validate_diabetes_data(get_raw_data())
    cols = st.columns(4)
    cols[0].metric("Patient Records", f"{len(df):,}")
    cols[1].metric("Outcome Rate", f"{df['Outcome'].mean():.1%}")
    cols[2].metric("Median Glucose", f"{df['Glucose'].median():.0f}")
    cols[3].metric("Invalid Raw Zeros", f"{sum(validation.invalid_zero_counts.values()):,}")

    st.subheader("Data Quality")
    quality = pd.DataFrame(
        {
            "check": ["Missing values", "Duplicate rows", "Incorrect data types"],
            "value": [
                sum(validation.missing_values.values()),
                validation.duplicate_rows,
                len(validation.incorrect_types),
            ],
        }
    )
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.dataframe(df.head(20), use_container_width=True)


def render_eda(df: pd.DataFrame) -> None:
    """Render exploratory charts."""
    col1, col2 = st.columns(2)
    with col1:
        outcome_counts = df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"}).value_counts().reset_index()
        outcome_counts.columns = ["outcome", "count"]
        draw_bar(outcome_counts, "outcome", "count", "Outcome Distribution")
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x="Glucose", hue="Outcome", bins=30, kde=True, ax=ax)
        ax.set_title("Glucose Distribution by Outcome")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x="Outcome", y="BMI", ax=ax)
        ax.set_title("BMI by Diabetes Outcome")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with col4:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df[NUMERIC_FEATURES + ["Outcome"]].corr(), cmap="Blues", ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


def render_model_performance(result) -> None:
    """Render model comparison and best-model diagnostics."""
    st.subheader(f"Best Model: {result.best_model_name}")
    numeric_columns = result.metrics.select_dtypes("number").columns
    st.dataframe(
        result.metrics.style.format({column: "{:.3f}" for column in numeric_columns}),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            confusion_matrix_frame(result.best_estimator, result.x_test, result.y_test),
            use_container_width=True,
        )
    with col2:
        importance = feature_importance_frame(result.best_estimator)
        draw_bar(importance, "feature", "importance", "Global Feature Importance")


def render_prediction(result) -> None:
    """Render individual patient prediction controls."""
    st.subheader("Patient Prediction")
    col1, col2, col3 = st.columns(3)
    patient = {
        "Pregnancies": col1.number_input("Pregnancies", min_value=0, max_value=20, value=2),
        "Glucose": col2.number_input("Glucose", min_value=1, max_value=250, value=120),
        "BloodPressure": col3.number_input("Blood Pressure", min_value=1, max_value=140, value=72),
        "SkinThickness": col1.number_input("Skin Thickness", min_value=1, max_value=100, value=23),
        "Insulin": col2.number_input("Insulin", min_value=1, max_value=900, value=80),
        "BMI": col3.number_input("BMI", min_value=1.0, max_value=80.0, value=31.5),
        "DiabetesPedigreeFunction": col1.number_input(
            "Diabetes Pedigree Function", min_value=0.01, max_value=3.0, value=0.45
        ),
        "Age": col2.number_input("Age", min_value=18, max_value=100, value=35),
    }
    result_payload = predict_patient(result.best_estimator, patient)
    st.metric(result_payload["risk_label"], f"{result_payload['probability']:.1%}")

    explanation = individual_prediction_explanation(
        result.best_estimator,
        pd.DataFrame([patient]),
    )
    st.dataframe(explanation, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Healthcare Diabetes ML", page_icon="H", layout="wide")
    st.title("Healthcare Diabetes ML Analytics")
    st.caption("End-to-end data validation, model comparison, interpretability, and patient risk scoring.")

    df = get_dashboard_data()
    result = get_model_result()

    tabs = st.tabs(
        [
            "Dataset Overview",
            "EDA",
            "Risk Insights",
            "Patient Prediction",
            "Feature Importance",
            "Model Performance",
            "About Project",
        ]
    )

    with tabs[0]:
        render_overview(df)
    with tabs[1]:
        render_eda(df)
    with tabs[2]:
        glucose_rate = df.groupby("glucose_risk", observed=True)["Outcome"].mean().reset_index(name="diabetes_rate")
        bmi_rate = df.groupby("bmi_group", observed=True)["Outcome"].mean().reset_index(name="diabetes_rate")
        col1, col2 = st.columns(2)
        with col1:
            draw_bar(glucose_rate, "glucose_risk", "diabetes_rate", "Diabetes Rate by Glucose Risk")
        with col2:
            draw_bar(bmi_rate, "bmi_group", "diabetes_rate", "Diabetes Rate by BMI Group")
    with tabs[3]:
        render_prediction(result)
    with tabs[4]:
        importance = feature_importance_frame(result.best_estimator)
        draw_bar(importance, "feature", "importance", "Best Model Feature Importance")
        st.info(
            "SHAP visualizations are enabled when the optional `shap` dependency is installed."
            if not shap_summary_available()
            else "SHAP is available in this environment for deeper local explanations."
        )
    with tabs[5]:
        render_model_performance(result)
    with tabs[6]:
        st.write(
            "This project is an educational healthcare analytics portfolio project. "
            "It supports data quality review, model benchmarking, interpretability, and patient-level risk scoring. "
            "It is not intended for clinical diagnosis."
        )


if __name__ == "__main__":
    main()
