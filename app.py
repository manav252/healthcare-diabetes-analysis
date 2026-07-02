from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_processing import clean_diabetes_data, load_diabetes_data
from src.feature_engineering import add_risk_features


PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data
def get_data() -> pd.DataFrame:
    """Load, clean, and enrich the diabetes dataset for the dashboard."""
    df = load_diabetes_data(PROJECT_ROOT / "data" / "diabetes.csv")
    df = clean_diabetes_data(df)
    return add_risk_features(df)


def draw_bar(data: pd.DataFrame, x: str, y: str, title: str) -> None:
    """Render a compact bar chart with consistent styling."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=data, x=x, y=y, ax=ax, color="#2F80ED")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig, use_container_width=True)


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train two baseline classifiers and return metrics, confusion matrix, and feature importance."""
    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    x = df[features]
    y = df["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    rows = []
    matrices = {}
    importances = pd.DataFrame()
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(x_train_scaled, y_train)
            predictions = model.predict(x_test_scaled)
            probabilities = model.predict_proba(x_test_scaled)[:, 1]
        else:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)[:, 1]
            importances = pd.DataFrame(
                {"feature": features, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        matrices[name] = confusion_matrix(y_test, predictions)

    return pd.DataFrame(rows), pd.DataFrame(matrices["Random Forest"]), importances


st.set_page_config(
    page_title="Healthcare Diabetes Analysis",
    page_icon="H",
    layout="wide",
)

st.title("Healthcare Diabetes Analysis")
st.caption("Clinical risk indicator analysis using Python, Pandas, Streamlit, and Scikit-Learn.")

df = get_data()

overview_tab, eda_tab, risk_tab, model_tab = st.tabs(
    ["Overview", "Exploratory Analysis", "Risk Insights", "Model Baseline"]
)

with overview_tab:
    total_patients = len(df)
    diabetes_rate = df["Outcome"].mean()
    median_glucose = df["Glucose"].median()
    median_bmi = df["BMI"].median()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patient Records", f"{total_patients:,}")
    col2.metric("Diabetes Outcome Rate", f"{diabetes_rate:.1%}")
    col3.metric("Median Glucose", f"{median_glucose:.0f}")
    col4.metric("Median BMI", f"{median_bmi:.1f}")

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

with eda_tab:
    col1, col2 = st.columns(2)
    with col1:
        outcome_counts = df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"}).value_counts().reset_index()
        outcome_counts.columns = ["outcome", "count"]
        draw_bar(outcome_counts, "outcome", "count", "Outcome Distribution")
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x="Glucose", hue="Outcome", bins=30, kde=True, ax=ax)
        ax.set_title("Glucose Distribution by Outcome")
        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x="Outcome", y="BMI", ax=ax)
        ax.set_title("BMI by Diabetes Outcome")
        st.pyplot(fig, use_container_width=True)
    with col4:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df.select_dtypes("number").corr(), cmap="Blues", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig, use_container_width=True)

with risk_tab:
    st.subheader("Risk Segment Comparison")
    glucose_rate = (
        df.groupby("glucose_risk", observed=True)["Outcome"]
        .mean()
        .reset_index(name="diabetes_rate")
    )
    bmi_rate = (
        df.groupby("bmi_group", observed=True)["Outcome"]
        .mean()
        .reset_index(name="diabetes_rate")
    )

    col1, col2 = st.columns(2)
    with col1:
        draw_bar(glucose_rate, "glucose_risk", "diabetes_rate", "Diabetes Rate by Glucose Risk Band")
    with col2:
        draw_bar(bmi_rate, "bmi_group", "diabetes_rate", "Diabetes Rate by BMI Group")

    st.info(
        "Higher glucose bands show the clearest rise in diabetes outcome rate. BMI and age add useful context, "
        "but glucose is the strongest single screening signal in this dataset."
    )

with model_tab:
    st.subheader("Baseline Classification Models")
    metric_table, rf_matrix, feature_importance = train_models(df)
    st.dataframe(metric_table.style.format({column: "{:.3f}" for column in metric_table.columns if column != "model"}), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(rf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Random Forest Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, use_container_width=True)
    with col2:
        draw_bar(feature_importance, "feature", "importance", "Random Forest Feature Importance")
