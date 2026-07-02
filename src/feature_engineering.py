import pandas as pd


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable diabetes risk bands for analysis."""
    featured_df = df.copy()
    featured_df["glucose_risk"] = pd.cut(
        featured_df["Glucose"],
        bins=[0, 99, 125, 300],
        labels=["Normal", "Prediabetes Range", "High"],
    )
    featured_df["bmi_group"] = pd.cut(
        featured_df["BMI"],
        bins=[0, 18.5, 24.9, 29.9, 80],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    return featured_df
