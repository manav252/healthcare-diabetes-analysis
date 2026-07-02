import pandas as pd


def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return diabetes outcome counts for visualization."""
    return df["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="count")
