"""Model training and comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import evaluate_classifier
from src.preprocessing import build_preprocessor, make_train_test_split


RANDOM_STATE = 42


@dataclass
class ModelComparisonResult:
    """Model comparison output used by scripts and dashboard views."""

    metrics: pd.DataFrame
    best_model_name: str
    best_estimator: Pipeline
    fitted_models: dict[str, Pipeline]
    x_test: pd.DataFrame
    y_test: pd.Series


def _optional_xgboost() -> tuple[str, Any] | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return (
        "XGBoost",
        XGBClassifier(
            eval_metric="logloss",
            learning_rate=0.05,
            max_depth=3,
            n_estimators=150,
            random_state=RANDOM_STATE,
            subsample=0.9,
        ),
    )


def get_candidate_models() -> dict[str, Any]:
    """Return baseline classifiers for diabetes outcome prediction."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    xgboost_model = _optional_xgboost()
    if xgboost_model is not None:
        name, model = xgboost_model
        models[name] = model
    return models


def build_model_pipeline(model: Any, scale: bool = True) -> Pipeline:
    """Combine preprocessing and a classifier in a single sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale=scale)),
            ("classifier", model),
        ]
    )


def tune_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Tune a Random Forest pipeline with a compact search space."""
    pipeline = build_model_pipeline(RandomForestClassifier(random_state=RANDOM_STATE), scale=False)
    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [4, 6, None],
        "classifier__min_samples_leaf": [2, 4],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    return search


def compare_models(df: pd.DataFrame, tune: bool = True) -> ModelComparisonResult:
    """Train candidate models, evaluate them, and select the best by ROC-AUC."""
    x_train, x_test, y_train, y_test = make_train_test_split(df)
    fitted_models: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, float | str]] = []

    for name, classifier in get_candidate_models().items():
        scale = name == "Logistic Regression"
        pipeline = build_model_pipeline(classifier, scale=scale)
        pipeline.fit(x_train, y_train)
        fitted_models[name] = pipeline
        metric_rows.append(evaluate_classifier(name, pipeline, x_test, y_test))

    if tune:
        tuned = tune_random_forest(x_train, y_train)
        tuned_name = "Tuned Random Forest"
        fitted_models[tuned_name] = tuned.best_estimator_
        tuned_metrics = evaluate_classifier(tuned_name, tuned.best_estimator_, x_test, y_test)
        tuned_metrics["best_params"] = str(tuned.best_params_)
        metric_rows.append(tuned_metrics)

    metrics = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_model_name = str(metrics.loc[0, "model"])
    return ModelComparisonResult(
        metrics=metrics,
        best_model_name=best_model_name,
        best_estimator=fitted_models[best_model_name],
        fitted_models=fitted_models,
        x_test=x_test,
        y_test=y_test,
    )
