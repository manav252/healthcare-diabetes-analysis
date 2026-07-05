# Architecture

- `data/`: diabetes CSV
- `notebooks/`: original EDA script with portable paths
- `src/data_processing.py`: loading and median-based medical cleaning
- `src/validation.py`: schema, missingness, duplicate, invalid-value, and outlier checks
- `src/preprocessing.py`: invalid-zero handling, train/test split, and sklearn preprocessing pipeline
- `src/modeling.py`: multi-model comparison, hyperparameter tuning, and best-model selection
- `src/evaluation.py`: metrics, confusion matrix, ROC curve, and precision-recall curve helpers
- `src/external_validation.py`: optional evaluation against a newer external diabetes dataset
- `src/interpretability.py`: feature importance and optional SHAP-ready explainability hooks
- `src/prediction.py`: single-patient prediction API
- `src/feature_engineering.py`: glucose and BMI risk features
- `scripts/train_model.py`: end-to-end training and artifact generation
- `dashboard/app.py`: Streamlit user interface
- `.streamlit/config.toml`: deployment-ready dashboard configuration
- `models/`: trained model artifacts
- `screenshots/`: EDA output
- `tests/`: reproducibility checks
