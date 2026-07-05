# Deployment

This repository is ready for Streamlit Community Cloud deployment.

## Streamlit Community Cloud

Use these settings:

- Repository: `manav252/healthcare-diabetes-analysis`
- Branch: `main`
- Main file path: `app.py`
- Python dependencies: `requirements.txt`

Deploy shortcut:

https://share.streamlit.io/deploy?repository=https://github.com/manav252/healthcare-diabetes-analysis&branch=main&mainModule=app.py

After Streamlit creates the app, copy the generated public `*.streamlit.app` URL into `README.md`.

## Local Smoke Test

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Notes

- The app retrains a compact model comparison pipeline and caches the result with Streamlit.
- The project includes `.streamlit/config.toml` for consistent dashboard styling.
- The dashboard is an educational analytics demo and must not be used for clinical diagnosis.
