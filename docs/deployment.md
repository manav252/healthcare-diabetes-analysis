# Deployment

This repository is ready for Streamlit Community Cloud deployment.

## Streamlit Community Cloud

Use these settings:

- Repository: `manav252/healthcare-diabetes-analysis`
- Branch: `main`
- Main file path: `app.py`
- Python dependencies: `requirements.txt`

Recommended public demo URL:

https://healthcare-diabetes-analysis.streamlit.app/

If Streamlit assigns a different slug, update the live demo link in `README.md`.

## Local Smoke Test

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Notes

- The app retrains a compact model comparison pipeline and caches the result with Streamlit.
- The project includes `.streamlit/config.toml` for consistent dashboard styling.
- The dashboard is an educational analytics demo and must not be used for clinical diagnosis.
