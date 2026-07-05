# Deployment

This repository is ready for Streamlit Community Cloud deployment.

## Streamlit Community Cloud

Use these settings:

- Repository: `manav252/healthcare-diabetes-analysis`
- Branch: `main`
- Main file path: `app.py`
- Python dependencies: `requirements.txt`

Open Streamlit Cloud:

https://share.streamlit.io/

Then create a new app with:

- Repository: `manav252/healthcare-diabetes-analysis`
- Branch: `main`
- Main file path: `app.py`

If Streamlit says the branch or file does not exist, reconnect GitHub in Streamlit Cloud and select the repository from the repository picker before entering the branch and file path. The GitHub remote contains `refs/heads/main`, and `app.py` is at the repository root.

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
