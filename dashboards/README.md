# Tutor dashboard

Live: https://engagement-early-warning.streamlit.app/

```bash
pip install -r dashboards/requirements.txt
streamlit run dashboards/app.py
```

Runs on the committed snapshot in `dashboards/snapshot/` (2014 presentations, held-out scores), so no raw data is needed.
To rebuild the snapshot after re-running the pipeline: `python -m eew.dashboard_data`.

**Deploy on Streamlit Community Cloud:** New app → this repository, branch `main`, main file `dashboards/app.py`.
Requirements are read from `dashboards/requirements.txt`.
