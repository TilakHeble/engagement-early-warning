.PHONY: install data test dashboard

install:
	pip install -e ".[dev]"

data:
	python -m eew.data

test:
	pytest -q

dashboard:
	streamlit run dashboards/app.py
