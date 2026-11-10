.PHONY: install data reproduce reproduce-quick notebooks test dashboard

install:
	pip install -e ".[dev]"

data:
	python -m eew.data

reproduce:          ## every results file from the raw data (~15-20 min)
	python -m eew.reproduce

reproduce-quick:    ## skip the slow stage 6 transfer experiments
	python -m eew.reproduce --quick

notebooks:          ## re-execute all notebooks in place
	cd notebooks && for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $$nb; done

test:
	pytest -q

dashboard:
	streamlit run dashboards/app.py
