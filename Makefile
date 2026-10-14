# Supplier Reliability & Procurement Risk Scoring - reproducible pipeline
PY      ?= .venv/bin/python
SRISK   ?= .venv/bin/supplier-risk

.PHONY: setup data sql train report test lint app app-data screenshots all clean

setup:            ## create venv and install pinned deps + package
	/home/box/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-deps

data:             ## download Olist CSVs from public mirror and convert to parquet
	$(PY) scripts/download_data.py

sql:              ## run numbered DuckDB SQL scripts (staging, DQ, delivery, seller panel)
	$(SRISK) sql

train:            ## seller reliability scoring, clustering, OOT validation, exposure scenarios
	$(SRISK) train

report:           ## R GLM/lead-time Quarto report, memo PDF, notebooks
	Rscript R/delivery_analysis.R
	@if command -v quarto >/dev/null 2>&1; then cd reports && quarto render delivery_report.qmd --to html; fi
	$(PY) scripts/build_memo.py
	$(PY) scripts/build_readme.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_model_walkthrough.ipynb

lint:
	$(PY) -m ruff check .

test: lint        ## lint + pytest
	$(PY) -m pytest

app-data:         ## export compact aggregated artifacts for the Streamlit app (app/data/, committed)
	$(PY) scripts/export_app_data.py

app:              ## run the multipage Streamlit app on :8501
	.venv/bin/streamlit run app/streamlit_app.py --server.port 8501

screenshots:      ## capture every app page (app must be running on :8501; needs playwright chromium)
	$(PY) scripts/screenshot_app.py

all: data sql train report app-data test

clean:
	rm -rf data/processed artifacts/*.parquet reports/*_files
