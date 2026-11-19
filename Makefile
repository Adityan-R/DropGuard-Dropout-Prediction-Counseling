.PHONY: setup download train experiments test app clean all

PYTHON ?= python

all: setup download train experiments test

setup:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) -m src.dropguard.train

experiments:
	$(PYTHON) -m src.dropguard.experiments

test:
	$(PYTHON) -m pytest tests/ -v

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

clean:
	rm -f reports/metrics.json reports/model_comparison.csv reports/sample_counseling_notes.md
	rm -f reports/threshold_results.json reports/ablation_results.json reports/regression_results.json
	rm -f reports/imbalance_comparison.csv
	rm -f reports/figures/*.png
	rm -f models/*.joblib
