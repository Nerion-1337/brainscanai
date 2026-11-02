.PHONY: install demo-data inspect features cluster semi scaleup test lint all clean

PY ?= python

install:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

demo-data:            ## images synthétiques pour tester sans le vrai dataset
	$(PY) scripts/make_demo_data.py

inspect:              ## étape 1 : exploration & qualité
	$(PY) scripts/inspect_data.py

features:             ## étape 2 : embeddings ResNet
	$(PY) scripts/extract_features.py

cluster:              ## étape 3 : clustering non supervisé
	$(PY) scripts/run_clustering.py

semi:                 ## étape 4 : semi-supervisé (CNN + sklearn)
	$(PY) scripts/run_semi_supervised.py

scaleup:              ## étape 5 : faisabilité 4 M d'images / 5 000 EUR
	$(PY) scripts/estimate_scale_up.py

test:
	pytest

lint:
	ruff check src tests scripts

all: inspect features cluster semi scaleup

clean:
	rm -rf data/processed/* data/features/* reports/figures/* reports/tables/*
