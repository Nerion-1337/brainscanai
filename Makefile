.PHONY: install install-torch demo-data inspect features cluster semi scaleup test lint all clean docker-build docker-test docker-all lock

RUN ?= uv run

install:              ## environnement verrouillé via uv (Python + dépendances + dev)
	uv sync --group dev --group notebooks

lock:                 ## régénérer uv.lock après modification de pyproject.toml
	uv lock

demo-data:            ## images synthétiques pour tester sans le vrai dataset
	$(RUN) python scripts/make_demo_data.py

inspect:              ## étape 1 : exploration & qualité
	$(RUN) python scripts/inspect_data.py

features:             ## étape 2 : embeddings ResNet
	$(RUN) python scripts/extract_features.py

cluster:              ## étape 3 : clustering non supervisé
	$(RUN) python scripts/run_clustering.py

semi:                 ## étape 4 : semi-supervisé (CNN + sklearn)
	$(RUN) python scripts/run_semi_supervised.py

scaleup:              ## étape 5 : faisabilité 4 M d'images / 5 000 EUR
	$(RUN) python scripts/estimate_scale_up.py

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests scripts

all: inspect features cluster semi scaleup

clean:
	rm -rf data/processed/* data/features/* reports/figures/* reports/tables/*

# ---- Docker
docker-build:         ## image d'exécution (CPU) ; GPU : make docker-build TORCH_VARIANT=cu121
	docker compose build --build-arg TORCH_VARIANT=$${TORCH_VARIANT:-cpu} inspect

docker-test:
	docker compose run --rm test

docker-all:
	docker compose run --rm inspect && docker compose run --rm features && \
	docker compose run --rm cluster && docker compose run --rm semi && docker compose run --rm scaleup
