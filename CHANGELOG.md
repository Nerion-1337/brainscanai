# Changelog

## [1.1.2]
- Correctif Docker : `uv sync` désinstallait PyTorch (installé hors lockfile) → ajout de `--inexact` et d'un contrôle `import torch` au build.

## [1.1.1]
- Docker devient le chemin d'exécution principal (aucun Python/uv/make requis en local) ; service `pipeline` pour tout enchaîner.
- uv et make restent disponibles en option pour le développement local.

## [1.1.0]
- Gestion des dépendances avec **uv** (`pyproject.toml` + `uv.lock`, `.python-version`) ; suppression des `requirements*.txt`.
- **Docker** : Dockerfile multi-stage (runtime/dev, torch CPU ou CUDA), `docker-compose.yml` (un service par étape, tests, Jupyter, profil GPU).
- CI : uv + build Docker ; Makefile basé sur `uv run` + cibles `docker-*`.

## [1.0.0]
- Pipeline complet : qualité des données → embeddings → clustering → semi-supervisé → scale-up.
- Garde-fous anti-fuite, métriques orientées rappel, modèle de coûts paramétrable.
- Résultats sur le jeu réel : à produire (voir README, section Résultats).
