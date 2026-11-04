# Changelog

## [1.1.0]
- Gestion des dépendances avec **uv** (`pyproject.toml` + `uv.lock`, `.python-version`) ; suppression des `requirements*.txt`.
- **Docker** : Dockerfile multi-stage (runtime/dev, torch CPU ou CUDA), `docker-compose.yml` (un service par étape, tests, Jupyter, profil GPU).
- CI : uv + build Docker ; Makefile basé sur `uv run` + cibles `docker-*`.

## [1.0.0]
- Pipeline complet : qualité des données → embeddings → clustering → semi-supervisé → scale-up.
- Garde-fous anti-fuite, métriques orientées rappel, modèle de coûts paramétrable.
- Résultats sur le jeu réel : à produire (voir README, section Résultats).
