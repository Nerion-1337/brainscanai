# Changelog

## [1.2.1]
- Correctif : le service `jupyter` quittait immédiatement (Jupyter refuse de tourner en root) → `--allow-root`, jeton vide via `IdentityProvider`, port lié à 127.0.0.1.

## [1.2.0]
- Accélération GPU multi-constructeurs : override `docker-compose.nvidia.yml` (CUDA) et `docker-compose.amd.yml` (ROCm, Linux natif) ; service `gpu-check`.
- Retrait de l'ancien service `features-gpu` (variante `cu121` obsolète) ; défaut NVIDIA `cu126`.
- Documentation `docs/gpu.md` (matrice OS/GPU, AMD sous Windows, accélération CPU).

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
