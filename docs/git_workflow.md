# Workflow Git / GitHub

Modèle **Git Flow allégé** : une branche `feature/*` par étape de la mission, fusion dans `develop` par Pull Request, `main` ne reçoit que des versions livrables (tags).

| Branche | Étape de la mission | Contenu principal | Tag / jalon |
|---|---|---|---|
| `main` | - | Version stable livrable | `v1.0.0` |
| `develop` | - | Intégration continue des étapes | - |
| `feature/00-project-setup` | Cadrage | Arborescence, config, CI, utilitaires, graphiques | - |
| `feature/01-data-exploration` | Étape 1 | Manifeste, qualité, doublons/outliers, split anti-fuite, notebook 01 | - |
| `feature/02-feature-extraction` | Étape 2 | Backbone ResNet gelé, embeddings, notebook 02 | - |
| `feature/03-clustering` | Étape 3 | PCA/t-SNE, 4 clusterings, ARI, labels faibles, notebook 03 | point d'étape avec Clara |
| `feature/04-semi-supervised` | Étape 4 | Métriques, sklearn semi-sup., CNN faible→fort, notebook 04 | - |
| `feature/05-scale-up` | Passage à l'échelle | Modèle de coûts 4 M images / 5 000 EUR | - |
| `feature/06-docs-presentation` | Livrables | Docs, auto-évaluation, plan de soutenance, README final | - |
| `feature/07-docker-uv` | Industrialisation | uv (`uv.lock`), Dockerfile multi-stage, docker-compose, CI | - |
| `feature/09-gpu-nvidia-amd` | Accélération | Overrides compose NVIDIA / AMD, `check_gpu.py`, `docs/gpu.md` | - |
| `release/v1.0.0` | Release | Gel, relecture, mise à jour de version | `v1.0.0`, `v1.1.0` |
| `hotfix/*` | - | Correctif urgent depuis `main` | `v1.0.x` |

## Conventions
- **Commits** : Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`).
- **Pull Request** : vers `develop`, modèle `.github/pull_request_template.md`, CI verte obligatoire.
- **Protection** (Settings > Branches) : `main` et `develop` protégées, review requise, historique linéaire ou squash.
- **Données** : jamais committées (`.gitignore`) ; notebooks committés **sans sorties** (`nbstripout --install`).

## Commandes
```bash
git checkout develop && git pull
git checkout -b feature/03-clustering
# ... travail, commits ...
git push -u origin feature/03-clustering   # puis PR vers develop
# release
git checkout -b release/v1.0.0 develop && git checkout main && git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "BrainScanAI v1.0.0" && git push origin main --tags
```
