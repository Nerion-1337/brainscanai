<p align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,numpy,pandas,jupyter,docker,git,github,githubactions,vscode&perline=11" alt="Stack technique" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img alt="pandas" src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
  <img alt="Pillow" src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="uv" src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img alt="GitHub Actions" src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
  <img alt="Licence" src="https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge" />
</p>

# 🧠 BrainScanAI - Détection de tumeurs cérébrales par apprentissage semi-supervisé

> Projet R&D de **CurelyticsIA** (mission fictive, parcours Data Scientist). Exploration analytique d'IRM cérébrales : **1 400 images non labellisées + 100 labellisées** (normal / cancer). Objectif : tirer parti des images sans label via des embeddings pré-entraînés, du clustering et des méthodes semi-supervisées, puis évaluer la faisabilité d'un passage à l'échelle (**4 M d'images, 5 000 EUR**).

> ⚠️ **Projet de recherche / pédagogique - aucune valeur diagnostique.** Toute utilisation clinique relève de la réglementation des dispositifs médicaux.

---

## 📑 Sommaire
1. [Objectifs et livrables](#-objectifs-et-livrables)
2. [Données](#-données)
3. [Architecture du projet](#-architecture-du-projet)
4. [Pipeline et méthodologie](#-pipeline-et-méthodologie)
5. [Diagrammes UML](#-diagrammes-uml)
6. [Stratégie Git / GitHub](#-stratégie-git--github)
7. [Installation et exécution (Docker uniquement)](#-installation-et-exécution-docker-uniquement)
8. [Métriques et definition of done](#-métriques-et-definition-of-done)
9. [Passage à l'échelle](#-passage-à-léchelle-4-m-dimages--5-000-eur)
10. [Résultats](#-résultats)
11. [Tests et qualité](#-tests-et-qualité)
12. [Limites et suites](#-limites-et-suites)

---

## 🎯 Objectifs et livrables

| # | Objectif (mail de Clara Martin) | Livrable | Emplacement |
|---|---|---|---|
| 1 | Explorer les images, vérifier qualité/format | Notebook préparation | `notebooks/01_exploration_preparation.ipynb` |
| 2 | Extraire des features (ResNet pré-entraîné) + preprocessing | Notebook features | `notebooks/02_extraction_features.ipynb` |
| 3 | Clustering exploratoire (K-Means, DBSCAN...) + ARI | Notebook non supervisé | `notebooks/03_clustering_non_supervise.ipynb` |
| 4 | Semi-supervisé (CNN faible → fort vs supervisé) | Notebook semi-supervisé | `notebooks/04_semi_supervise.ipynb` |
| 5 | Support + recommandations de passage à l'échelle | Plan de présentation + modèle de coûts | `docs/presentation_outline.md`, `reports/` |
| 6 | Fiche d'auto-évaluation | Correspondance indicateurs ↔ preuves | `docs/autoevaluation_checklist.md` |

**Contraintes respectées** : Python, plusieurs algorithmes comparés, métriques justifiées par le risque clinique, *definition of done* explicite ([docs/definition_of_done.md](docs/definition_of_done.md)).

## 🗂️ Données

Source : « Jeu de données d'images cérébrales pour la détection de tumeurs » (usage académique libre) - 1 500 images JPEG 512×512 : **1 400 non étiquetées**, **100 étiquetées** (`Normal` / `Cancer`).

Arborescence attendue (non versionnée) :

```text
data/raw/
├── labeled/
│   ├── normal/   *.jpg
│   └── cancer/   *.jpg
└── unlabeled/    *.jpg
```

> 📝 L'énoncé de la mission évoque du PNG + métadonnées, le descriptif du jeu annonce du JPEG : le notebook 01 **vérifie** format, résolution et canaux. Sans le vrai jeu, `docker compose run --rm demo-data` génère des images **synthétiques** (smoke test uniquement).

## 🏗️ Architecture du projet

```text
brainscanai/
├── README.md
├── LICENSE
├── Makefile                      # raccourcis optionnels (dev local / Linux-macOS) ; Docker n'en a pas besoin
├── pyproject.toml                # dépendances, groupes dev/notebooks (uv), ruff, pytest
├── uv.lock                       # versions verrouillées (reproductibilité)
├── .python-version               # version de Python utilisée par uv
├── Dockerfile                    # multi-stage : deps → runtime / dev, torch CPU ou CUDA
├── docker-compose.yml            # un service par étape + test + jupyter + profil GPU
├── .dockerignore
├── .gitignore                    # données et poids jamais versionnés
├── .github/
│   ├── workflows/ci.yml          # uv : lint + tests + notebooks, puis build Docker
│   └── pull_request_template.md
├── configs/
│   └── config.yaml               # chemins, backbone, clustering, CNN, budget (source unique de vérité)
├── data/
│   ├── raw/{labeled,unlabeled}/  # images brutes (hors Git)
│   ├── processed/                # manifeste nettoyé, stats qualité, assignations de clusters
│   └── features/                 # embeddings (.npz)
├── notebooks/                    # 01 → 04 (générés par scripts/build_notebooks.py, sans sorties)
├── src/brainscanai/
│   ├── config.py · utils.py
│   ├── data/            loader.py · quality.py · split.py · dataset.py
│   ├── features/        extractor.py (PyTorch) · io.py
│   ├── clustering/      reduce.py · cluster.py · evaluate.py
│   ├── semi_supervised/ weak_labels.py · sklearn_methods.py · embedding_experiment.py
│   │                    models.py · train.py · cnn_experiment.py
│   ├── evaluation/      metrics.py · plots.py
│   └── scale_up/        cost.py
├── scripts/                      # CLI : inspect_data, extract_features, run_clustering,
│                                 #       run_semi_supervised, estimate_scale_up, make_demo_data,
│                                 #       build_notebooks, check_notebooks
├── tests/                        # pytest (qualité, split anti-fuite, clustering, métriques, coûts)
├── docs/                         # workflow Git, definition of done, auto-évaluation, plan de soutenance
└── reports/{figures,tables,presentation}/
```

## 🔬 Pipeline et méthodologie

```mermaid
flowchart LR
    A[(IRM brutes<br/>1500 images)] --> B[1. Qualité & nettoyage<br/>doublons, outliers, biais]
    B --> C[2. Embeddings<br/>ResNet50 gelé]
    C --> D[Standardisation + PCA]
    D --> E[3. Clustering<br/>KMeans · GMM · Agglo · DBSCAN]
    E --> F[Labels faibles<br/>weak_pool]
    B --> G{Split stratifié<br/>labellisées}
    G --> H[strong_train]
    G --> I[test<br/>jamais vu]
    H --> F
    F --> J[4a. CNN faible]
    J --> K[4b. Fine-tuning fort]
    H --> L[4c. CNN fort seul]
    K --> M[(Évaluation sur test<br/>recall · F1 · ROC)]
    L --> M
    I --> M
    M --> N[5. Reco. passage à l'échelle<br/>4 M images · 5 000 EUR]
```

**Règles anti-fuite (non négociables)**
- Les images labellisées sont scindées en `strong_train` / `test` (stratifié) ; les non labellisées forment `weak_pool`. Un garde-fou (`assert_no_overlap`) interdit tout chevauchement.
- Le mapping *cluster → classe* est appris **uniquement** sur `strong_train`.
- Les labels « faibles » restent dans un jeu séparé du jeu « fort » et sont réintroduits avec un poids réduit ou en pré-entraînement.
- Le test n'est jamais utilisé pour l'entraînement, le choix de seuil ou le mapping. Résultats moyennés sur **5 répétitions** (seeds/splits différents) car le test est petit (≈ 30 images).

**Approches comparées** : régression logistique supervisée (référence), labels faibles seuls, faible + fort pondéré, *self-training* (pseudo-labelling à seuil de confiance), *Label Spreading*, *Label Propagation*, et CNN (ResNet18) **faible → fort** vs **fort seul**.

## 📐 Diagrammes UML

### Diagramme de cas d'utilisation

```mermaid
flowchart LR
    DS([👩‍💻 Data Scientist])
    RM([👩‍💼 Responsable DS])
    RAD([🩺 Radiologue])
    subgraph SYS[Système BrainScanAI]
        UC1(Contrôler la qualité des images)
        UC2(Extraire les embeddings)
        UC3(Clusteriser et labelliser faiblement)
        UC4(Entraîner et comparer les modèles)
        UC5(Estimer le coût du passage à l'échelle)
        UC6(Valider un échantillon de prédictions)
        UC7(Consulter résultats et recommandations)
    end
    DS --> UC1 & UC2 & UC3 & UC4 & UC5
    RAD --> UC6
    RM --> UC7
    UC4 -. include .-> UC3
    UC3 -. include .-> UC2
    UC2 -. include .-> UC1
    UC6 -. extend .-> UC4
```

### Diagramme de classes

```mermaid
classDiagram
    direction LR
    class MRIDataset {
        +list paths
        +list labels
        +Compose transform
        +__getitem__(idx) tuple
    }
    class FeatureExtractor {
        +str backbone
        +str layer
        +str device
        +extract(paths) ndarray
    }
    class ClusterResult {
        <<dataclass>>
        +str name
        +ndarray labels
        +ndarray confidence
        +dict params
    }
    class TrainConfig {
        <<dataclass>>
        +int epochs
        +float lr
        +int batch_size
        +int patience
    }
    class ScaleUpAssumptions {
        <<dataclass>>
        +int n_images
        +float budget_eur
        +float gpu_eur_per_hour
        +from_config(cfg) ScaleUpAssumptions
    }
    class data_loader {
        <<module>>
        +build_manifest(cfg) DataFrame
    }
    class data_quality {
        <<module>>
        +compute_image_stats(manifest) DataFrame
        +flag_outliers(stats) DataFrame
        +flag_duplicates(stats) DataFrame
        +clean_manifest(manifest, stats) tuple
    }
    class data_split {
        <<module>>
        +assign_roles(manifest) DataFrame
        +assert_no_overlap(df) None
    }
    class clustering {
        <<module>>
        +run_all(X, k) dict
        +map_clusters_to_classes(c, y) dict
        +evaluate_all(results, X, y) DataFrame
    }
    class weak_labels {
        <<module>>
        +build_weak_labels(df, result) DataFrame
    }
    class train {
        <<module>>
        +train_model(model, paths, labels, cfg) list
        +predict_proba(model, paths, cfg) ndarray
    }
    class experiments {
        <<module>>
        +run_embedding_experiment(df, X, cfg) tuple
        +run_cnn_experiment(df, X, cfg) tuple
    }
    class metrics {
        <<module>>
        +compute_metrics(y, p) dict
        +bootstrap_ci(y, p) tuple
    }
    class cost {
        <<module>>
        +estimate(assumptions) dict
        +validation_sample_size() dict
    }
    data_loader ..> data_quality : manifeste
    data_quality ..> data_split : manifeste nettoyé
    FeatureExtractor ..> MRIDataset : utilise
    clustering --> ClusterResult : produit
    weak_labels ..> ClusterResult : consomme
    weak_labels ..> data_split : rôles
    experiments ..> weak_labels
    experiments ..> train
    experiments ..> metrics
    train ..> MRIDataset
    train --> TrainConfig : configure
    cost --> ScaleUpAssumptions : paramètre
```

### Diagramme de paquetages / composants

```mermaid
flowchart TB
    subgraph CLI[scripts / notebooks - couche d'orchestration]
        S1[inspect_data]
        S2[extract_features]
        S3[run_clustering]
        S4[run_semi_supervised]
        S5[estimate_scale_up]
    end
    subgraph CORE[src/brainscanai - couche métier]
        D[data]
        F[features]
        C[clustering]
        SS[semi_supervised]
        E[evaluation]
        SC[scale_up]
        CFG[config + utils]
    end
    subgraph EXT[Bibliothèques]
        TORCH[PyTorch / torchvision]
        SK[scikit-learn / SciPy]
        VIZ[matplotlib / seaborn]
        PD[pandas / NumPy / Pillow]
    end
    subgraph STORE[Stockage - hors Git]
        RAW[(data/raw)]
        PROC[(data/processed)]
        FEAT[(data/features)]
        REP[(reports)]
    end
    CLI --> CORE
    CORE --> CFG
    F --> TORCH
    SS --> TORCH
    C --> SK
    SS --> SK
    E --> VIZ
    D --> PD
    D --> RAW
    D --> PROC
    F --> FEAT
    E --> REP
    SC --> REP
```

### Diagramme de séquence - exécution du pipeline

```mermaid
sequenceDiagram
    autonumber
    actor DS as Data Scientist
    participant NB as Notebook / CLI
    participant DAT as data
    participant FE as FeatureExtractor
    participant CL as clustering
    participant WL as weak_labels
    participant TR as train
    participant MT as metrics
    DS->>NB: docker compose run --rm pipeline
    NB->>DAT: build_manifest, compute_image_stats
    DAT-->>NB: manifest_clean.csv
    NB->>FE: extract(paths)
    FE-->>NB: embeddings.npz
    NB->>CL: standardize, PCA, run_all(k=2)
    CL-->>NB: ClusterResult x4 + ARI
    loop 5 répétitions (seeds)
        NB->>DAT: assign_roles(strong_train, test, weak_pool)
        NB->>WL: build_weak_labels (mapping sur strong_train)
        WL-->>NB: jeu faiblement labellisé
        NB->>TR: train(weak) puis fine-tune(strong)
        NB->>TR: train(strong seul)
        TR-->>NB: probabilités sur test
        NB->>MT: compute_metrics
    end
    MT-->>NB: moyenne ± écart-type, IC bootstrap
    NB-->>DS: reports/tables + reports/figures
```

### Diagramme d'activité - décisions clés

```mermaid
flowchart TD
    START((Début)) --> L[Charger et inspecter les images]
    L --> Q{Fichier corrompu<br/>ou doublon ?}
    Q -- oui --> RM[Retirer / journaliser]
    Q -- non --> OUT{Outlier ?}
    RM --> OUT
    OUT -- non labellisée --> RM2[Retirer]
    OUT -- labellisée --> FL[Signaler, conserver]
    OUT -- non --> EMB
    RM2 --> EMB[Extraire embeddings]
    FL --> EMB
    EMB --> CLU[Clustering k=2, 4 algorithmes]
    CLU --> ARI{ARI exploitable ?}
    ARI -- non --> ALT[Changer de couche, k, ou méthode]
    ALT --> EMB
    ARI -- oui --> WEAK[Labels faibles confiants]
    WEAK --> TRAIN[Entraîner : faible → fort, fort seul]
    TRAIN --> EVAL{Recall ≥ supervisé<br/>et AUC ≥ 0,85 ?}
    EVAL -- oui --> SCALE[Recommander le passage à l'échelle]
    EVAL -- non --> NEG[Documenter le résultat négatif<br/>et pistes d'amélioration]
    SCALE --> END((Fin))
    NEG --> END
```

### Diagramme d'états - cycle de vie d'une image

```mermaid
stateDiagram-v2
    [*] --> Brute
    Brute --> Rejetee : corrompue / doublon / outlier non labellisé
    Brute --> Labellisee : label expert
    Brute --> NonLabellisee : sans label
    Labellisee --> StrongTrain : split stratifié
    Labellisee --> Test : split stratifié
    NonLabellisee --> WeakPool
    WeakPool --> FaiblementLabellisee : cluster → classe (mapping strong_train)
    FaiblementLabellisee --> EntrainementFaible
    StrongTrain --> FineTuning
    EntrainementFaible --> FineTuning
    Test --> Evaluation : jamais vu à l'entraînement
    FineTuning --> Evaluation
    Rejetee --> [*]
    Evaluation --> [*]
```

## 🌿 Stratégie Git / GitHub

Git Flow allégé : une branche `feature/*` par étape, PR vers `develop`, versions livrables taguées sur `main`. Détails : [docs/git_workflow.md](docs/git_workflow.md).

| Branche | Étape | Contenu |
|---|---|---|
| `feature/00-project-setup` | Cadrage | Arborescence, config, CI, utilitaires, graphiques |
| `feature/01-data-exploration` | 1 | Manifeste, qualité, split anti-fuite, notebook 01 |
| `feature/02-feature-extraction` | 2 | Backbone gelé, embeddings, notebook 02 |
| `feature/03-clustering` | 3 | PCA/t-SNE, clustering, ARI, labels faibles, notebook 03 |
| `feature/04-semi-supervised` | 4 | Métriques, méthodes sklearn, CNN faible→fort, notebook 04 |
| `feature/05-scale-up` | Échelle | Modèle de coûts 4 M images / 5 000 EUR |
| `feature/06-docs-presentation` | Livrables | Docs, auto-évaluation, plan de soutenance, README |
| `feature/07-docker-uv` | Industrialisation | uv (`uv.lock`), Dockerfile multi-stage, docker-compose, CI mise à jour |
| `release/v1.0.0`, `release/v1.1.0` → `main` | Release | Tags `v1.0.0`, `v1.1.0` |

```mermaid
gitGraph
    commit id: "init"
    branch develop
    checkout develop
    commit id: "chore: gitignore, licence"
    branch "feature/00-project-setup"
    commit id: "feat: structure, config, CI"
    checkout develop
    merge "feature/00-project-setup"
    branch "feature/01-data-exploration"
    commit id: "feat: qualité, split, nb 01"
    checkout develop
    merge "feature/01-data-exploration"
    branch "feature/02-feature-extraction"
    commit id: "feat: ResNet gelé, nb 02"
    checkout develop
    merge "feature/02-feature-extraction"
    branch "feature/03-clustering"
    commit id: "feat: clustering, ARI, nb 03"
    checkout develop
    merge "feature/03-clustering"
    branch "feature/04-semi-supervised"
    commit id: "feat: semi-supervisé, nb 04"
    checkout develop
    merge "feature/04-semi-supervised"
    branch "feature/05-scale-up"
    commit id: "feat: modèle de coûts"
    checkout develop
    merge "feature/05-scale-up"
    branch "feature/06-docs-presentation"
    commit id: "docs: soutenance, README"
    checkout develop
    merge "feature/06-docs-presentation"
    branch "release/v1.0.0"
    commit id: "chore: release 1.0.0"
    checkout main
    merge "release/v1.0.0" tag: "v1.0.0"
    checkout develop
    merge "release/v1.0.0"
    branch "feature/07-docker-uv"
    commit id: "feat: uv + Docker + CI"
    checkout develop
    merge "feature/07-docker-uv"
    branch "release/v1.1.0"
    commit id: "chore: release 1.1.0"
    checkout main
    merge "release/v1.1.0" tag: "v1.1.0"
```

## 🐳 Installation et exécution (Docker uniquement)

**Seul Docker est requis** sur votre machine (Windows, macOS ou Linux). **Aucune installation locale de Python, uv, PyTorch ou `make`** : toutes les dépendances (verrouillées par `uv.lock`) sont installées **dans l'image** au premier build.

**Prérequis** : [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et lancé (Windows : backend WSL2).

```powershell
git clone <votre-repo>.git ; cd brainscanai

# 1) Données : copier le jeu dans data/raw/ (voir arborescence) - OU générer un jeu synthétique de test
docker compose run --rm demo-data

# 2) Tout le pipeline en une commande (le 1er lancement construit l'image : plusieurs minutes, PyTorch est lourd)
docker compose run --rm pipeline

# ... ou étape par étape
docker compose run --rm inspect       # 1. qualité + nettoyage
docker compose run --rm features      # 2. embeddings ResNet
docker compose run --rm cluster       # 3. clustering + ARI + figures
docker compose run --rm semi          # 4. supervisé vs semi-supervisé (CNN inclus)
docker compose run --rm scaleup       # 5. budget 4 M d'images

# Tests et notebooks, toujours dans un conteneur
docker compose run --rm test          # ruff + pytest
docker compose up jupyter             # JupyterLab -> http://localhost:8888
```

- Les dossiers `data/` et `reports/` de votre machine sont **montés en volumes** : résultats (`reports/figures`, `reports/tables`) et embeddings apparaissent directement chez vous ; les données ne sont jamais copiées dans l'image. Les poids ImageNet sont mis en cache dans `data/.torch_cache`.
- Modifier `src/` ou `scripts/` ? Reconstruire : `docker compose build` (le service `jupyter` monte le code en direct).
- Paramètres : `configs/config.yaml` (monté en lecture seule, pas de rebuild nécessaire).
- **GPU** (NVIDIA Container Toolkit requis) : `$env:TORCH_VARIANT="cu121"; docker compose build inspect` puis `docker compose --profile gpu run --rm features-gpu` (PowerShell ; sous bash : `TORCH_VARIANT=cu121 docker compose ...`).
- Test hors-ligne sans poids pré-entraînés : `docker compose run --rm features python scripts/extract_features.py --no-pretrained`.

```mermaid
flowchart LR
    subgraph IMG[Image Docker multi-stage]
        B[base<br/>python-slim + uv] --> D[deps<br/>uv sync --frozen<br/>+ torch cpu/cuda]
        D --> R[runtime<br/>src, scripts, configs<br/>user non-root]
        D --> V[dev<br/>pytest, ruff, jupyter]
    end
    LOCK[(pyproject.toml<br/>uv.lock)] --> D
    HOSTD[(./data)] -. volume .-> R
    HOSTR[(./reports)] -. volume .-> R
    R --> OUT[Embeddings, tables,<br/>figures]
    V --> CI[CI GitHub Actions<br/>build + tests]
```

### Option : développement local sans Docker (facultatif)

Réservé à l'autocomplétion IDE ou au débogage : `uv sync --group dev --group notebooks`, puis `uv run python scripts/<script>.py` (ou `make <cible>` si `make` est disponible). Ce n'est **pas** nécessaire pour exécuter le projet.

## 📏 Métriques et definition of done

**Classe positive = cancer.** L'erreur la plus grave est le **faux négatif** (tumeur non détectée) :

| Rôle | Métrique | Raison |
|---|---|---|
| **Principale** | Rappel (sensibilité) « cancer » | Minimiser les tumeurs manquées |
| Garde-fou | Précision, spécificité | Éviter de submerger les radiologues de faux positifs |
| Synthèse | F1 et F2 (pondère le rappel) | Compromis |
| Indépendante du seuil | AUC-ROC, AUC-PR | Comparer les modèles sans fixer le seuil |
| Clustering | ARI, NMI, pureté, silhouette | Externe (sur les 100 labels) et interne |
| Incertitude | IC 95 % bootstrap | Test ≈ 30 images : un chiffre seul serait trompeur |

L'accuracy est rapportée mais **jamais** utilisée seule. Le seuil de décision (ex. rappel ≥ 95 %) se calibre sur la validation, pas sur le test.

**Definition of done** ([détail](docs/definition_of_done.md)) : pipeline reproductible, zéro fuite, ≥ 3 clusterings et ≥ 3 approches semi-supervisées comparées sur 5 répétitions, semi-supervisé ≥ supervisé en rappel avec AUC-ROC ≥ 0,85 - **ou** résultat négatif analysé et argumenté.

## 💶 Passage à l'échelle (4 M d'images / 5 000 EUR)

- Budget disponible : **0,00125 EUR/image** contre **0,20 EUR/image** au pilote (300 EUR / 1 500) → extrapolation naïve = **800 000 EUR** : un étiquetage payé à l'image est exclu.
- **Faisable sous conditions** : labellisation automatique auto-hébergée (embeddings + pseudo-labels), puis budget restant consacré à une **revue experte ciblée** (validation statistique ≈ 146 images pour un rappel de 95 % à ± 5 %, + cas à faible confiance), suivi du décalage de domaine, statut « R&D » des labels.
- Chiffrage et sensibilité : `python scripts/estimate_scale_up.py` → `reports/tables/scale_up_*.csv`. **Tous les tarifs sont des hypothèses** (`configs/config.yaml`, section `scale_up`) à remplacer par des devis réels.
- Synthèse : [docs/scale_up_recommendations.md](docs/scale_up_recommendations.md).

## 📊 Résultats

> Les résultats sur le jeu réel **ne sont pas encore produits** : le dépôt livre le pipeline, pas des chiffres inventés. Après exécution, reporter ici les tableaux générés dans `reports/tables/`.

| Approche | Rappel cancer | F1 | AUC-ROC |
|---|---|---|---|
| Supervisé (labels forts seuls) | _à compléter_ | _à compléter_ | _à compléter_ |
| Semi-supervisé (meilleure approche) | _à compléter_ | _à compléter_ | _à compléter_ |

## ✅ Tests et qualité

```bash
docker compose run --rm test    # ruff + pytest dans le conteneur : qualité/doublons, split anti-fuite, clustering, mapping, métriques, coûts, pipeline embeddings
```
La CI GitHub Actions (uv, `--frozen`) exécute lint + tests + validation des notebooks (sans sorties), puis construit l'image Docker, à chaque push/PR. Les tests PyTorch sont ignorés automatiquement si `torch` n'est pas installé.

## 🚧 Limites et suites

- **100 labels** : variance élevée, test ≈ 30 images → conclusions prudentes, IC larges, répétitions indispensables.
- Les clusters peuvent refléter l'intensité, la coupe ou le scanner plutôt que la pathologie (à vérifier visuellement, notebook 03).
- Risque de décalage de domaine entre labellisées et non labellisées (test KS, notebook 01).
- Pistes : couches intermédiaires, modèles pré-entraînés médicaux, FixMatch/MixMatch, apprentissage actif, calibration, explicabilité (Grad-CAM).

---
<sub>Licence MIT · Données : usage académique libre · Auteur : Data Scientist junior, CurelyticsIA.</sub>
