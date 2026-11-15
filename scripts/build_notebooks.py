"""Régénère les 4 notebooks (sans sorties) à partir de ce script : source unique, diff lisibles."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = '''import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from brainscanai.config import load_config, resolve_path
from brainscanai.utils import set_seed

cfg = load_config()
set_seed(cfg["seed"])
PROC, FEAT = resolve_path(cfg, "processed", create=True), resolve_path(cfg, "features", create=True)
FIG, TAB = resolve_path(cfg, "figures", create=True), resolve_path(cfg, "tables", create=True)'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    n["cells"] = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    return n


NOTEBOOKS = {}

NOTEBOOKS["01_exploration_preparation.ipynb"] = [
    ("md", "# 01 - Exploration visuelle & préparation des données\n**Branche** : `feature/01-data-exploration`\n\n**Objectifs** : vue d'ensemble du jeu, qualité (corruption, doublons, outliers), homogénéité des sources, nettoyage traçable.\n\n> Hypothèse d'arborescence : `data/raw/labeled/{normal,cancer}/` et `data/raw/unlabeled/`."),
    ("code", SETUP + "\nfrom brainscanai.data.loader import build_manifest\nfrom brainscanai.data.quality import *\nfrom brainscanai.evaluation.plots import plot_sample_grid, save_fig"),
    ("md", "## 1. Manifeste & structure"),
    ("code", "manifest = build_manifest(cfg)\nprint(manifest.shape)\nmanifest.groupby(['source', 'label_name']).size().rename('n_images').to_frame()"),
    ("md", "## 2. Exemples d'images (par classe)"),
    ("code", "rng = np.random.default_rng(cfg['seed'])\nfor name in ['normal', 'cancer', 'unlabeled']:\n    sub = manifest[manifest.label_name == name]\n    pick = sub.sample(min(5, len(sub)), random_state=cfg['seed'])\n    fig = plot_sample_grid(pick.path.tolist(), [name] * len(pick))\n    fig.suptitle(name, y=1.02); save_fig(fig, f'samples_{name}', FIG); plt.show()"),
    ("md", "## 3. Résolution, canaux, format, intensité, netteté\nLe fichier de description annonce 512×512 JPEG ; l'énoncé de la mission parle de PNG : **on vérifie plutôt que de supposer**."),
    ("code", "stats = compute_image_stats(manifest)\nprint(stats[['width', 'height', 'mode', 'format']].value_counts(dropna=False))\nstats[['mean', 'std', 'sharpness', 'black_ratio']].describe().round(2)"),
    ("md", "## 4. Outliers, doublons, fichiers corrompus\nZ-score robuste (médiane/MAD). **Règle** : jamais de suppression d'une image *labellisée* (signalement seulement) ; doublons : on garde la copie labellisée."),
    ("code", "q = cfg['data']['quality']\nstats = flag_duplicates(flag_outliers(stats, z=q['outlier_robust_z']).assign(md5=lambda d: d['md5'].fillna('x_' + d['image_id'].astype(str))))\nprint('corrompues :', int(stats.corrupt.sum()), '| doublons :', int(stats.is_duplicate.sum()), '| outliers :', int(stats.is_outlier.sum()))\nout = stats[stats.is_outlier | stats.is_duplicate]\nout.head(10)"),
    ("code", "fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))\nfor ax, col in zip(axes, ['mean', 'std', 'sharpness']):\n    for src, g in stats.groupby('source'):\n        ax.hist(g[col].dropna(), bins=30, alpha=.6, density=True, label=src)\n    ax.set_title(col); ax.legend()\nsave_fig(fig, 'intensity_by_source', FIG); plt.show()"),
    ("md", "## 5. Biais potentiels : labellisées vs non labellisées (test de Kolmogorov-Smirnov)\nUn décalage de distribution (scanner, protocole) fausserait la propagation des labels."),
    ("code", "domain_shift_report(stats)"),
    ("md", "## 6. Nettoyage & sauvegarde"),
    ("code", "cleaned, removed = clean_manifest(manifest, stats, q['drop_duplicates'], q['drop_unlabeled_outliers'])\nprint(f'{len(manifest)} -> {len(cleaned)} images ; retirées : {len(removed)}')\nstats.to_csv(PROC / 'image_stats.csv', index=False)\ncleaned.to_csv(PROC / 'manifest_clean.csv', index=False)\nremoved.to_csv(PROC / 'removed_images.csv', index=False)\nremoved.reason.value_counts()"),
    ("md", "## Observations\n_À compléter après exécution sur le vrai jeu : résolution constante ? canaux ? outliers (artefacts, coupes quasi vides) ? écart labellisé / non labellisé ?_"),
]

NOTEBOOKS["02_extraction_features.ipynb"] = [
    ("md", "# 02 - Prétraitement & extraction d'embeddings (ResNet pré-entraîné)\n**Branche** : `feature/02-feature-extraction`\n\nBackbone **gelé** ; images en niveaux de gris répliquées sur 3 canaux, redimensionnées à 224×224, normalisées ImageNet. Pas d'augmentation à l'extraction."),
    ("code", SETUP + "\nfrom brainscanai.features.extractor import FeatureExtractor\nfrom brainscanai.features.io import save_features, load_features\nfrom brainscanai.clustering.reduce import standardize, pca_reduce"),
    ("code", "manifest = pd.read_csv(PROC / 'manifest_clean.csv')\nfc = cfg['features']\n# Passez pretrained=False pour tester hors-ligne (embeddings non pertinents)\next = FeatureExtractor(fc['backbone'], pretrained=fc['pretrained'], layer=fc['layer'], device=fc['device'])\nprint('device :', ext.device)"),
    ("code", "# Astuce : tester d'abord sur un échantillon (ex. manifest.sample(64)) avant le jeu complet\nX = ext.extract(manifest.path, cfg['data']['image_size'], fc['batch_size'], fc['num_workers'])\nprint(X.shape, X.dtype)"),
    ("md", "## Contrôles de sortie (exploitabilité)"),
    ("code", "assert np.isfinite(X).all(), 'NaN/Inf dans les embeddings'\nprint('dimensions constantes :', int((X.std(0) < 1e-8).sum()))\nXs, _ = standardize(X)\nZ, pca = pca_reduce(Xs, cfg['clustering']['pca_components'], cfg['seed'])\nprint(f'PCA : {X.shape[1]} -> {Z.shape[1]} dims ({pca.explained_variance_ratio_.sum():.1%} de variance)')\nplt.plot(np.cumsum(pca.explained_variance_ratio_)); plt.xlabel('composantes'); plt.ylabel('variance cumulée'); plt.show()"),
    ("md", "## Comparaison de couches d'extraction (optionnel)\nUne couche intermédiaire (`layer3`) capte parfois mieux les textures que la dernière couche."),
    ("code", "# for layer in ['layer3', 'avgpool']:\n#     e = FeatureExtractor(fc['backbone'], True, layer, fc['device'])\n#     Xl = e.extract(manifest.path); print(layer, Xl.shape)  # puis comparer l'ARI (notebook 03)"),
    ("code", "path = save_features(manifest, X, FEAT)\nXb, mb = load_features(path, manifest)   # relecture = vérification d'alignement\nassert (mb.image_id.values == manifest.image_id.values).all()\npath"),
]

NOTEBOOKS["03_clustering_non_supervise.ipynb"] = [
    ("md", "# 03 - Analyse non supervisée & labellisation faible\n**Branche** : `feature/03-clustering`\n\nPCA → K-Means / GMM / Agglomératif / DBSCAN (k = 2 comme le jeu labellisé) ; ARI calculé sur les 100 images labellisées. **Ne pas attendre un ARI parfait.**"),
    ("code", SETUP + "\nfrom brainscanai.features.io import load_features\nfrom brainscanai.clustering.reduce import *\nfrom brainscanai.clustering.cluster import run_all\nfrom brainscanai.clustering.evaluate import evaluate_all, purity\nfrom brainscanai.evaluation.plots import plot_embedding_2d, save_fig\nfrom sklearn.decomposition import PCA\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import adjusted_rand_score, silhouette_score\nfrom sklearn.preprocessing import normalize"),
    ("code", "cc = cfg['clustering']\nmanifest = pd.read_csv(PROC / 'manifest_clean.csv')\nX, m = load_features(FEAT / 'embeddings.npz', manifest)\nXs, _ = standardize(X)\nZ, pca = pca_reduce(Xs, 0.95, cfg['seed'])   # PCA exploratoire (95 % de variance) : sert à l'analyse de sensibilité\nZ.shape"),
    ("md", "## Choix de la dimension de la PCA (analyse de sensibilité)\nLa PCA à 95 % de variance conserve beaucoup de dimensions (663 pour ~1 300 images) : trop pour un clustering (concentration des distances, bruit). On compare plusieurs dimensions, avec et sans normalisation L2, sur la silhouette (sans labels) et l'ARI (sur les labellisées)."),
    ("code", "mask = m.label.to_numpy() >= 0\npca_full = PCA(random_state=cfg['seed']).fit(Xs)      # une seule PCA, on tronque ensuite\nZfull = pca_full.transform(Xs)\nrows = []\nfor d in [5, 10, 20, 30, 50, 100, 200, Z.shape[1]]:\n    Zd = Zfull[:, :d]\n    for l2 in [False, True]:                           # normaliser L2 = géométrie « cosinus »\n        Zk = normalize(Zd) if l2 else Zd\n        lab = KMeans(2, n_init=20, random_state=cfg['seed']).fit_predict(Zk)\n        rows.append(dict(dims=d, l2_norm=l2, var=pca_full.explained_variance_ratio_[:d].sum(),\n                         silhouette=silhouette_score(Zk, lab),\n                         ARI=adjusted_rand_score(m.label[mask], lab[mask])))\npca_sensitivity = pd.DataFrame(rows).round(3)\npca_sensitivity.to_csv(TAB / 'pca_sensitivity.csv', index=False)\npca_sensitivity"),
    ("md", "## Espace de clustering retenu\nLa dimension (`pca_components`) et la normalisation L2 (`l2_normalize`) sont lues dans `configs/config.yaml` : **le même espace est utilisé par les scripts et le notebook 04** pour fabriquer les labels faibles. Les ARI sur 100 images ont une forte incertitude (écart-type de l'ordre de 0,1) : ne pas sur-interpréter des différences de quelques centièmes."),
    ("code", "Zc, _ = prepare_clustering_space(X, cc['pca_components'], cc['l2_normalize'], cfg['seed'])\nprint('espace de clustering :', Zc.shape, '| normes ~ 1 :', bool(np.allclose(np.linalg.norm(Zc, axis=1), 1)) if cc['l2_normalize'] else 'L2 désactivée')"),
    ("md", "## Visualisation 2D (t-SNE sur l'espace retenu) - vérité terrain"),
    ("code", "coords = tsne_embed(Zc, cc['tsne_perplexity'], cfg['seed'])\nplot_embedding_2d(coords, m.label_name, 'Classes connues (labellisées) vs inconnues'); plt.show()"),
    ("md", "## Comparaison de 4 algorithmes (dans l'espace retenu)\nÀ comparer aussi avec le même tableau sans L2 (`l2_normalize: false`) pour documenter l'effet de la normalisation. Renseigner ensuite `selected_method` dans la config."),
    ("code", "results = run_all(Zc, cc['n_clusters'], cfg['seed'], cc['dbscan_min_samples'])\ntable = evaluate_all(results, Zc, m.label.to_numpy())\ntable.to_csv(TAB / 'clustering_comparison.csv')\ntable"),
    ("code", "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\nfor ax, (name, r) in zip(axes, results.items()):\n    plot_embedding_2d(coords, r.labels, f\"{name} (ARI={table.loc[name, 'ARI']:.2f})\", ax)\nsave_fig(fig, 'clustering_tsne', FIG); plt.show()"),
    ("md", "## Sensibilité au nombre de clusters k\nL'ARI compare à 2 classes : il *pénalise* k > 2 par construction, et peut varier de façon non monotone. La **pureté** (chaque cluster assigné à sa classe majoritaire) est plus adaptée pour juger un sur-découpage."),
    ("code", "rows = []\nfor k in [2, 3, 4, 5, 8]:\n    lab = KMeans(k, n_init=20, random_state=cfg['seed']).fit_predict(Zc)\n    rows.append(dict(k=k, silhouette=silhouette_score(Zc, lab), ARI=adjusted_rand_score(m.label[mask], lab[mask]),\n                     purity=purity(m.label[mask].to_numpy(), lab[mask])))\npd.DataFrame(rows).round(3)"),
    ("md", "## Interprétation\n_À compléter : quelle méthode est la meilleure (ARI, pureté) ? Les clusters suivent-ils la pathologie ou plutôt l'intensité/la coupe/le scanner ? Vérifier en comparant clusters et statistiques d'image (`mean`, `black_ratio`) et en regardant des images de chaque cluster._"),
    ("code", "st = pd.read_csv(PROC / 'image_stats.csv').set_index('image_id')\nbest = results[cc['selected_method']]\nchk = pd.DataFrame({'cluster': best.labels, 'mean': st.loc[m.image_id, 'mean'].to_numpy(),\n                    'black_ratio': st.loc[m.image_id, 'black_ratio'].to_numpy(),\n                    'sharpness': st.loc[m.image_id, 'sharpness'].to_numpy()})\nprint(chk.groupby('cluster').median().round(2))   # des clusters très différents en luminosité/netteté = biais visuel\nfrom brainscanai.evaluation.plots import plot_sample_grid\nfor c in sorted(set(best.labels[best.labels >= 0])):\n    idx = np.where(best.labels == c)[0][:5]\n    plot_sample_grid(m.path.iloc[idx].tolist(), [f'cluster {c}'] * len(idx)); plt.show()"),
    ("md", "## Labellisation faible\nLe mapping cluster → classe est appris sur `strong_train` **uniquement** (voir notebook 04, où le split est créé). Les labels faibles restent dans un jeu séparé (`weak_pool`), jamais mélangé aux labels forts."),
]

NOTEBOOKS["04_semi_supervise.ipynb"] = [
    ("md", "# 04 - Apprentissage semi-supervisé & comparaison\n**Branche** : `feature/04-semi-supervised`\n\n**Definition of done** : le meilleur modèle semi-supervisé atteint un rappel « cancer » ≥ celui du supervisé seul (moyenne sur 5 répétitions), avec un F1 au moins égal et un AUC-ROC ≥ 0,85 - sinon on conclut honnêtement que le semi-supervisé n'apporte rien ici.\n\n**Métrique principale** : rappel (sensibilité) sur « cancer » - un faux négatif (tumeur manquée) est l'erreur la plus grave. Compléments : F1/F2, précision, spécificité, AUC-ROC/PR. Accuracy seule = trompeuse (test petit, classes éventuellement déséquilibrées)."),
    ("code", SETUP + "\nfrom brainscanai.features.io import load_features\nfrom brainscanai.data.split import assign_roles\nfrom brainscanai.semi_supervised.embedding_experiment import run_embedding_experiment\nfrom brainscanai.evaluation.metrics import *\nfrom brainscanai.evaluation.plots import *"),
    ("code", "manifest = pd.read_csv(PROC / 'manifest_clean.csv')\nX, df = load_features(FEAT / 'embeddings.npz', manifest)\nd = assign_roles(df, cfg['data']['test_size'], cfg['seed'])\nd.groupby(['role', 'label_name']).size().unstack(fill_value=0)"),
    ("md", "## A. Méthodes sur embeddings (rapide)\nSupervisé, labels faibles, faible+fort, Self-training (pseudo-labelling), Label Spreading, Label Propagation - 5 répétitions (splits différents)."),
    ("code", "summary, probs = run_embedding_experiment(df, X, cfg)\nsummary.to_csv(TAB / 'semi_supervised_embeddings.csv', index=False)\nsummary[['approach', 'recall_cancer_mean', 'recall_cancer_std', 'precision_cancer_mean', 'f1_mean', 'roc_auc_mean', 'accuracy_mean']].round(3)"),
    ("code", "fig = plot_metric_comparison(summary, title='Embeddings : supervisé vs semi-supervisé'); save_fig(fig, 'compare_embeddings', FIG); plt.show()"),
    ("code", "from sklearn.metrics import confusion_matrix\ncurves = {k: roc_points(probs['y_test'], v) for k, v in probs.items() if k != 'y_test'}\nplot_roc_curves(curves); plt.show()\nbest = summary.sort_values('recall_cancer_mean').iloc[-1].approach\nplot_confusion_matrix(confusion_matrix(probs['y_test'], probs[best] >= .5), cfg['data']['classes'], best); plt.show()"),
    ("md", "## B. Protocole CNN (mission) : faible → fort vs fort seul (+ pseudo-labels du modèle)\nQuatre bras comparés sur les **mêmes** splits : **A** labels faibles seuls ; **B** faible → fort (semi-supervisé) ; **C** fort seul (supervisé) ; **D** pseudo-labelling par le modèle (C prédit les images sans label, on garde les prédictions très confiantes, puis réentraînement conjoint). B et C utilisent les **mêmes hyperparamètres** pour être comparables.\nDemande un GPU idéalement. Réduire `n_repeats` et les epochs pour un essai."),
    ("code", "from brainscanai.semi_supervised.cnn_experiment import run_cnn_experiment\ncnn_summary, cnn_probs, cnn_diag = run_cnn_experiment(df, X, cfg, n_repeats=2)   # passer à 5 pour le rendu final\ncnn_summary.to_csv(TAB / 'semi_supervised_cnn.csv', index=False)\ncnn_diag.to_csv(TAB / 'cnn_diagnostics.csv', index=False)\ndisplay(cnn_diag.round(3))   # weak_label_acc_on_test = qualité des labels faibles ; n_pseudo_kept = pseudo-labels retenus pour D\ncnn_summary[['approach', 'recall_cancer_mean', 'f1_mean', 'roc_auc_mean', 'accuracy_mean']].round(3)"),
    ("code", "fig = plot_metric_comparison(cnn_summary, title='CNN : supervisé vs semi-supervisé'); save_fig(fig, 'compare_cnn', FIG); plt.show()"),
    ("md", "## C. Incertitude (test petit) & seuil orienté rappel\nAvec ~30 images de test, un IC bootstrap est indispensable ; le seuil de décision doit être calibré sur la **validation**, pas sur le test."),
    ("code", "y = cnn_probs['y_test']\nfor k in [a for a in ['A', 'B', 'C', 'D'] if a in cnn_probs]:\n    lo, hi = bootstrap_ci(y, cnn_probs[k], 'recall_cancer')\n    print(k, 'rappel IC95% = [%.2f, %.2f]' % (lo, hi))"),
    ("md", "## D. Hyperparamètres\nUn script d'ablation est fourni : `python scripts/run_ablation.py --keep 0.3 0.5 0.8 --repeats 3` (qualité vs quantité des labels faibles, mêmes splits). Grille complète à explorer (validation croisée sur `strong_train`) : `lr ∈ {3e-5, 1e-4, 3e-4}`, `epochs ∈ {10, 15, 25}`, `keep_fraction ∈ {0.6, 0.8, 1.0}`, `self_training_threshold ∈ {0.8, 0.9, 0.95}`, `n_neighbors ∈ {5, 7, 11}`. Reporter les résultats dans `reports/tables/`."),
    ("md", "## E. Conclusion & recommandations\n_À compléter avec les vrais chiffres : le semi-supervisé bat-il le supervisé ? De combien (± écart-type) ? Les labels faibles sont-ils assez propres ? Lien avec le scale-up (`reports/tables/scale_up_budget.csv`)._"),
]


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nbf.write(nb(cells), NB_DIR / name)
        print("écrit", name)


if __name__ == "__main__":
    main()
