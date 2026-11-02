# Correspondance avec la fiche d'auto-évaluation

| Indicateur de réussite | Où le justifier |
|---|---|
| Données nettoyées (manquantes, aberrantes) | `data/quality.py`, notebook 01 §4 et §6, `removed_images.csv` |
| Biais réduits (outliers, homogénéité des sources) | notebook 01 §4-5 (`domain_shift_report`) |
| Code commenté | docstrings + commentaires dans `src/`, notebooks |
| Transformations adaptées au modèle | `data/dataset.py` (RGB, resize, norm. ImageNet), notebook 02 |
| Filtrage / normalisation | `standardize`, `pca_reduce`, notebook 02 |
| Sorties (embeddings) exploitables | notebook 02 « Contrôles de sortie », `features/io.py` |
| Plusieurs méthodes de clustering | notebook 03, `clustering/cluster.py` |
| Justification de la semi-supervision | notebook 04 §A-B (avec vs sans labels partiels) |
| Métriques pertinentes (F1, ROC...) | `evaluation/metrics.py`, justification dans README |
| Hyperparamètres ajustés | notebook 04 §D, `configs/config.yaml` |
| Support de présentation (4 M images, 5 000 EUR) | `docs/presentation_outline.md`, `reports/presentation/` |
| Lien choix techniques ↔ contraintes métier | `scale_up/cost.py`, `reports/tables/scale_up_*.csv` |
| Arguments pour la soutenance | `docs/presentation_outline.md` (section « questions probables ») |
| Cohérence notebook ↔ support | chiffres du support = `reports/tables/*.csv` générés par les notebooks |
