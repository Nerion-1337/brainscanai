# Hyperparamètres du CNN : choix et justification

| Paramètre | Valeur | Rôle |
|---|---|---|
| `cnn.arch` | resnet18, pré-entraîné ImageNet | Petit réseau, adapté à 70 images ; réutilise des connaissances visuelles déjà apprises |
| `weak_stage` | 5 epochs, lr 1e-4 | Étape A : apprentissage sur les labels faibles (clusters). Court, car ces labels sont bruités |
| `strong_stage` | 15 epochs, lr 1e-4 | **Identique pour B (fine-tuning) et C (supervisé)** : seule différence entre B et C = ce que le modèle a vu avant. Comparaison équitable |
| `pseudo_stage` | 8 epochs, lr 1e-4 | Étape D : entraînement conjoint labels forts + pseudo-labels (jeu plus grand, donc moins d'epochs) |
| `pseudo_threshold` | 0,90 | D : un pseudo-label n'est retenu que si le modèle est sûr à ≥ 90 % |
| `pseudo_max_images` | 400 | D : plafond (temps CPU), réparti à parts égales entre les deux classes |
| `clustering.keep_fraction` | 0,5 | A/B : on garde les 50 % de labels faibles les plus sûrs de chaque classe (qualité > quantité) |
| `val_fraction` / `patience` | 0,2 / 4 | Arrêt anticipé : on arrête si la perte de validation ne s'améliore plus |

## Pourquoi ces choix (version courte)
- **Équité B/C** : avant correction, B était réentraîné avec un lr plus faible que C, ce qui le désavantageait. Les deux ont maintenant les mêmes réglages.
- **Deux sources de labels non étiquetés** : les clusters (A, B) sont bruités ; les pseudo-labels du modèle (D) sont en général plus propres car issus d'un classifieur entraîné sur de vrais labels.
- **Aucune fuite** : le test n'intervient ni dans les clusters→classes, ni dans le choix des pseudo-labels, ni dans l'arrêt anticipé.

## Diagnostics à regarder (`reports/tables/cnn_diagnostics.csv`)
- `weak_label_acc_on_test` : part de bons labels parmi ceux que donneraient les clusters (mesurée sur les images de test). Autour de 0,5 = labels faibles inutiles ; > 0,9 = bons.
- `n_pseudo_kept` : nombre de pseudo-labels retenus pour D. Proche de 0 = modèle peu sûr, abaisser `pseudo_threshold`.

## Ablation
`python scripts/run_ablation.py --keep 0.3 0.5 0.8 --repeats 3` compare qualité contre quantité des labels faibles sur les mêmes splits (`reports/tables/cnn_ablation.csv`). Dans Docker : `docker compose run --rm semi python scripts/run_ablation.py ...`.

## Lecture honnête des résultats
Avec 70 images fortes et des features ImageNet déjà performantes, le supervisé seul peut être très bon et laisser peu de marge au semi-supervisé. Un résultat « le semi-supervisé n'apporte rien ici » est valide s'il est expliqué par ces diagnostics.
