# Definition of done (objectif atteint)

## Technique
1. Pipeline reproductible de bout en bout (`docker compose run --rm pipeline`) à partir d'une config unique et d'une graine fixée.
2. Embeddings sauvegardés, alignés avec le manifeste, sans NaN/Inf.
3. ≥ 3 algorithmes de clustering comparés (ARI, NMI, silhouette, pureté) ; ARI discuté même s'il est modeste.
4. ≥ 3 approches semi-supervisées comparées au supervisé seul, **5 répétitions**, moyenne ± écart-type.
5. **Aucune fuite** : jeu de test jamais utilisé à l'entraînement ni pour le mapping clusters→classes ; labels faibles et forts jamais mélangés (garde-fou `assert_no_overlap`).
6. Tests (`pytest`) et lint (`ruff`) verts ; notebooks propres et commentés.

## Performance (à valider sur le vrai jeu)
| Critère | Seuil de succès |
|---|---|
| Rappel « cancer » (métrique principale : un faux négatif est le pire cas) | semi-supervisé ≥ supervisé seul (moyenne sur 5 répétitions) |
| AUC-ROC | ≥ 0,85 |
| F1 | ≥ supervisé seul |
| Incertitude | IC bootstrap 95 % rapporté (test ≈ 30 images) |

Si ces seuils ne sont pas atteints, le livrable reste valide **à condition de l'expliquer** : l'exploration est une étude de faisabilité, pas une promesse de performance. Un résultat négatif argumenté est un résultat.

## Business
- Recommandation chiffrée sur 4 M d'images / 5 000 EUR avec hypothèses explicites et analyse de sensibilité.
- Conditions de faisabilité et risques listés (décalage de domaine, validation experte, cadre réglementaire).
