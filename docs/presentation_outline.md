# Plan du support de présentation (≈ 15 diapositives)

1. Contexte & objectif (BrainScanAI, 1 400 non labellisées / 100 labellisées)
2. Jeu de données & qualité (résolution, doublons, outliers, biais de source)
3. Méthode globale (schéma du pipeline + règle anti-fuite)
4. Extraction de features (ResNet gelé, pourquoi, choix de couche)
5. Clustering : comparaison des 4 algorithmes, ARI/pureté, visualisation t-SNE
6. Labellisation faible : qualité des pseudo-labels, seuil de confiance
7. Approches semi-supervisées testées (Self-training, Label Spreading/Propagation, CNN faible→fort)
8. Métriques : pourquoi le **rappel** « cancer » prime (faux négatif = erreur la plus grave)
9. Résultats : supervisé vs semi-supervisé (moyenne ± écart-type, IC bootstrap)
10. Limites (100 labels, test ≈ 30 images, risque de décalage de domaine)
11. Passage à l'échelle : 0,00125 EUR/image disponibles vs 0,20 EUR/image du pilote
12. Architecture cible (pipeline GPU batch + file de revue experte)
13. Budget détaillé & sensibilité (tableau `scale_up_budget.csv`)
14. Conditions de faisabilité & risques (validation experte, dérive, réglementaire)
15. Recommandations & prochaines étapes

## Questions probables (soutenance)
- Pourquoi k = 2 ? Que se passe-t-il avec k = 3-5 ? (notebook 03, sensibilité)
- Comment avez-vous évité la fuite de données ? (rôles, mapping sur `strong_train`, assert)
- Pourquoi le rappel plutôt que l'accuracy ? Comment choisir le seuil ? (validation, pas test)
- Pourquoi n'est-il pas possible de payer un labelling par image à l'échelle ? (800 000 EUR extrapolés)
- Que valent les labels faibles ? Comment le mesurer sans radiologue ? (échantillon expert, IC)
- Le modèle est-il utilisable en clinique ? (non : R&D, marquage CE / MDR, validation prospective)
