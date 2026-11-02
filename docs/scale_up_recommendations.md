# Passage à l'échelle : 4 M d'images, 5 000 EUR

**Constat** : budget disponible = 5 000 / 4 000 000 = **0,00125 EUR/image**, contre 300 / 1 500 = **0,20 EUR/image** pour le pilote (extrapolation naïve : 800 000 EUR). Un étiquetage payé à l'image est donc exclu.

**Faisable sous conditions** (chiffres détaillés : `python scripts/estimate_scale_up.py`, hypothèses modifiables dans `configs/config.yaml`) :
1. Labellisation 100 % automatique, **auto-hébergée** (embeddings + classifieur/pseudo-labels), coût technique de l'ordre de quelques dizaines d'euros sous les hypothèses par défaut.
2. Le budget restant sert à la **revue experte ciblée** : échantillon de validation (≈ 146 images pour estimer un rappel de 95 % à ± 5 %) + cas à faible confiance (apprentissage actif).
3. Les performances mesurées sur le pilote (100 labels) doivent être confirmées sur un échantillon plus grand avant extrapolation.
4. Surveillance du **décalage de domaine** (nouveaux scanners/hôpitaux) : contrôle des distributions d'embeddings.
5. Statut : labels **faibles pour la R&D**, pas de valeur diagnostique ; usage clinique = cadre réglementaire dispositif médical.

**Risques** : débit réel inférieur aux hypothèses (voir sensibilité), tarif des radiologues, coût de transfert des données, qualité des pseudo-labels plus faible qu'au pilote.
