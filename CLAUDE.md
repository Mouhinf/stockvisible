# StockVisible

## 5 fonctions, jamais plus
F1 DATA · F2 REVEAL (ventes+disponibilité) · F3 VERIFY (B0/B1/ML) ·
F4 DECIDE (panier sous budget) · F5 ACT (valider/exporter).

## Discipline scientifique — non négociable
DATA → TRAIN → VALIDATION → CHOIX DU MOTEUR → FREEZE → TEST FINAL UNE SEULE FOIS.
Le test final ne sert jamais à choisir un modèle/feature/seuil. Sélection sur
VALIDATION uniquement. Après freeze, plus aucune modification. Si le ML perd
sur validation, B1 devient le moteur — ce n'est pas un échec, c'est un résultat.

## Test à données masquées
C'est un test de RECONSTRUCTION/FIDÉLITÉ, jamais une preuve de récupération de
la vraie demande perdue pendant une rupture réelle. Coffre de vérité isolé,
seed fixée, anti-fuite testé explicitement.

## Stack figée
Python 3.12, Streamlit, pandas, NumPy, scikit-learn, Plotly, pytest, ruff,
pip-audit. Aucun ajout sans écrire d'abord une ARCHITECTURE CHANGE REQUEST
(raison/bénéfice/coût/risque/alternative) et attendre validation humaine.

## Interdits
Résultat hardcodé présenté comme réel. Donnée synthétique présentée comme
réelle sans étiquette. Secret dans le repo/logs. Suppression de test pour
faire passer la CI. Modification silencieuse du périmètre.

## Git
Un milestone = un commit `M{n}-{nom}`. Jamais plus d'un milestone cassé à la fois.

## Rapport de fin de tâche
Toujours terminer par :
MILESTONE: / STATUS: PASS|FAIL / FILES_CHANGED: / TESTS_PASSING: /
TESTS_FAILING: / KEY_RESULT: / RISKS: / READY_FOR_NEXT: YES|NO
