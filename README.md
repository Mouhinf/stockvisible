# StockVisible

*English version: [README.en.md](README.en.md).*

**Voir la demande que les ruptures cachent, puis décider quoi racheter sous budget — avec des
prévisions dont la fiabilité est prouvée.**

Application : https://stockvisible.galsentechnologie.com (secours : https://stockvisible.onrender.com)

## Le problème

Quand un produit est en rupture, la caisse enregistre zéro vente. Ce zéro n'est pas une absence
de clients : les ventes observées **sous-estiment la demande**, et un gérant qui recommande sur
cette base recommande trop peu, ce qui entretient les ruptures. Dans les données utilisées
(500 séries magasin × produit réelles de produits frais), **20,2 % des heures de 6 h à 22 h sont
en rupture** sur la période d'entraînement (`data/raw/manifest.json`).

## Utilisateur

Le **gérant ou responsable des achats d'un point de vente de produits frais**, qui doit décider
chaque jour quoi racheter avec un budget limité. Il n'a pas besoin de lire le code : l'application
lui montre d'abord la preuve de fiabilité (écran Vérifier), puis la demande d'un produit
(Comprendre), puis un panier sous budget qu'il valide et exporte (Acheter).

Aucun entretien ni pilote avec un gérant n'a encore eu lieu : ce besoin est une hypothèse de
travail, pas un besoin validé sur le terrain.

## Ce que fait l'application (3 écrans)

| Écran | Ce qu'on y voit |
|---|---|
| **Vérifier** (accueil) | Le problème en une phrase, puis la preuve : erreurs du moteur ML et des deux références sur la validation et sur le test final, incertitude, garanties, limites |
| **Comprendre** | Pour un produit : ventes heure par heure, disponibilité, ventes observées vs demande estimée par jour, prévisions B0 / B1, abstention ; import d'un fichier contrôlé par le contrat de données |
| **Acheter** | Coûts, lots et stocks saisis comme hypothèses ; budget ; panier optimal (demande couverte, manque) ; validation puis export CSV / JSON avec provenance |

## Architecture

Cinq fonctions dans `stockvisible/`, l'interface dans `ui/`.

| Fonction | Modules | Rôle |
|---|---|---|
| F1 DATA | `data.py`, `validation.py` | Sous-ensemble déterministe et vérifié (SHA-256) du jeu de données ; contrat de données ; lecture sûre des fichiers importés (CSV / Parquet, plafonds de taille et de lignes, aucune exécution de contenu) |
| F2 REVEAL | `ui/pages/comprendre.py`, `ui/charts.py` | Ventes horaires, disponibilité déclarée, demande estimée |
| F3 VERIFY | `splits.py`, `baselines.py`, `features.py`, `model.py`, `evaluation.py`, `selection.py`, `final_test.py`, `uncertainty.py` | Découpage chronologique, références B0 / B1, variables strictement passées, modèle ML, sélection sur validation, gel, test final unique, intervalle empirique |
| F4 DECIDE | `allocation.py` | Panier optimal sous budget, énumération exhaustive bornée (≤ 3 produits), lots entiers |
| F5 ACT | `exports.py` | Validation du panier, export CSV / JSON strictement identique à l'écran |

- `app.py` : routeur des trois écrans (`st.navigation`). Aucun modèle n'est entraîné à
  l'exécution : l'écran Vérifier lit `engine_freeze.json`, `logs/final_test_result.json` et
  `logs/interval_calibration.json`.
- Interface : Streamlit 1.64, graphiques Plotly, design system dans `design-system/MASTER.md`.
- Contrat de données et journal des décisions : [docs/data-contract.md](docs/data-contract.md).

## Résultats (lus dans `engine_freeze.json` et `logs/final_test_result.json`)

Prévisions horaires glissantes à J+1, heures où le produit était disponible, mêmes heures pour
les trois modèles. Valeurs normalisées, sans unité.

| Modèle | MAE validation | MAE test final | Biais test final |
|---|---|---|---|
| **ML (moteur gelé)** | 0,0456 | **0,0492** | −0,0355 |
| B1 (médiane, heures disponibles) | 0,0482 | 0,0508 | −0,0267 |
| B0 (médiane, ruptures comprises) | 0,0475 | 0,0508 | −0,0358 |

- Sur le test final, ouvert **une seule fois** (2026-09-26 22:23 UTC, 134 744 heures communes),
  le ML a une erreur 3,2 % plus basse que B1 (5,4 % en validation).
- Il **sous-prévoit** (biais négatif, proche de B0) : c'est affiché à l'écran, et c'est pourquoi
  le panier ne l'utilise pas.

## Discipline scientifique

DATA → TRAIN (60 j) → VALIDATION (15 j) → CHOIX DU MOTEUR → GEL → TEST FINAL UNE SEULE FOIS (15 j).

- Découpage chronologique par série, jamais aléatoire.
- Règle de choix écrite avant tout résultat : le ML n'est retenu que s'il bat B1 sur la
  validation, sinon B1 devient le moteur.
- Anti-fuite testé : canaris, perturbation des jours futurs, et 9 contrôles négatifs (fuites
  injectées volontairement, que la suite doit détecter).
- Deux regards sur la validation sont déclarés : le premier modèle était dégénéré (il prédisait 0
  partout) ; il a été corrigé une fois, puis la configuration a été gelée.

## IA responsable

- **Pas de LLM** ni d'API d'IA externe dans l'application : le moteur est un modèle tabulaire
  (scikit-learn) mis en concurrence avec deux règles simples.
- **Abstention visible** : quand aucune période comparable n'existe, B1 ne prévoit rien et
  l'écran le dit (totale ou partielle) ; les heures restent vides, jamais remplies par défaut.
- **Incertitude mesurée** : intervalle 80 % autour du moteur, couverture empirique 79,5 % sur des
  jours de validation jamais vus en calibration ; « intervalle non calibré » affiché quand le
  support est insuffisant. La nuit, l'intervalle vaut [0, 0] : calibré mais peu informatif.
- **Étiquetage** : données réelles signalées comme telles ; coûts et stocks signalés comme
  hypothèses ; fichiers de test synthétiques nommés `*_SYNTHETIQUE.csv` ; demande perdue
  toujours présentée comme **estimée**.
- **Limites affichées à l'écran** (écran Vérifier), pas seulement dans ce document.
- Outils d'IA utilisés pour développer le projet : [AI_USAGE.md](AI_USAGE.md).

## Données

Sous-ensemble déterministe (500 séries, seed 42) de **FreshRetailNet-50K**, Dingdong-Inc,
licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), révision
`08c1fab7f9257bc73679d415d65d644165d351d4`
([arXiv:2505.16319](https://arxiv.org/abs/2505.16319)). Les ventes sont normalisées par le
fournisseur : aucune unité monétaire. Le fichier versionné est `data/raw/train.parquet`
(+ `manifest.json`), filtré sur les séries retenues, sans aucune valeur modifiée. Le fichier
« eval » officiel du jeu de données n'est pas versionné et n'a jamais servi à évaluer un modèle.

Reconstruire les données depuis la source : `python -m stockvisible.data`.

## Limites

- Données réelles **chinoises** (Dingdong), normalisées : aucune unité monétaire, aucun FCFA,
  aucune donnée du Sénégal.
- Coûts, tailles de lot et stocks sont des **hypothèses saisies** : le jeu de données n'en
  contient pas.
- La demande perdue pendant une rupture est **estimée**, jamais observée. Le test à données
  masquées mesure la fidélité de la reconstruction, pas la demande perdue réelle.
- Gain du ML modeste (3,2 % sur le test) et biais négatif ; le panier utilise B1.
- Les lignes de la période test ont été lues avant le découpage (contrôle qualité des données) et
  sont publiques dans le dépôt ; elles n'ont servi à aucune décision de modèle.
- Offre gratuite Render : démarrage à froid observé d'environ 2 minutes après 15 minutes
  d'inactivité.

## Installation

Python 3.12 requis.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py                       # http://localhost:8501
```

## Tests

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q                                  # suite complète (519 tests collectés)
.venv/bin/pytest -q -m "not touches_test_period"     # sans lire la période test réservée
npm ci && npx playwright test                        # parcours E2E, ordinateur + mobile
E2E_BASE_URL=https://stockvisible.onrender.com npx playwright test   # même parcours en production
```

Sur un clone neuf, 2 tests sont sautés car ils demandent des fichiers de données non versionnés
(eval officiel, cache de téléchargement), avec une raison explicite.
CI GitHub Actions : ruff, pytest, pip-audit (informatif). Sécurité : [SECURITY.md](SECURITY.md).
État et historique : [STATUS.md](STATUS.md). Preuves par critère du jury :
[docs/judging-evidence.md](docs/judging-evidence.md).

## Déploiement

Render, service Web, région Frankfurt, redéployé à chaque push sur `master`. Le fichier
[`render.yaml`](render.yaml) décrit le service. Domaine personnalisé : CNAME `stockvisible` →
`stockvisible.onrender.com`, déclaré dans Render → Settings → Custom Domains.

| Champ | Valeur |
|---|---|
| Runtime | Python 3 |
| Branch | `master` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| Health Check Path | `/_stcore/health` |
| Variable d'environnement | `PYTHON_VERSION` = `3.12.3` |
| Instance | Free (512 Mo, mise en veille après 15 min d'inactivité) |

Aucun secret n'est nécessaire.

## Licences

- **Données** : FreshRetailNet-50K, Dingdong-Inc, CC BY 4.0 (attribution ci-dessus).
- **Dépendances** : bibliothèques open source listées dans `requirements.txt` (pandas, NumPy,
  scikit-learn, Streamlit, Plotly, pyarrow, pytest, ruff, pip-audit) et `package.json`
  (@playwright/test, outil de test uniquement), chacune sous sa propre licence.
- **Code de StockVisible** : aucun fichier de licence n'a encore été ajouté au dépôt ; sans
  licence explicite, les droits restent réservés à l'auteur. Choisir une licence est une
  prochaine étape.
- **Logo** de l'onglet : icône de Galsen Technologie (galsentechnologie.com).

## Prochaine étape

Pilote avec un point de vente équipé d'une caisse et d'un suivi de stock, avec ses coûts réels ;
puis un moteur orienté achat (quantile supérieur à la médiane) choisi sur la validation et évalué
sur la période « eval » officielle du jeu de données, jamais utilisée pour évaluer un modèle.

## Chronologie

L'historique Git fait foi. Premier commit le 26/09/2026 à 16:03 UTC ; données, modèle, gel, test
final et écrans (M0 à M11) commités le 26/09/2026 et dans la nuit jusqu'au 27/09 à 00:56 UTC ;
tests E2E, audit de sécurité, export, CI, production et finitions le 27/09/2026.
