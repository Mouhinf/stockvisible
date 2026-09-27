# StockVisible

**Voir la demande que les ruptures cachent, puis décider quoi racheter sous budget — avec des
prévisions dont la fiabilité est prouvée.**

Application : https://stockvisible.galsentechnologie.com (secours : https://stockvisible.onrender.com)

## Le problème

Quand un produit est en rupture, la caisse enregistre zéro vente. Ce zéro n'est pas une absence
de clients : les ventes observées **sous-estiment la demande**, et un gérant qui recommande sur
cette base recommande trop peu, ce qui entretient les ruptures. Dans les données utilisées
(500 produits réels de magasins de produits frais), environ **20 % des heures de 6 h à 22 h
sont en rupture**.

## Ce que fait l'application (3 écrans)

| Écran | Ce qu'on y voit |
|---|---|
| **Vérifier** (accueil) | Le problème en une phrase, puis la preuve : erreurs du moteur ML et des deux références sur la validation et sur le test final, incertitude, garanties, limites |
| **Comprendre** | Pour un produit : ventes heure par heure, disponibilité, ventes observées vs demande estimée par jour, prévisions B0 / B1, abstention ; import d'un fichier contrôlé par le contrat de données |
| **Acheter** | Coûts, lots et stocks saisis comme hypothèses ; budget ; panier optimal (demande couverte, manque) ; validation puis export CSV / JSON avec provenance |

## Résultats (lus dans `engine_freeze.json` et `logs/final_test_result.json`)

Prévisions horaires glissantes à J+1, heures où le produit était disponible, mêmes heures pour
les trois modèles. Valeurs normalisées, sans unité.

| Modèle | MAE validation | MAE test final | Biais test final |
|---|---|---|---|
| **ML (moteur gelé)** | 0,0456 | **0,0492** | −0,0355 |
| B1 (médiane, heures disponibles) | 0,0482 | 0,0508 | −0,0267 |
| B0 (médiane, ruptures comprises) | 0,0475 | 0,0508 | −0,0358 |

- Sur le test final, ouvert **une seule fois**, le ML a une erreur 3,2 % plus basse que B1
  (5,4 % en validation).
- Il **sous-prévoit** (biais négatif, proche de B0) : c'est affiché à l'écran, et c'est pourquoi
  le panier ne l'utilise pas.
- Intervalle 80 % autour du moteur : couverture mesurée 79,5 % sur des jours de validation
  jamais vus en calibration.

## Rôle de l'IA — et ses limites

- Le ML (scikit-learn, gradient boosting quantile) prévoit la vente de chaque heure du
  lendemain. Il a été **mis en concurrence** avec deux références simples, **choisi sur la
  validation seule** selon une règle écrite avant tout résultat, **gelé** (empreintes SHA-256
  vérifiées à chaque affichage), puis évalué **une fois** sur une période réservée.
- Le panier de l'écran Acheter utilise la demande estimée avec **B1** (ventes des heures
  disponibles + B1 aux heures en rupture), pas le ML. C'est dit à l'écran et dans les exports.
- Détail complet, y compris les outils IA de développement : [AI_USAGE.md](AI_USAGE.md).
  NVIDIA Brev : non utilisé.

## Discipline scientifique

DATA → TRAIN (60 j) → VALIDATION (15 j) → CHOIX DU MOTEUR → GEL → TEST FINAL UNE SEULE FOIS (15 j).
Découpage chronologique par série, jamais aléatoire. Anti-fuite testé : canaris, perturbation
des jours futurs, et 9 contrôles négatifs (fuites injectées volontairement, que la suite doit
détecter). Règles du projet : [CLAUDE.md](CLAUDE.md).

## Limites

- Données réelles **chinoises** (Dingdong), normalisées : aucune unité monétaire, aucun FCFA,
  aucune donnée du Sénégal.
- Coûts, tailles de lot et stocks sont des **hypothèses saisies** : le jeu de données n'en
  contient pas.
- La demande perdue pendant une rupture est **estimée**, jamais observée. Le test à données
  masquées mesure la fidélité de la reconstruction, pas la demande perdue réelle.
- Gain du ML modeste (3,2 % sur le test) et biais négatif.

## Prochaine étape

Pilote avec un point de vente équipé d'une caisse et d'un suivi de stock, avec ses coûts réels ;
puis un moteur orienté achat (quantile supérieur à la médiane) choisi sur la validation et évalué
sur la période « eval » officielle du jeu de données, jamais utilisée pour évaluer un modèle.

## Tests et fiabilité

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/ruff check .
.venv/bin/pytest -q                                  # suite complète
.venv/bin/pytest -q -m "not touches_test_period"     # sans lire la période test réservée
npm ci && npx playwright test                        # parcours E2E, ordinateur + mobile
.venv/bin/streamlit run app.py                       # http://localhost:8501
```

CI GitHub Actions : ruff, pytest, pip-audit. Sécurité : [SECURITY.md](SECURITY.md).
État et historique : [STATUS.md](STATUS.md).

## Données

Sous-ensemble déterministe (500 séries, seed 42) de **FreshRetailNet-50K**, Dingdong-Inc,
licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), révision
`08c1fab7f9257bc73679d415d65d644165d351d4`
([arXiv:2505.16319](https://arxiv.org/abs/2505.16319)). Les ventes sont normalisées par le
fournisseur : aucune unité monétaire. Le fichier versionné est `data/raw/train.parquet`
(+ `manifest.json`). Il a été filtré sur les séries retenues ; aucune valeur n'a été modifiée.

Reconstruire les données depuis la source : `python -m stockvisible.data`.

## Déploiement Render

Service en ligne : https://stockvisible.onrender.com (région Frankfurt), aussi servi sur
https://stockvisible.galsentechnologie.com (CNAME `stockvisible` → `stockvisible.onrender.com`,
domaine déclaré dans Render → Settings → Custom Domains). Le fichier [`render.yaml`](render.yaml) décrit le service (Blueprint). Pour une configuration
à la main dans le dashboard, reporter les mêmes valeurs :

| Champ | Valeur |
|---|---|
| Runtime | Python 3 |
| Branch | `master` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| Health Check Path | `/_stcore/health` |
| Variable d'environnement | `PYTHON_VERSION` = `3.12.3` |
| Instance | Free (512 Mo, mise en veille après 15 min d'inactivité : ouvrir l'URL quelques minutes avant une démo) |

Aucun secret n'est nécessaire.

## Chronologie

L'historique Git fait foi : M0 à M12 (données, modèle, gel, test final, écrans, déploiement,
tests E2E) ont été commités du 26/09/2026 16:03 UTC au 27/09/2026 05:48 UTC ; M13 et la suite
(audit de sécurité, export, CI, finitions) le 27/09/2026.
