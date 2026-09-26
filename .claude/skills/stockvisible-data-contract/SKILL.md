---
name: stockvisible-data-contract
description: Consulter avant toute modification de data.py, validation.py,
splits.py, baselines.py, features.py, model.py, evaluation.py.
---
Schéma FreshRetailNet-50K : city_id, store_id, management_group_id,
first/second/third_category_id, product_id, dt, sale_amount (normalisé),
hours_sale (Sequence[float], normalisé), stock_hour6_22_cnt (int),
hours_stock_status (Sequence[int], 1=rupture), discount, holiday_flag,
activity_flag, precpt, avg_temperature, avg_humidity, avg_wind_level.
~20% de ruptures organiques réelles déjà annotées, ne pas les recréer
artificiellement. Une vente nulle N'EST PAS une preuve de rupture — seul
hours_stock_status fait foi. Ne jamais renommer les valeurs normalisées en
FCFA. Vérifier si le split train/eval officiel du dataset correspond à des
séries différentes ou à un découpage temporel des mêmes séries AVANT de
superposer un split maison.

## Split officiel — VÉRIFIÉ (2026-09-26, M1)

Source : `Dingdong-Inc/FreshRetailNet-50K`, révision figée
`08c1fab7f9257bc73679d415d65d644165d351d4`, fichiers complets (SHA-256 vérifiés,
voir `stockvisible/data.py::SOURCE_SHA256`).

| | train.parquet | eval.parquet |
|---|---|---|
| lignes | 4 500 000 | 350 000 |
| séries (store_id, product_id) | 50 000 | 50 000 |
| jours par série | 90 (toutes) | 7 (toutes) |
| dt | 2024-03-28 → 2024-06-25 | 2024-06-26 → 2024-07-02 |

- Séries communes : 50 000 / 50 000. Séries propres à un seul fichier : 0.
- Dates communes : 0. Doublons (série, dt) : 0 dans chaque fichier.
- Chaque série a une seule valeur de city_id et des catégories 1 à 3 / management_group_id.

**Conclusion : le split officiel est un DÉCOUPAGE TEMPOREL DES MÊMES SÉRIES**
(horizon de 7 jours juste après train), et non un split par séries différentes.

Implications pour splits.py et la suite :
- `eval` officiel = TEST FINAL. On n'y touche qu'une seule fois, après le freeze.
- La VALIDATION doit être découpée dans `train`, temporellement (les derniers jours
  de train), sur les mêmes séries — c'est l'analogue fidèle du split officiel.
  Pas de split aléatoire de lignes (fuite temporelle).
- Toute feature ou statistique par série doit être calculée uniquement sur le passé
  de la fenêtre évaluée.

### Décision M2 (utilisateur, 2026-09-26) — remplace le premier point ci-dessus
- `splits.chronological_split` : pour chaque série, les 90 jours de train sont découpés en
  60 jours train (2024-03-28 → 05-26), 15 jours validation (05-27 → 06-10) et 15 jours test
  (06-11 → 06-25). Ce test est rendu scellé (`SealedFrame`) et ne s'ouvre qu'avec
  `UNSEAL_PHRASE`, après le freeze.
- Le fichier `eval` officiel (7 jours) n'est PAS utilisé. Il reste intouché, et son rôle
  (second test externe ?) est à décider par l'humain.
- Protocole d'évaluation fixé avant tout résultat (`evaluation.py`) :
  - périmètre principal = heures déclarées disponibles dans la cible ;
  - modèles comparés sur les mêmes cellules ;
  - biais = moyenne(préd − réel) ;
  - B1 : `min_obs=3`, fixé a priori, jamais ajusté.

Recalcul : `stockvisible.data.inspect_official_split()` (lit uniquement les colonnes
clés et dt). Le résultat est aussi écrit dans `data/raw/manifest.json` par
`python -m stockvisible.data`.

## Invariants mesurés sur les données réelles (M1)

Mesurés sur 300 000 lignes de train (1 row group sur 15) + eval complet.
Ce sont eux que `validation.validate()` applique comme règles bloquantes.

- `hours_sale` et `hours_stock_status` : longueur 24 ; statut ∈ {0, 1} ; aucun NaN nulle part.
- `sum(hours_sale) == sale_amount` (écart max observé 7e-15).
- `stock_hour6_22_cnt == sum(hours_stock_status[6:22])` (heures 6 à 21 incluses) sur 100 % des
  lignes. Les fenêtres [6:23], [7:23] et [6:21] échouent (~60 % seulement).
- `stock_hour6_22_cnt` ∈ [0, 16] ; `discount` ∈ [0, 1] (0.0 existe) ; drapeaux ∈ {0, 1}.

Faits réels qui NE sont PAS des incohérences (ne jamais les rejeter) :
- Vente > 0 pendant une heure marquée rupture : ~0,7 % des heures, ~13 % des lignes
  (rupture survenue en cours d'heure). `validate()` le signale en `info` seulement.
- `sale_amount == 0` sans aucune heure de rupture : ~0,8 % des lignes. Une vente nulle
  n'est pas une rupture.
- Heures de nuit (0–5 h, 22–23 h) : statut = 1 dans ~35–42 % des cas, contre ~5 % à 7 h.
  Hypothèse non vérifiée : fermeture ou absence de réassort la nuit, d'où un compteur
  officiel limité à la fenêtre 6–22. Ne pas compter ces heures comme de la demande perdue
  sans l'avoir établi.

## Sous-ensemble local (data/raw/)

Critères fixés AVANT de regarder les résultats, indépendants des ventes et des ruptures :
toutes les clés (store_id, product_id) de train, triées, puis tirage uniforme sans remise
de `n_series=500` clés avec `numpy.random.default_rng(42)`. Toutes les dates de chaque
série retenue sont gardées, dans train ET dans eval. Voir `SubsetSpec` dans data.py.
Les données sont RÉELLES (pas synthétiques) et sous licence CC BY 4.0. Les valeurs de vente
restent normalisées : aucune unité monétaire.

Résultat (`data/raw/manifest.json`) : train = 45 000 lignes (500 × 90), eval = 3 500 lignes
(500 × 7). La part d'heures en rupture sur la fenêtre 6–22 vaut 20,2 % sur train et 18,7 % sur
eval, ce qui correspond aux « ~20 % » annoncés par le dataset. Ce sont des ruptures organiques
déjà présentes : ne jamais en ajouter.
