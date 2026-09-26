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

### M3 — travail dev-only et test masqué
- Développement : `splits.load_dev()` renvoie `DevSplits(train, validation)`, sans attribut test.
  Ne pas utiliser `chronological_split(load_raw("train"))` pendant le dev : il construit le
  test scellé en mémoire.
- Les tests qui lisent la période test sont marqués `touches_test_period`. Run dev-only :
  `pytest -m "not touches_test_period"`.
- Test masqué (`evaluation.build_masked_task`), spec figée `MaskSpec(250, 3, (6, 22), 42)` :
  - positions = heures DISPONIBLES de la fenêtre 6–22, à ≤ 3 h d'une rupture organique de la
    même journée ;
  - tirage stratifié par distance × côté (avant / après / entre ruptures) ;
  - jamais de journée sans rupture ;
  - positions choisies à partir du seul `hours_stock_status`.
- Les valeurs masquées vivent dans le `TruthVault` (copie en lecture seule, jamais transmise aux
  prédicteurs). `sale_amount` est mis à NaN sur les lignes masquées, sinon masqué = total −
  heures visibles.
- `check_no_leakage` (canaris) doit passer sur tout nouveau prédicteur avant de le comparer.

### M7 — ML horaire (décisions utilisateur, 2026-09-26, fixées avant tout entraînement)
- Granularité horaire, avec la variable `hour`. Une ligne = (série, jour D, heure).
- Variables (`features.FEATURES`), toutes strictement passées sauf le calendrier de D :
  - `hour`, `day_of_week`, `holiday_flag` du jour D (calendrier, connu d'avance) ;
  - `discount_d1` : remise de D-1, JAMAIS celle de D (hypothèse « promo planifiée » refusée,
    invérifiable) ;
  - `lag_d1`, `lag_d7`, `rolling_median_7` (médiane des 7 jours précédents, même heure, au moins
    3 observations) ; une vente à une heure en rupture est censurée → NaN ;
  - hiérarchie catégorielle : management_group, catégories 1 à 3, `product_id`, `city_id`.
- Cardinalité maximale de HistGradientBoosting : 255 (vérifié sur sklearn 1.9.1).
  - Sous-ensemble dev : product_id 227 (accepté), store_id 347 et séries 500 (refusés).
  - Dataset complet : 865 produits → product_id ne passerait plus.
- Modèle : `HistGradientBoostingRegressor(loss='quantile', quantile=0.5, early_stopping=False,
  random_state=42)`, tout le reste par défaut. Entraîné sur les heures DISPONIBLES du TRAIN seul.
  `TemporalLeakError` si le modèle a vu des dates >= la première date cible.
- Évaluation glissante à J+1 (`evaluation.rolling_origin`) : B0/B1 recalculés avec l'historique
  jusqu'à D-1, et le ML aussi, sans jamais être réentraîné.
- Règle de sélection (`evaluation.SELECTION_RULE`) : ML retenu seulement si MAE(ML) < MAE(B1),
  en glissant, heures disponibles, mêmes cellules ; sinon B1 reste le moteur.
- Anti-fuite : `features.check_features_past_only` (perturbation des issues des jours >= C) doit
  passer sur toute nouvelle variable. 4 contrôles négatifs en xfail strict.
- RÉSULTAT M7 (validation, glissant, 138 210 heures) : ML MAE 0,0533 vs B1 0,0482 → B1 reste le
  moteur. MAIS le ML est DÉGÉNÉRÉ : il prédit exactement 0 partout (in-sample aussi), soit
  2 feuilles par arbre et des valeurs de feuille nulles sur les 100 arbres.
  - Cause diagnostiquée sur le train et sur un jouet : HGB quantile/absolute_error de sklearn
    1.9.1 ne sort pas de sa valeur initiale quand une majorité de cibles lui est exactement
    égale (71 % de ventes horaires nulles, médiane initiale 0).
  - Un bruit de 1e-6 sur la cible débloque les arbres.
  - Des groupes à médiane > 0 existent pourtant (rolling_median_7 > 0 : 56 % d'heures avec
    vente, médiane 0,10).
  - Toute correction = changement de méthode après un résultat de validation : décision
    humaine requise.
- M7b (décision utilisateur, option 2) : `ModelSpec.target_jitter = 1e-6`, bruit uniforme à seed
  fixe sur la cible d'ENTRAÎNEMENT uniquement.
  - Test de non-régression : sans le bruit, HGB reste à 0 ; avec, il apprend une médiane
    positive.
  - UNE seule relance déclarée sur la validation (second regard, sans aucun autre
    changement) : ML 0,0456 < B1 0,0482 → la règle retient ML. Biais ML −0,031 contre B1
    −0,020 : le ML sous-estime davantage.
  - Test masqué : ML 0,0953 contre B1 0,0962 (quasi égalité).
  - Non figé, test non ouvert.
  - Machine de dev à 2 cœurs chargée : lancer avec `OMP_NUM_THREADS=1` (entraînement 5×
    plus rapide).

### M8 — moteur GELÉ (2026-09-26)
- Choix sur la validation seule (`selection.run_selection`) : **ML** retenu.
  - MAE ML 0,0456 < B1 0,0482 (B0 0,0475), sur 138 210 h glissantes.
  - Biais : ML −0,031, B1 −0,020, B0 −0,029.
  - Identique à M7b : recalcul déterministe.
- `engine_freeze.json` (racine du dépôt) contient :
  - la décision, la règle et les métriques ;
  - les specs et la liste des variables ;
  - le SHA-256 de train.parquet ;
  - les SHA-256 de `model.py`, `features.py` et `baselines.py`.
- NE PLUS MODIFIER model.py, features.py, baselines.py : `check_freeze` fait échouer
  `tests/test_selection.py` à la moindre modification.
- `select_engine(table, test_start=...)` exige une provenance (split=validation, dt_max <
  test_start) ; `splits.test_start(dev)` est déduit du dev seul.
- AVANT d'ouvrir le test (M9), décider et écrire :
  - l'entraînement final (train seul ou train + validation, même configuration) ;
  - le protocole du test (glissant J+1, mêmes métriques).
  Puis ouvrir le test une seule fois (`UNSEAL_PHRASE`).

### M9 — TEST FINAL CONSOMMÉ (2026-09-26 22:23:00 UTC) — ne plus jamais le relancer
- Exécution unique (`stockvisible/final_test.py`) : modèle gelé M8, entraîné sur le train seul ;
  prévision glissante J+1 sur 2024-06-11 → 2024-06-25 ; 500 séries ; 134 744 heures communes.
- Résultat brut (`logs/final_test_result.json`, mode d'écriture exclusif) :
  - ML : MAE 0,049152, biais −0,035528 ;
  - B1 : MAE 0,050774, biais −0,026657 ;
  - B0 : MAE 0,050799, biais −0,035783.
- Validation M8 pour rappel : ML 0,0456, B1 0,0482, B0 0,0475. Toutes les erreurs sont plus
  élevées sur le test. L'écart ML/B1 passe de −5,4 % à −3,2 %, et le ML sous-estime davantage.
- Le test réservé est désormais « vu » : toute nouvelle évaluation qui l'utiliserait pour décider
  serait biaisée. Le fichier `eval` officiel (7 jours) reste intact : c'est le seul jeu vierge.

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
