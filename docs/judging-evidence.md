# Preuves par critère du jury — StockVisible

Chaque ligne relie un critère officiel du hackathon (poids de la grille) à la fonction concernée,
à une preuve vérifiable dans le dépôt ou en production, au test qui la garantit et à l'écran où la
voir. Les chemins de tests sont relatifs à `tests/`. Lancer toute la suite : `pytest -q` ;
le parcours complet : `npx playwright test`.

Application : https://stockvisible.galsentechnologie.com (secours : https://stockvisible.onrender.com).

| Critère (poids) | Fonction | Preuve | Test | Écran |
|---|---|---|---|---|
| **Problème et valeur (20)** | F2 REVEAL | 20,2 % des heures de 6 h à 22 h en rupture sur la période d'entraînement (`data/raw/manifest.json`, champ `stockout_hour_share_6_22`) ; bandeau du problème calculé sur les données, pas saisi | `test_ui.py::test_problem_banner_numbers_are_computed_not_typed` | Vérifier (bandeau) |
| | F2 REVEAL | Ventes observées et demande estimée affichées côte à côte, par jour, pour un produit | `test_ui.py::test_observed_vs_estimated_matches_the_basket_demand` | Comprendre |
| **Fonctionnement (20)** | F1 → F5 | Parcours complet import → visualisation → preuve → budget → panier → validation → export, en local et en production, ordinateur et mobile, zéro erreur console | `e2e/parcours.spec.ts` (Playwright, 2 profils) ; rejeu en production : `E2E_BASE_URL=https://stockvisible.onrender.com npx playwright test` | les trois |
| | F4 DECIDE | Le curseur de budget recalcule réellement le panier ; l'optimum est recoupé par une énumération naïve indépendante sur 200 instances aléatoires | `test_ui.py::test_budget_slider_really_recomputes_the_basket` ; `test_allocation.py::test_matches_naive_enumeration` | Acheter |
| | F5 ACT | Export CSV et JSON strictement identique à ce qui est affiché (indicateurs, panier, hypothèses, horodatage, empreinte) | `test_exports.py::test_export_is_strictly_what_the_screen_shows` | Acheter |
| **Usage de l'IA (20)** | F3 VERIFY | Modèle ML mis en concurrence avec B0 et B1, règle de choix écrite avant tout résultat, choix sur la validation seule, configuration gelée (`engine_freeze.json`, empreintes SHA-256 de `model.py`, `features.py`, `baselines.py`) | `test_selection.py::test_select_engine_never_receives_test_data` ; `test_selection.py::test_frozen_code_and_specs_are_unchanged` | Vérifier (tableau) |
| | F3 VERIFY | Test final exécuté une seule fois : `logs/final_test_result.json` (`ouvertures_du_test` = 1, horodatage 2026-09-26 22:23:00 UTC, écriture en mode exclusif) ; ML 0,0492 contre B1 0,0508 (MAE) | `test_final_test.py::test_result_is_a_single_opening_of_the_frozen_engine` ; `test_final_test.py::test_nothing_changed_after_the_final_test` | Vérifier |
| | F3 VERIFY | Variables strictement passées, prévision glissante sans jamais voir le jour cible | `test_features.py::test_real_dev_features_are_strictly_past` ; `test_model.py::test_rolling_ml_uses_only_days_before_each_target` | — |
| **Tests et fiabilité (15)** | F3 VERIFY | Découpage chronologique reproductible ; anti-fuite par canaris et perturbation ; 9 contrôles négatifs (fuites injectées que la suite doit détecter, en échec attendu strict) | `test_splits.py::test_split_is_reproducible_whatever_the_row_order_seed` ; `test_anti_leakage.py::test_anti_leak_fails_on_injected_leak` ; `test_model.py::test_real_ml_passes_canary_anti_leak` | — |
| | F3 VERIFY | Répétition générale du test final sur la validation : retrouve exactement les métriques gelées | `test_final_test.py::test_dress_rehearsal_on_validation_reproduces_frozen_metrics` | — |
| | F1 DATA | Contrat de données : 100 % des corruptions synthétiques détectées ; fichiers importés plafonnés avant décodage, cellules lues en JSON, jamais évaluées | `test_validation.py::test_detection_rate_is_100_percent` ; `test_validation.py::test_parquet_row_count_is_capped_from_metadata_before_decoding` ; `test_validation.py::test_csv_list_cells_are_parsed_as_json_never_evaluated` | Comprendre (import) |
| | Tout | CI GitHub Actions (ruff, pytest, pip-audit informatif) ; audit de sécurité en lecture seule (`SECURITY.md`) | `.github/workflows/ci.yml` | — |
| **Expérience et démo (15)** | Interface | Trois écrans dans l'ordre Vérifier → Comprendre → Acheter ; au plus trois indicateurs par écran ; contrastes WCAG AA calculés ; design system `design-system/MASTER.md` | `test_design.py::test_at_most_three_metrics_per_screen` ; `test_design.py::test_every_text_color_meets_wcag_aa` | les trois |
| | Production | Démarrage à froid d'environ 2 minutes observé dans les logs Render après 15 minutes d'inactivité (voir `STATUS.md`) : ouvrir l'URL avant la démo | — | — |
| **IA responsable et données (10)** | F3 VERIFY | Abstention visible quand aucune période comparable n'existe (heures laissées vides, jamais remplies) | `test_uncertainty.py::test_synthetic_series_without_enough_history_triggers_abstention` | Comprendre |
| | F3 VERIFY | Intervalle 80 % : couverture empirique 79,5 % sur des jours de validation jamais vus en calibration (`logs/interval_calibration.json`) ; « intervalle non calibré » si le support est insuffisant | `test_uncertainty.py::test_real_report_uses_validation_only_and_is_consistent` | Vérifier |
| | F1 DATA | Données réelles FreshRetailNet-50K (CC BY 4.0, révision figée, SHA-256 vérifiés) ; hypothèses et fichiers synthétiques étiquetés ; limites affichées à l'écran | `test_data_contract.py` (manifeste, source figée) | Vérifier (limites, provenance) |

## Ce que ce tableau ne prouve pas

- Aucun besoin validé sur le terrain : aucun entretien ni pilote avec un gérant.
- Données chinoises normalisées : aucune unité monétaire, aucune donnée du Sénégal ; coûts et
  stocks sont des hypothèses saisies.
- La demande perdue pendant une rupture est estimée, jamais observée.
- Le gain du ML est modeste (3,2 % sur le test) et son biais est négatif ; le panier utilise B1.
- Aucune vidéo de démonstration n'est versionnée dans ce dépôt.
