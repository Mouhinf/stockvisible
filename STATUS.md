# Statut

| | |
|---|---|
| Dernier milestone | M11-ui-3-screens (M7 → M11 non encore poussés, donc pas encore déployés) |
| URL publique | https://stockvisible.onrender.com |
| Vérification HTTPS | 2026-09-26 18:4x GMT : `curl -sf` → 200 ; `/_stcore/health` → `ok` ; http → 301 vers https ; certificat valide (expire le 2026-12-20) |
| Vérification navigateur | session complète sur l'URL publique : graphique, 4 tableaux, changement de série ; 0 erreur page / console / logs |
| Hébergeur | Render, service Web `srv-das10h3bc2fs738t57ag`, offre gratuite, région Frankfurt |
| Déploiement | automatique à chaque push sur `master` (github.com/Mouhinf/stockvisible), une fois GitHub connecté à Render (2026-09-26). Le push de d3307c3, fait avant la connexion, n'avait déclenché aucun déploiement. |
| Données servies | sous-ensemble RÉEL dev (train + validation) ; période test non affichée |
| Moteur de prévision | **ML gelé** (HistGradientBoosting quantile 0,5) — validation, glissant J+1, 138 210 h : MAE 0,0456 vs B1 0,0482 vs B0 0,0475 ; biais −0,031 / −0,020 / −0,029. Détails : `engine_freeze.json`. |
| Test final (unique, scellé) | 2026-09-26 22:23:00 UTC, période 2024-06-11 → 2024-06-25, 500 séries, 134 744 h : **ML MAE 0,04915, biais −0,03553** ; B1 0,05077 / −0,02666 ; B0 0,05080 / −0,03578. Résultat brut : `logs/final_test_result.json`. Plus aucune modification du moteur. |

Offre gratuite : mise en veille après 15 min sans visite, réveil ≈ 1 min. Ouvrir l'URL quelques
minutes avant une démo.

## Fonctions livrées

- F1 DATA : sous-ensemble déterministe, contrat de données, validation d'entrée.
- F2 REVEAL : ventes horaires et disponibilité par série.
- F3 VERIFY : B0 / B1 / ML comparés sur la validation, moteur ML gelé ; test masqué et anti-fuite
  (hors écran). L'écran affiche encore B0 / B1 seulement.
- F4 DECIDE : panier optimal sous budget (coûts et stocks = hypothèses saisies).
- IA responsable : abstention visible à l'écran ; intervalle 80 % du moteur, couverture
  empirique 79,5 % sur la validation (jours jamais vus en calibration).
- Interface : 3 écrans dans l'ordre Vérifier → Comprendre → Acheter ; design system sobre
  (design-system/MASTER.md), contrastes WCAG AA vérifiés par les tests.
- F5 ACT : pas encore commencé.
