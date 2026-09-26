# Statut

| | |
|---|---|
| Dernier milestone | M6-first-deploy |
| URL publique | https://stockvisible.onrender.com |
| Vérification HTTPS | 2026-09-26 18:4x GMT : `curl -sf` → 200 ; `/_stcore/health` → `ok` ; http → 301 vers https ; certificat valide (expire le 2026-12-20) |
| Vérification navigateur | session complète sur l'URL publique : graphique, 4 tableaux, changement de série ; 0 erreur page / console / logs |
| Hébergeur | Render, service Web `srv-das10h3bc2fs738t57ag`, offre gratuite, région Frankfurt |
| Déploiement | automatique à chaque push sur `master` (github.com/Mouhinf/stockvisible), une fois GitHub connecté à Render (2026-09-26). Le push de d3307c3, fait avant la connexion, n'avait déclenché aucun déploiement. |
| Données servies | sous-ensemble RÉEL dev (train + validation) ; période test non affichée |

Offre gratuite : mise en veille après 15 min sans visite, réveil ≈ 1 min. Ouvrir l'URL quelques
minutes avant une démo.

## Fonctions livrées

- F1 DATA : sous-ensemble déterministe, contrat de données, validation d'entrée.
- F2 REVEAL : ventes horaires et disponibilité par série.
- F3 VERIFY : B0 / B1 sur la validation ; test masqué et anti-fuite (hors écran).
- F4 DECIDE : panier optimal sous budget (coûts et stocks = hypothèses saisies).
- F5 ACT : pas encore commencé.
