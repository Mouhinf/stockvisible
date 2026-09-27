# Déclaration d'utilisation de l'IA — StockVisible

Ce document sert de base au formulaire « Outils IA et utilisation de NVIDIA Brev » du hackathon
GOMYCODE. Il sépare l'IA **dans le produit** de l'IA **utilisée pour développer** le produit.

## 1. IA dans le produit

| Élément | Détail |
|---|---|
| Modèle | `HistGradientBoostingRegressor` (scikit-learn 1.9.1), perte quantile 0,5, `random_state = 42`, réglages par défaut sauf `early_stopping = False` |
| Tâche | Prévoir la vente de chaque heure du lendemain (J+1) pour 500 séries magasin × produit |
| Variables | Heure, jour de semaine, jour férié, ventes de J-1 et J-7 et médiane glissante sur 7 jours (heures disponibles uniquement), remise de J-1, hiérarchie produit / ville. Aucune donnée du jour prévu (anti-fuite testé) |
| Concurrents | B0 : médiane des ventes au même créneau, ruptures comprises. B1 : même médiane limitée aux heures disponibles, avec abstention |
| Choix | Règle fixée avant tout résultat : ML retenu seulement si MAE(ML) < MAE(B1) sur la validation. Choix fait sur la validation, puis configuration gelée (`engine_freeze.json`, empreintes SHA-256) |
| Preuve | Test final ouvert une seule fois (`logs/final_test_result.json`) : MAE 0,0492 (ML) contre 0,0508 (B1) et 0,0508 (B0) sur 134 744 heures communes |
| Incertitude | Intervalle 80 % empirique autour du moteur, couverture mesurée 79,5 % sur des jours de validation jamais vus en calibration |
| Ce que le ML ne fait pas | Il **ne calcule pas le panier**. La demande de l'écran Acheter est estimée avec B1 (ventes des heures disponibles + B1 aux heures en rupture). Le ML sous-prévoit (biais −0,0355 sur le test) : l'utiliser tel quel pour acheter pousserait à commander trop peu |
| Solution de repli | Si le ML avait perdu sur la validation, B1 devenait le moteur (règle écrite en M7). Le panier ne dépend pas du ML |
| Coût d'exécution | Aucun modèle n'est entraîné en production : l'application lit les résultats versionnés. Réentraîner prend environ 8 s mais environ 850 Mo de mémoire, au-delà d'une instance gratuite de 512 Mo |

Aucun LLM, aucune API d'IA externe, aucun contenu généré n'est utilisé **dans** l'application.

## 2. IA utilisée pour développer

| Outil | Rôle réel |
|---|---|
| Claude Code (Anthropic) — Claude Sonnet 4.6 pour le premier commit, Claude Opus 5.5 ensuite | Assistant de programmation : écriture du code, des tests et de la documentation. Le porteur du projet fixe les règles (`CLAUDE.md` : discipline scientifique, stack figée, interdits) et prend les décisions notées « décision utilisateur » dans le code (ex. correctif M7b, protocole du test final M9). Chaque commit porte la mention `Co-Authored-By` correspondante |
| Claude (Anthropic), en agent séparé | Audit de sécurité en lecture seule (M13, `SECURITY.md`) et audit final avant soumission |

Les décisions scientifiques (découpage, règle de sélection, gel, ouverture unique du test) ont été
fixées **avant** de voir les résultats correspondants et sont tracées dans l'historique Git.

## 3. Données

FreshRetailNet-50K (Dingdong-Inc), licence CC BY 4.0, révision
`08c1fab7f9257bc73679d415d65d644165d351d4`. Données réelles de commerce de produits frais
(Chine), ventes normalisées par le fournisseur : **aucune unité monétaire, aucune donnée du
Sénégal, aucune donnée personnelle**. Les fichiers d'import des tests E2E sont synthétiques et
étiquetés comme tels (`*_SYNTHETIQUE.csv`).

## 4. NVIDIA Brev

**Non utilisé.** Le projet tourne sur CPU (entraînement en quelques secondes) ; aucun crédit
n'a été demandé.
