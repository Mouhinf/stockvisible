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
| Coût d'exécution | Aucun modèle n'est entraîné en production : l'application lit les résultats versionnés. Réentraîner sur les 533 071 heures d'entraînement a pris 17,6 s et 691 Mo de mémoire au maximum (mesure du 27/09/2026 sur la machine de développement, 2 cœurs, un thread OpenMP), au-delà d'une instance gratuite de 512 Mo |

Aucun LLM, aucune API d'IA externe, aucun contenu généré n'est utilisé **dans** l'application.

## 2. IA utilisée pour développer

| Outil | Rôle réel |
|---|---|
| Claude Code (Anthropic) — Claude Sonnet 4.6 pour le premier commit, Claude Opus 5.5 ensuite | Assistant de programmation : écriture du code, des tests et de la documentation, exécution des commandes (tests, déploiement, Git). Le porteur du projet fixe les règles (discipline scientifique, stack figée, interdits, dans un fichier de consignes local non versionné) et prend les décisions notées « décision utilisateur » dans le code et l'historique (ex. correctif M7b, protocole du test final M9, choix d'intervalle M10). Chaque commit porte la mention `Co-Authored-By` correspondante |
| Claude, dans une session séparée | Les 4 commits de finition de la pull request #1 (bandeau du problème, durcissement de l'import, dégagement de l'en-tête, documentation jury) : auteur Git « Claude », importés dans le dépôt sous forme de patch |
| Claude, en agent séparé et en lecture seule | Audit de sécurité (M13, `SECURITY.md`) ; évaluation simulée selon la grille du jury, avant soumission |
| Skills Claude Code | `stockvisible-data-contract` : skill propre au projet (contrat de données et journal des décisions, repris dans `docs/data-contract.md`). `dataviz` : méthode et validation des couleurs des graphiques (M4). La skill `ui-ux-pro-max` n'était pas disponible et n'a pas été utilisée : la palette a été faite à la main puis vérifiée par calcul (contrastes WCAG) |
| Serveurs MCP | **Aucun.** Ni Context7 ni GitHub MCP n'ont été utilisés. GitHub est piloté avec le CLI `gh`, Render avec le CLI `render`, la documentation de Streamlit a été consultée par inspection locale des API installées |

Les décisions scientifiques (découpage, règle de sélection, gel, ouverture unique du test) ont été
fixées **avant** de voir les résultats correspondants et sont tracées dans l'historique Git.

### Ce qui a été codé, et quand

L'historique Git fait foi (heures UTC). Tout le code du dépôt a été écrit à partir du premier
commit, avec les outils ci-dessus ; aucun code antérieur n'a été réutilisé, hormis des
bibliothèques open source (`requirements.txt`, `package.json`) et le jeu de données.

| Période | Milestones | Contenu |
|---|---|---|
| 26/09/2026, 16:03 → 23:29 | M0 → M10 | Structure, contrat de données, découpage, B0 / B1, test masqué, écran, panier, premier déploiement, modèle ML, gel, test final unique, incertitude |
| 27/09/2026, 00:56 → 13:11 | M11 → M17 | Trois écrans et design system, import et export, tests E2E, audit de sécurité, CI, production, DNS, finitions et documentation |

La date du hackathon et ses règles sur le code préparé avant l'événement sont celles du règlement
officiel : c'est au porteur du projet de confronter ce tableau au règlement.

## 3. Données

FreshRetailNet-50K (Dingdong-Inc), licence CC BY 4.0, révision
`08c1fab7f9257bc73679d415d65d644165d351d4`. Données réelles de commerce de produits frais
(Chine), ventes normalisées par le fournisseur : **aucune unité monétaire, aucune donnée du
Sénégal, aucune donnée personnelle**. Les fichiers d'import des tests E2E sont synthétiques et
étiquetés comme tels (`*_SYNTHETIQUE.csv`).

## 4. NVIDIA Brev

**Non utilisé.** Le projet tourne sur CPU (entraînement en quelques secondes) ; aucun crédit
n'a été demandé.
