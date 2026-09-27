# StockVisible — design system (MASTER)

Outil Data/B2B de confiance : sobre, lisible, chiffré. La preuve passe avant la décoration.
Ce fichier fait référence. Sa traduction se trouve dans `ui/theme.py` (jetons + CSS injecté) et
dans `.streamlit/config.toml` (thème natif Streamlit). Toute modification commence ici.

## Principes

1. **Preuve d'abord** : l'ordre des écrans est Vérifier → Comprendre → Acheter.
2. **Un chiffre = une source.** Aucune valeur de résultat n'est tapée dans l'interface : tout est
   lu dans `engine_freeze.json`, `logs/*.json` ou calculé en direct.
3. **Au plus 3 indicateurs par écran.** Au-delà, on passe à un tableau.
4. **L'incertitude et les limites sont affichées, pas cachées.** Abstention, « intervalle non
   calibré », biais.
5. **Interdits** : dégradé, néon, ombre marquée, emoji décoratif, couleur sans libellé.

## Couleurs (thème clair uniquement)

| Jeton | Hex | Rôle | Contraste sur #FFFFFF / #F5F6F8 |
|---|---|---|---|
| `background` | `#FFFFFF` | fond de page | — |
| `surface` | `#F5F6F8` | encadrés, en-têtes de tableaux, champs | — |
| `border` | `#D9DDE3` | séparateurs fins (décoratif) | 1,36 (non textuel) |
| `text` | `#1B1F24` | titres, valeurs | 16,56 / 15,31 |
| `text_secondary` | `#4B5563` | légendes, notes | 7,56 / 6,99 |
| `accent` | `#1F4E8C` | actions : curseur, onglet actif, focus, tags | 8,31 / 7,69 |
| `success` | `#1E7B4A` | état « calibré », « scellé » | 5,27 / 4,87 |
| `warning` | `#A15C00` | abstention partielle, limites | 5,19 / 4,80 |
| `critical` | `#B42318` | abstention totale, erreur | 6,57 / 6,08 |

Tous les textes respectent WCAG 2.1 AA (≥ 4,5:1), vérifié par calcul dans `tests/test_design.py`.
Les couleurs d'état sont RÉSERVÉES : jamais utilisées pour une série de données, toujours
accompagnées d'un libellé.

### Données (graphiques) — inchangées depuis M4

Ce sont les 3 premiers slots de la palette de référence dataviz, validés en clair, toutes paires
confondues.

| Série | Hex | Second codage |
|---|---|---|
| Ventes observées | `#2a78d6` | trait plein |
| B0 | `#eb6834` | tirets |
| B1 | `#1baf7a` | pointillés (< 3:1 : relief assuré par le tableau) |
| Disponibilité | `#52514e` | aire grise, panneau séparé |

L'accent d'interface (`#1F4E8C`) est volontairement distinct du bleu des données (`#2a78d6`).

## Typographie

- Pile système, sans chargement externe : `system-ui, -apple-system, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif`.
- Tailles :

  | Niveau | Taille | Graisse |
  |---|---|---|
  | titre de page (h1) | 1,75 rem (28 px) | 600 |
  | section (h2) | 1,25 rem (20 px) | 600 |
  | sous-section (h3) | 1,0625 rem (17 px) | 600 |
  | corps | 15 px | 400 |
  | légendes | 13 px | 400 |

- Interligne du corps : 1,5.
- Chiffres : `font-variant-numeric: tabular-nums` (colonnes alignées).

## Espacement et forme

- Échelle de 8 px : 4 · 8 · 16 · 24 · 32 · 48.
- 32 px entre sections, 16 px entre un titre et son contenu.
- Largeur de contenu maximale : 1 200 px.
- Rayon des coins : 6 px. Bordures de 1 px, sans ombre.

## Écrans

| # | Écran | Contenu | Indicateurs (max 3) |
|---|---|---|---|
| 1 | **Vérifier** | Phrase-titre ; un tableau validation + test final (ML, B1, B0) ; garanties ; limites ; provenance | couverture de l'intervalle |
| 2 | **Comprendre** | Série ; ventes + disponibilité ; B0/B1 de la série ; abstention | — |
| 3 | **Acheter** | Hypothèses (coût, lot, stock) ; budget ; panier | coût, demande couverte, manque |

La demande de l'écran Acheter vient de B1 (reconstruction des heures en rupture), et non du
moteur ML gelé. C'est dit explicitement à l'écran.
