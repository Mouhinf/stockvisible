# Sécurité — StockVisible

## Audit M13 (2026-09-27)

**Méthode** : audit en lecture seule par un agent indépendant, sans le contexte du développement.
Il a regardé l'historique Git complet (`git log -p --all`), les dépendances (`pip-audit`,
`npm audit`), le chemin d'import de fichiers, les exécutions ou désérialisations possibles et la
configuration de déploiement. Il a tout vérifié par des commandes, et plusieurs essais de charge
ont été faits hors du dépôt. Le mainteneur a ensuite recontrôlé chaque affirmation dans le code,
ainsi que l'absence de secrets dans l'historique.

**Résultat : 0 P0, 1 P1, 6 P2.** Règle du milestone : seuls les P0 sont corrigés
immédiatement. Il n'y en a aucun, donc aucun code n'a été modifié pour M13. Les points ci-dessous
restent **ouverts**, par ordre de priorité.

| Niveau | Définition |
|---|---|
| P0 | à corriger tout de suite : secret exposé, exécution de code à distance, désérialisation non sûre d'une entrée utilisateur, CVE exploitable dans une dépendance déployée |
| P1 | risque réel, mais pas exploitable immédiatement, ou limité, ou propre au dev |
| P2 | durcissement, hygiène, information |

## Suivi des correctifs (M16, 2026-09-27)

| ID | Statut | Correctif | Preuve |
|---|---|---|---|
| P1-1 | **corrigé** | `MAX_BYTES` et `server.maxUploadSize` = 20 Mo ; Parquet : nombre de lignes (≤ 20 000), de colonnes (≤ 64) et taille décompressée (≤ 64 Mo) lus dans les métadonnées **avant** tout décodage ; CSV : plafonds de lignes et de colonnes avant lecture, cellule liste ≤ 1 000 caractères avant `json.loads`. Re-mesuré : le Parquet de 5 M lignes (63 Ko) est refusé sur ses métadonnées ; un fichier valide au plafond (19 980 lignes) ajoute ≈ 30 Mo. | `tests/test_validation.py` (plafonds) |
| P2-1 | **corrigé** | Types Parquet contrôlés (struct / map / liste interdits hors `hours_sale`, `hours_stock_status` ; listes de nombres seulement) ; `RecursionError` capturée ; `TypeError` / `ValueError` résiduelles transformées en refus propre à l'écran ; `client.showErrorDetails = "none"`. | idem |
| P2-2 | **corrigé** | Le détail d'un refus est affiché en texte brut (`st.text`), jamais en markdown. | `ui/pages/comprendre.py` |
| P2-3 à P2-6 | ouverts | inchangés (voir ci-dessous). | — |

Barre d'outils : `client.toolbarMode = "viewer"` (plus de bouton « Deploy » ni d'options de
développement pour les visiteurs).

## P1 — historique de l'audit M13

### P1-1 — Un petit fichier importé peut saturer la mémoire du serveur (corrigé en M16)

- **Où** :
  - `stockvisible/validation.py:22` : `MAX_BYTES` vaut 200 Mo ;
  - `stockvisible/validation.py:112` : `pq.read_table(...).to_pandas()`, sans limite de lignes ;
  - `stockvisible/validation.py:125-138` : le CSV et les cellules JSON sont décodés entièrement ;
  - `server.maxUploadSize` n'est pas réglé (défaut de Streamlit : 200 Mo).
- **Preuve mesurée** :
  - un Parquet de 178 Ko (20 M lignes de zéros, compressé en zstd) fait monter le processus à
    2,1 Go de mémoire pendant `read_user_bytes` + `validate` ;
  - une cellule `hours_sale` de 38 Mo en CSV ajoute environ 240 Mo.
- **Impact** : l'app est publique et sans authentification, sur une seule instance Render de
  512 Mo. Un seul envoi peut faire tuer le service, pour tout le monde. Ce n'est pas une exécution
  de code.
- **Correctif proposé** :
  - `server.maxUploadSize` et `MAX_BYTES` entre 10 et 20 Mo ;
  - lire `pq.ParquetFile(...).metadata` avant de charger quoi que ce soit, et refuser au-delà d'un
    plafond de lignes ou de taille décompressée ;
  - pour le CSV, un plafond de lignes (`nrows`) et de longueur de cellule avant `json.loads`.

## P2 — constats de l'audit M13 (P2-1 et P2-2 corrigés en M16)

| ID | Constat | Où | Correctif proposé |
|---|---|---|---|
| P2-1 | Certaines erreurs d'import ne sont pas transformées en refus propre et remontent avec leur trace (`RecursionError` sur un JSON imbriqué 50 000 fois, `TypeError` sur un Parquet à colonnes liste ou struct). `client.showErrorDetails` reste au défaut `"full"`. | `validation.py:137`, `:221` ; `ui/pages/comprendre.py` | Capturer `RecursionError`, `TypeError` et `ValueError` sur le chemin d'import ; contrôler le type des colonnes scalaires ; `showErrorDetails = "none"`. |
| P2-2 | Des fragments du fichier envoyé (extension, début de cellule, message pyarrow ou pandas) sont affichés en **markdown** dans `st.error`. Le HTML est échappé, mais un lien ou une image peut apparaître. Cela ne touche que la session de celui qui envoie. | `ui/pages/comprendre.py:69` | Afficher ces détails avec `st.code` ou `st.text`, ou échapper le markdown. |
| P2-3 | `np.arange` est alloué avant le contrôle `MAX_COMBINATIONS`. La limite `min_value=0.1` de la taille de lot n'est appliquée que dans le navigateur ; **non vérifié de bout en bout** : un message websocket forgé pourrait provoquer une allocation de plusieurs Go. | `stockvisible/allocation.py:92-100` ; `ui/pages/acheter.py` | Calculer le nombre de combinaisons avant toute allocation ; imposer `lot_size >= 0.1` côté serveur. |
| P2-4 | Des outils de dev (pytest, ruff, pip-audit) sont installés en production. Les dépendances transitives ne sont pas figées, donc le build Render peut différer de l'environnement audité. | `requirements.txt:7-9`, `render.yaml` | Séparer un `requirements-dev.txt`, figer les versions transitives dans un fichier de contraintes. |
| P2-5 | La version de pip du venv local (24.0) a 12 alertes (PYSEC-2026-1795, -1796, -2875, -2876, -196, -3721 ; corrigées en 25.3 à 26.2). C'est de l'outillage local, pas un élément déployé. | `.venv` | `.venv/bin/pip install -U pip`. |
| P2-6 | Échappement des formules à l'export : une valeur qui commence par un espace puis `=` n'est pas préfixée. En pratique, les colonnes texte ne sont pas saisies par l'utilisateur. | `stockvisible/exports.py:21,66` | Faire `lstrip()` avant le test des préfixes. |

## Vérifié et propre

- **Secrets** : l'historique Git complet (17 commits) ne contient aucun jeton (`hf_`, `ghp_`,
  `gho_`, `github_pat_`, `rnd_`, `AKIA`, `sk-`, `xox`), aucune clé privée, aucune affectation de
  secret. Aucun stash ni objet inaccessible. `.env`, `.claude/settings.local.json` et
  `.streamlit/secrets.toml` sont ignorés et n'ont jamais été commités. `.env.example` ne contient
  que des commentaires.
- **Données personnelles** : seul l'e-mail d'auteur des commits apparaît (public sur GitHub).
  Seules adresses IP : `0.0.0.0` et `127.0.0.1`. Aucun chemin local dans les fichiers suivis.
- **Dépendances déployées** : `pip-audit -r requirements.txt` ne trouve aucune vulnérabilité ;
  `npm audit` non plus (outil de test, jamais déployé).
- **Exécution et désérialisation** : aucun `eval`, `exec`, `pickle`, `joblib`, `marshal`, `shelve`,
  `yaml.load`, `read_pickle` ni `allow_pickle`. Les listes CSV sont lues avec `json.loads`, avec
  contrôle de type. pyarrow 25.0.1 n'a plus `PyExtensionType` (CVE-2023-47248 ne s'applique pas).
  Un Parquet doit porter la signature `PAR1` au début et à la fin. Les extensions acceptées sont
  sur liste blanche, les octets nuls sont refusés et l'UTF-8 est obligatoire.
- **HTML injecté** : `unsafe_allow_html` n'est utilisé qu'une fois (`app.py`), pour du CSS constant.
- **Sous-processus** : `final_test.py` appelle `git` avec des arguments fixes, sans shell. Ce
  module n'est pas importé par l'interface.
- **Téléchargement du dataset** (`data.py`) : HTTPS, révision figée, SHA-256 vérifié. Jamais appelé
  par l'app.
- **Configuration** : XSRF et CORS aux valeurs par défaut (activés), `gatherUsageStats = false`,
  `0.0.0.0` seulement sur Render (nécessaire), `localhost` en local.
- **Artefacts publiés** (`logs/*.json`, `engine_freeze.json`) : uniquement des métriques, des
  empreintes et des chemins du dépôt.

## Signaler une vulnérabilité

Ouvrir un avis privé (*Security advisory*) sur le dépôt GitHub `Mouhinf/stockvisible` plutôt
qu'une issue publique.
