# StockVisible

Voir les ventes **et** la disponibilité réelle en rayon, vérifier les prévisions (B0 / B1, puis ML)
et décider d'un panier d'achat sous budget. Règles du projet : [CLAUDE.md](CLAUDE.md).
État du déploiement : [STATUS.md](STATUS.md).

## Données

Sous-ensemble déterministe (500 séries, seed 42) de **FreshRetailNet-50K**, Dingdong-Inc,
licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), révision
`08c1fab7f9257bc73679d415d65d644165d351d4`
([arXiv:2505.16319](https://arxiv.org/abs/2505.16319)). Les ventes sont normalisées par le
fournisseur : aucune unité monétaire. Le fichier versionné est `data/raw/train.parquet`
(+ `manifest.json`). Il a été filtré sur les séries retenues ; aucune valeur n'a été modifiée.

Reconstruire les données depuis la source : `python -m stockvisible.data`.

## Local

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q -m "not touches_test_period"   # run dev-only
.venv/bin/streamlit run app.py                     # http://localhost:8501
```

## Déploiement Render

Le fichier [`render.yaml`](render.yaml) décrit le service (Blueprint). Pour une configuration
à la main dans le dashboard, reporter les mêmes valeurs :

| Champ | Valeur |
|---|---|
| Runtime | Python 3 |
| Branch | `master` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| Health Check Path | `/_stcore/health` |
| Variable d'environnement | `PYTHON_VERSION` = `3.12.3` |
| Instance | Free (512 Mo, mise en veille après 15 min d'inactivité) |

Aucun secret n'est nécessaire.
