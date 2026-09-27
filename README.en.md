# StockVisible

*Version française : [README.md](README.md).*

**See the demand that stock-outs hide, then decide what to reorder within budget — with forecasts
whose reliability is proven.**

Application: https://stockvisible.galsentechnologie.com (fallback: https://stockvisible.onrender.com)

## The problem

When a product is out of stock, the till records zero sales. That zero is not an absence of
customers: observed sales **underestimate demand**, and a manager who reorders on that basis
reorders too little, which sustains the stock-outs. In the data used (500 real store × product
series of fresh produce), **20.2% of the hours from 6:00 to 22:00 are out of stock** over the
training period (`data/raw/manifest.json`).

## User

The **manager or purchasing lead of a fresh-produce outlet**, who must decide every day what to
reorder with a limited budget. They do not need to read the code: the application first shows them
the evidence of reliability (Verify screen), then the demand for one product (Understand), then a
basket within budget that they validate and export (Buy).

No interview or pilot with a manager has taken place yet: this need is a working assumption, not
a need validated in the field.

## What the application does (3 screens)

| Screen | What you see |
|---|---|
| **Verify** (home) | The problem in one sentence, then the evidence: errors of the ML engine and of the two baselines on validation and on the final test, uncertainty, guarantees, limitations |
| **Understand** | For one product: hour-by-hour sales, availability, observed sales vs estimated demand per day, B0 / B1 forecasts, abstention; import of a file checked against the data contract |
| **Buy** | Costs, lots and stock levels entered as assumptions; budget; optimal basket (covered demand, shortfall); validation then CSV / JSON export with provenance |

## Architecture

Five functions in `stockvisible/`, the interface in `ui/`.

| Function | Modules | Role |
|---|---|---|
| F1 DATA | `data.py`, `validation.py` | Deterministic, verified (SHA-256) subset of the dataset; data contract; safe reading of imported files (CSV / Parquet, size and row caps, no execution of content) |
| F2 REVEAL | `ui/pages/comprendre.py`, `ui/charts.py` | Hourly sales, declared availability, estimated demand |
| F3 VERIFY | `splits.py`, `baselines.py`, `features.py`, `model.py`, `evaluation.py`, `selection.py`, `final_test.py`, `uncertainty.py` | Chronological split, B0 / B1 baselines, strictly past features, ML model, selection on validation, freeze, single final test, empirical interval |
| F4 DECIDE | `allocation.py` | Optimal basket within budget, bounded exhaustive enumeration (≤ 3 products), whole lots |
| F5 ACT | `exports.py` | Basket validation, CSV / JSON export strictly identical to the screen |

- `app.py`: router of the three screens (`st.navigation`). No model is trained at run time: the
  Verify screen reads `engine_freeze.json`, `logs/final_test_result.json` and
  `logs/interval_calibration.json`.
- Interface: Streamlit 1.64, Plotly charts, design system in `design-system/MASTER.md`.
- Data contract and decision log: [docs/data-contract.md](docs/data-contract.md).

## Results (read from `engine_freeze.json` and `logs/final_test_result.json`)

Rolling day-ahead (J+1) hourly forecasts, hours when the product was available, the same hours for
all three models. Normalised values, no unit.

| Model | Validation MAE | Final test MAE | Final test bias |
|---|---|---|---|
| **ML (frozen engine)** | 0.0456 | **0.0492** | −0.0355 |
| B1 (median, available hours) | 0.0482 | 0.0508 | −0.0267 |
| B0 (median, stock-outs included) | 0.0475 | 0.0508 | −0.0358 |

- On the final test, opened **a single time** (2026-09-26 22:23 UTC, 134,744 common hours), the
  ML's error is 3.2% lower than B1's (5.4% on validation).
- It **under-forecasts** (negative bias, close to B0): this is shown on screen, and it is why the
  basket does not use it.

## Scientific discipline

DATA → TRAIN (60 d) → VALIDATION (15 d) → ENGINE CHOICE → FREEZE → FINAL TEST ONCE ONLY (15 d).

- Chronological split per series, never random.
- Selection rule written before any result: the ML is kept only if it beats B1 on validation,
  otherwise B1 becomes the engine.
- Anti-leakage tested: canaries, perturbation of future days, and 9 negative controls (leaks
  injected on purpose, which the suite must detect).
- Two looks at the validation set are declared: the first model was degenerate (it predicted 0
  everywhere); it was fixed once, then the configuration was frozen.

## Responsible AI

- **No LLM** and no external AI API in the application: the engine is a tabular model
  (scikit-learn) competing against two simple rules.
- **Visible abstention**: when no comparable period exists, B1 predicts nothing and the screen
  says so (total or partial); the hours stay empty, never filled by default.
- **Measured uncertainty**: 80% interval around the engine, empirical coverage of 79.5% on
  validation days never seen during calibration; “interval not calibrated” is shown when support
  is insufficient. At night the interval is [0, 0]: calibrated but not very informative.
- **Labelling**: real data flagged as such; costs and stock levels flagged as assumptions;
  synthetic test files named `*_SYNTHETIQUE.csv`; lost demand always presented as **estimated**.
- **Limitations shown on screen** (Verify screen), not only in this document.
- AI tools used to build the project: [AI_USAGE.md](AI_USAGE.md) (in French).

## Data

Deterministic subset (500 series, seed 42) of **FreshRetailNet-50K**, Dingdong-Inc,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence, revision
`08c1fab7f9257bc73679d415d65d644165d351d4`
([arXiv:2505.16319](https://arxiv.org/abs/2505.16319)). Sales are normalised by the provider: no
currency unit. The versioned file is `data/raw/train.parquet` (+ `manifest.json`), filtered on the
selected series, without any value modified. The dataset's official “eval” file is not versioned
and has never been used to evaluate a model.

Rebuild the data from the source: `python -m stockvisible.data`.

## Limitations

- Real **Chinese** data (Dingdong), normalised: no currency unit, no FCFA, no data from Senegal.
- Costs, lot sizes and stock levels are **entered assumptions**: the dataset contains none.
- Demand lost during a stock-out is **estimated**, never observed. The masked-data test measures
  the fidelity of the reconstruction, not the real lost demand.
- Modest ML gain (3.2% on the test) and negative bias; the basket uses B1.
- The rows of the test period were read before the split (data quality check) and are public in
  the repository; they were not used for any model decision.
- Render free tier: cold start observed at about 2 minutes after 15 minutes of inactivity.

## Installation

Python 3.12 required.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py                       # http://localhost:8501
```

## Tests

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q                                  # full suite (519 tests collected)
.venv/bin/pytest -q -m "not touches_test_period"     # without reading the reserved test period
npm ci && npx playwright test                        # E2E path, desktop + mobile
E2E_BASE_URL=https://stockvisible.onrender.com npx playwright test   # same path in production
```

On a fresh clone, 2 tests are skipped because they need data files that are not versioned
(official eval, download cache), with an explicit reason.
GitHub Actions CI: ruff, pytest, pip-audit (informational). Security: [SECURITY.md](SECURITY.md).
Status and history: [STATUS.md](STATUS.md). Evidence per jury criterion:
[docs/judging-evidence.md](docs/judging-evidence.md) (in French).

## Deployment

Render, web service, Frankfurt region, redeployed on every push to `master`. The
[`render.yaml`](render.yaml) file describes the service. Custom domain: CNAME `stockvisible` →
`stockvisible.onrender.com`, declared in Render → Settings → Custom Domains.

| Field | Value |
|---|---|
| Runtime | Python 3 |
| Branch | `master` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| Health Check Path | `/_stcore/health` |
| Environment variable | `PYTHON_VERSION` = `3.12.3` |
| Instance | Free (512 MB, sleeps after 15 min of inactivity) |

No secret is needed.

## Licences

- **Data**: FreshRetailNet-50K, Dingdong-Inc, CC BY 4.0 (attribution above).
- **Dependencies**: open-source libraries listed in `requirements.txt` (pandas, NumPy,
  scikit-learn, Streamlit, Plotly, pyarrow, pytest, ruff, pip-audit) and `package.json`
  (@playwright/test, testing tool only), each under its own licence.
- **StockVisible code**: no licence file has been added to the repository yet; without an explicit
  licence, rights remain reserved to the author. Choosing a licence is a next step.
- **Tab logo**: Galsen Technologie icon (galsentechnologie.com).

## Next step

Pilot with an outlet equipped with a till and stock tracking, with its real costs; then a
purchase-oriented engine (quantile above the median) chosen on validation and evaluated on the
dataset's official “eval” period, never used to evaluate a model.

## Timeline

The Git history is authoritative. First commit on 26/09/2026 at 16:03 UTC; data, model, freeze,
final test and screens (M0 to M11) committed on 26/09/2026 and overnight until 27/09 at 00:56 UTC;
E2E tests, security audit, export, CI, production and finishing work on 27/09/2026.
