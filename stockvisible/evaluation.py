"""Métriques et évaluation — F3 VERIFY. Dev uniquement (train + validation).

Protocole fixé avant tout résultat (M2). Périmètre principal : heures déclarées disponibles
dans la cible (statut 0), où la vente observée est la demande non censurée. Les modèles sont
comparés sur les mêmes cellules. Biais = moyenne(prédiction − réel) : négatif = sous-prévision.

Test à données masquées (M3) : test de RECONSTRUCTION / FIDÉLITÉ sur des heures disponibles
proches de ruptures organiques réelles. Ce n'est PAS une preuve de récupération de la demande
perdue pendant une rupture réelle : les valeurs masquées sont des ventes observées.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from stockvisible.baselines import (
    HourlyForecast,
    b0_forecast,
    b1_forecast,
    check_no_future,
    hourly_matrix,
)
from stockvisible.data import SERIES_KEY

HOURS = 24


def error_stats(pred: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    keep = mask & ~np.isnan(pred) & ~np.isnan(actual)
    n = int(keep.sum())
    if n == 0:
        return {"n": 0, "mae": np.nan, "bias": np.nan}
    err = pred[keep] - actual[keep]
    return {"n": n, "mae": float(np.abs(err).mean()), "bias": float(err.mean())}


def compare_baselines(
    targets: pd.DataFrame, forecasts: list[HourlyForecast], label: str
) -> pd.DataFrame:
    actual = hourly_matrix(targets, "hours_sale")
    available = hourly_matrix(targets, "hours_stock_status") == 0
    common = np.logical_and.reduce([~f.abstained for f in forecasts])
    everything = np.ones_like(available)
    scopes = [
        ("heures disponibles, cellules communes", available & common),
        ("heures disponibles, toute la couverture", available),
        ("toutes heures (cible censurée), cellules communes", everything & common),
    ]
    rows = []
    for scope, mask in scopes:
        for f in forecasts:
            stats = error_stats(f.pred, actual, mask)
            rows.append({"split": label, "périmètre": scope, "modèle": f.name,
                         "couverture": f.coverage, **stats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- masquage


@dataclass(frozen=True)
class MaskSpec:
    """Fixé avant tout résultat (M3). window = heures [6, 22), comme stock_hour6_22_cnt."""

    n_per_stratum: int = 250
    max_distance: int = 3
    window: tuple[int, int] = (6, 22)
    seed: int = 42


MASK_SPEC = MaskSpec()
QUERY_COLUMNS = ["query_id", "row", *SERIES_KEY, "dt", "hour", "distance", "side"]


def canonical(targets: pd.DataFrame) -> pd.DataFrame:
    return targets.sort_values([*SERIES_KEY, "dt"], kind="mergesort").reset_index(drop=True)


def mask_candidates(frame: pd.DataFrame, spec: MaskSpec = MASK_SPEC) -> pd.DataFrame:
    """Heures disponibles de la fenêtre à <= max_distance h d'une rupture organique de la même
    journée (rupture elle aussi dans la fenêtre). Ne lit QUE hours_stock_status, jamais les ventes.
    Une journée sans rupture dans la fenêtre ne fournit aucune position."""
    st = hourly_matrix(frame, "hours_stock_status")
    lo, hi = spec.window
    in_window = np.zeros(HOURS, dtype=bool)
    in_window[lo:hi] = True
    stock = (st == 1) & in_window
    available = (st == 0) & in_window

    ahead = np.full(st.shape, np.inf)
    behind = np.full(st.shape, np.inf)
    for d in range(1, HOURS):
        a = ahead[:, : HOURS - d]
        ahead[:, : HOURS - d] = np.where(np.isinf(a) & stock[:, d:], d, a)
        b = behind[:, d:]
        behind[:, d:] = np.where(np.isinf(b) & stock[:, : HOURS - d], d, b)
    distance = np.minimum(ahead, behind)

    rows, hours = np.nonzero(available & (distance <= spec.max_distance))
    a, b = ahead[rows, hours], behind[rows, hours]
    side = np.where(a < b, "avant_rupture", np.where(b < a, "apres_rupture", "entre_ruptures"))
    return pd.DataFrame(
        {"row": rows, "hour": hours, "distance": distance[rows, hours].astype(int), "side": side}
    )


def sample_positions(candidates: pd.DataFrame, spec: MaskSpec = MASK_SPEC) -> pd.DataFrame:
    """Tirage stratifié (distance × côté), seed fixée, min(n_per_stratum, disponibles) par strate."""
    if candidates.empty:
        raise ValueError("aucune position candidate : pas de rupture organique dans la fenêtre")
    rng = np.random.default_rng(spec.seed)
    chosen = []
    for _, group in candidates.groupby(["distance", "side"], sort=True):
        take = min(spec.n_per_stratum, len(group))
        chosen.append(group.iloc[np.sort(rng.choice(len(group), size=take, replace=False))])
    return pd.concat(chosen).sort_values(["row", "hour"]).reset_index(drop=True)


class TruthVault:
    """Coffre de vérité : copie en lecture seule des valeurs masquées, hors de tout DataFrame.

    Il n'est jamais passé aux prédicteurs et n'expose que des scores agrégés.
    """

    __slots__ = ("_strata", "_truth")

    def __init__(self, truth: np.ndarray, strata: pd.DataFrame):
        values = np.array(truth, dtype=float, copy=True)
        values.setflags(write=False)
        self._truth = values
        self._strata = strata.loc[:, ["distance", "side"]].reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self._truth)

    def __repr__(self) -> str:
        return f"<TruthVault {len(self)} valeurs masquées scellées>"

    def _check(self, pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        if pred.shape != self._truth.shape:
            raise ValueError(f"prédictions {pred.shape} != requêtes {self._truth.shape}")
        return pred

    def score(self, pred: np.ndarray, where: np.ndarray | None = None) -> dict[str, float]:
        pred = self._check(pred)
        mask = np.ones(len(self), dtype=bool) if where is None else np.asarray(where, dtype=bool)
        coverage = float((~np.isnan(pred)).mean()) if len(self) else 0.0
        return {"couverture": coverage, **error_stats(pred, self._truth, mask)}

    def score_by_stratum(self, pred: np.ndarray) -> pd.DataFrame:
        pred = self._check(pred)
        rows = []
        for (dist, side), idx in self._strata.groupby(["distance", "side"]).groups.items():
            mask = np.zeros(len(self), dtype=bool)
            mask[np.asarray(idx)] = True
            rows.append({"distance": dist, "side": side, **error_stats(pred, self._truth, mask)})
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class MaskedTask:
    """Tout ce qu'un prédicteur a le droit de voir : cible masquée + requêtes sans valeurs."""

    frame: pd.DataFrame
    queries: pd.DataFrame


def materialize_task(frame: pd.DataFrame, positions: pd.DataFrame) -> tuple[MaskedTask, TruthVault]:
    """frame canonique ; positions (row, hour, distance, side). Aucune écriture sur frame."""
    rows = positions["row"].to_numpy()
    hours = positions["hour"].to_numpy()
    sales = hourly_matrix(frame, "hours_sale")
    vault = TruthVault(sales[rows, hours], positions)

    sales[rows, hours] = np.nan
    masked = frame.copy(deep=True)
    masked["hours_sale"] = [line.tolist() for line in sales]
    masked["sale_amount"] = masked["sale_amount"].astype(float)
    masked.loc[np.unique(rows), "sale_amount"] = np.nan  # sinon masqué = total − heures visibles

    queries = positions.assign(query_id=np.arange(len(positions)))
    for col in (*SERIES_KEY, "dt"):
        queries[col] = frame[col].to_numpy()[rows]
    return MaskedTask(frame=masked, queries=queries.loc[:, QUERY_COLUMNS]), vault


def build_masked_task(
    targets: pd.DataFrame, spec: MaskSpec = MASK_SPEC
) -> tuple[MaskedTask, TruthVault]:
    frame = canonical(targets)
    return materialize_task(frame, sample_positions(mask_candidates(frame, spec), spec))


Predictor = Callable[[pd.DataFrame, MaskedTask], np.ndarray]


def forecast_predictor(forecaster: Callable[[pd.DataFrame, pd.DataFrame], HourlyForecast]) -> Predictor:
    def predict(history: pd.DataFrame, task: MaskedTask) -> np.ndarray:
        f = forecaster(history, task.frame)
        return f.pred[task.queries["row"].to_numpy(), task.queries["hour"].to_numpy()]

    return predict


def run_masked_eval(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    predictors: dict[str, Predictor],
    spec: MaskSpec = MASK_SPEC,
) -> pd.DataFrame:
    task, vault = build_masked_task(targets, spec)
    preds = {name: np.asarray(fn(history, task), dtype=float) for name, fn in predictors.items()}
    common = np.logical_and.reduce([~np.isnan(p) for p in preds.values()])
    rows = []
    for scope, where in (("cellules communes", common), ("couverture propre", None)):
        for name, p in preds.items():
            rows.append({"périmètre": scope, "modèle": name, **vault.score(p, where=where)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- anti-fuite

CANARY_BASE = 7_777_777.0
CANARY_FLOOR = CANARY_BASE / 2  # les ventes réelles normalisées restent < 100


class LeakageError(AssertionError):
    """Une valeur du coffre (ou le futur) est atteignable par la prédiction."""


def history_train_only(train: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    return train


@dataclass(frozen=True)
class MaskedPipeline:
    predictor: Predictor
    build: Callable[[pd.DataFrame, MaskSpec], tuple[MaskedTask, TruthVault]] = build_masked_task
    history_for: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] = history_train_only


def _contains_canary(df: pd.DataFrame) -> bool:
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            arrays = [np.asarray(v, dtype=float).ravel() for v in s if isinstance(v, (list, tuple, np.ndarray))]
            if arrays and (np.nan_to_num(np.concatenate(arrays)) >= CANARY_FLOOR).any():
                return True
        elif pd.api.types.is_numeric_dtype(s) and (np.nan_to_num(s.to_numpy(dtype=float)) >= CANARY_FLOOR).any():
            return True
    return False


def check_no_leakage(
    train: pd.DataFrame, targets: pd.DataFrame, pipeline: MaskedPipeline, spec: MaskSpec = MASK_SPEC
) -> None:
    """Canaris : les valeurs à masquer sont remplacées par des valeurs impossibles (≥ 3,9 M),
    sale_amount restant cohérent. Toute trace de canari dans ce que voit le prédicteur, ou dans
    ses sorties, est une fuite. Lève LeakageError avec la liste des canaux détectés."""
    reasons: list[str] = []
    frame = canonical(targets)
    reference, _ = pipeline.build(frame, spec)
    q = reference.queries
    sales = hourly_matrix(frame, "hours_sale")
    sales[q["row"].to_numpy(), q["hour"].to_numpy()] = CANARY_BASE + 0.125 * np.arange(len(q))
    poisoned = frame.copy(deep=True)
    poisoned["hours_sale"] = [line.tolist() for line in sales]
    poisoned["sale_amount"] = sales.sum(axis=1)

    task, _vault = pipeline.build(poisoned, spec)
    if not task.queries[["row", "hour"]].reset_index(drop=True).equals(q[["row", "hour"]].reset_index(drop=True)):
        reasons.append("positions masquées dépendantes des ventes (biais de sélection sur l'issue)")
    history = pipeline.history_for(train, poisoned)

    try:
        check_no_future(history, task.frame)
    except ValueError as exc:
        reasons.append(f"historique non strictement antérieur à la cible : {exc}")
    for label, df in (("cible masquée", task.frame), ("requêtes", task.queries), ("historique", history)):
        if _contains_canary(df):
            reasons.append(f"valeur du coffre visible dans : {label}")
        if df.attrs:
            reasons.append(f"métadonnées attachées (attrs) à : {label}")

    masked_rows = np.unique(task.queries["row"].to_numpy())
    visible = hourly_matrix(task.frame.loc[masked_rows], "hours_sale")
    residual = task.frame.loc[masked_rows, "sale_amount"].to_numpy(dtype=float) - np.nansum(visible, axis=1)
    if (np.nan_to_num(residual) >= CANARY_FLOOR).any():
        reasons.append("valeur du coffre déductible : sale_amount − somme des heures visibles")

    try:
        pred = np.asarray(pipeline.predictor(history, task), dtype=float)
    except Exception as exc:
        if not reasons:
            raise
        reasons.append(f"prédicteur en erreur : {exc!r}")
    else:
        if pred.shape != (len(task.queries),):
            reasons.append(f"prédictions de forme {pred.shape}")
        elif (np.nan_to_num(pred) >= CANARY_FLOOR).any():
            reasons.append("prédictions contenant des valeurs du coffre")

    if reasons:
        raise LeakageError(" ; ".join(reasons))


# ---------------------------------------------------------------- origine glissante + sélection (M7)

PRIMARY_SCOPE = "heures disponibles, cellules communes"
SELECTION_RULE = (
    "ML retenu seulement si MAE(ML) < MAE(B1), prévisions glissantes à J+1, heures disponibles, "
    "mêmes cellules ; sinon B1 reste le moteur. Fixée avant tout résultat (M7)."
)


def rolling_origin(forecaster: Callable, name: str) -> Callable:
    """Glissant à J+1 : chaque jour D de targets est prévu avec history + les jours de targets < D.
    Le prévisionniste lui-même n'est jamais modifié ; seul l'historique observé s'allonge."""

    def forecast(history: pd.DataFrame, targets: pd.DataFrame) -> HourlyForecast:
        t = targets.reset_index(drop=True)
        shape = (len(t), HOURS)
        pred = np.full(shape, np.nan)
        n_obs = np.zeros(shape, dtype=int)
        abstained = np.ones(shape, dtype=bool)
        for day in sorted(t["dt"].unique()):
            idx = np.flatnonzero((t["dt"] == day).to_numpy())
            past = t[t["dt"] < day]
            f = forecaster(pd.concat([history, past], ignore_index=True), t.iloc[idx])
            pred[idx], n_obs[idx], abstained[idx] = f.pred, f.n_obs, f.abstained
        keys = t.loc[:, [*SERIES_KEY, "dt"]]
        return HourlyForecast(name=name, keys=keys, pred=pred, n_obs=n_obs, abstained=abstained)

    return forecast


def select_engine(
    table: pd.DataFrame, ml: str = "ML (glissant)", baseline: str = "B1 (glissant)"
) -> dict:
    """Applique SELECTION_RULE à une table compare_baselines. Toute ambiguïté → B1."""
    primary = table[table["périmètre"] == PRIMARY_SCOPE].set_index("modèle")
    missing = [m for m in (ml, baseline) if m not in primary.index]
    if missing:
        raise ValueError(f"modèles absents du périmètre principal : {missing}")
    n_ml, n_base = int(primary.at[ml, "n"]), int(primary.at[baseline, "n"])
    if n_ml != n_base or n_ml == 0:
        raise ValueError(f"comparaison invalide : {n_ml} vs {n_base} cellules")
    mae_ml, mae_base = float(primary.at[ml, "mae"]), float(primary.at[baseline, "mae"])
    ml_wins = bool(np.isfinite(mae_ml) and np.isfinite(mae_base) and mae_ml < mae_base)
    return {
        "moteur": "ML" if ml_wins else "B1",
        "mae_ml": mae_ml,
        "mae_b1": mae_base,
        "n": n_ml,
        "règle": SELECTION_RULE,
    }


def rolling_forecasters(model) -> dict[str, Callable]:
    from stockvisible.model import ml_forecaster

    ml = ml_forecaster(model)
    return {
        "B0 (glissant)": rolling_origin(b0_forecast, "B0 (glissant)"),
        "B1 (glissant)": rolling_origin(b1_forecast, "B1 (glissant)"),
        "ML (glissant)": lambda h, t: replace(ml(h, t), name="ML (glissant)"),
    }


def rolling_validation_table(dev, model) -> pd.DataFrame:
    forecasts = [fn(dev.train, dev.validation) for fn in rolling_forecasters(model).values()]
    return compare_baselines(dev.validation, forecasts, label="validation")


def masked_rolling_table(dev, model) -> pd.DataFrame:
    predictors = {name: forecast_predictor(fn) for name, fn in rolling_forecasters(model).items()}
    return run_masked_eval(dev.train, dev.validation, predictors)


# ---------------------------------------------------------------- CLI (dev uniquement)


def validation_table() -> pd.DataFrame:
    from stockvisible.splits import load_dev

    dev = load_dev()
    forecasts = [b0_forecast(dev.train, dev.validation), b1_forecast(dev.train, dev.validation)]
    return compare_baselines(dev.validation, forecasts, label="validation")


def masked_validation_table() -> pd.DataFrame:
    from stockvisible.splits import load_dev

    dev = load_dev()
    predictors = {"B0": forecast_predictor(b0_forecast), "B1": forecast_predictor(b1_forecast)}
    return run_masked_eval(dev.train, dev.validation, predictors)


def main() -> None:
    from stockvisible.model import fit_model
    from stockvisible.splits import load_dev

    fmt = {"float_format": lambda v: f"{v:.4f}", "index": False}
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("== M2 : origine fixe (fin du train) ==")
        print(validation_table().to_string(**fmt))
        print("\n== M3 : test masqué, origine fixe ==")
        print(masked_validation_table().to_string(**fmt))
        dev = load_dev()
        model = fit_model(dev.train)
        print(f"\n== M7 : glissant J+1 — ML entraîné sur {model.train_start} → {model.train_end}, "
              f"{model.n_train_rows} lignes horaires disponibles ==")
        table = rolling_validation_table(dev, model)
        print(table.to_string(**fmt))
        print("\n== M7 : test masqué, glissant ==")
        print(masked_rolling_table(dev, model).to_string(**fmt))
        print("\n== M7 : sélection ==")
        print(select_engine(table))


if __name__ == "__main__":
    main()
