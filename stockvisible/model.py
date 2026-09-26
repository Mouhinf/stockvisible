"""Moteur ML (M7) — HistGradientBoostingRegressor quantile 0,5, entraîné sur TRAIN uniquement.

Entraînement : lignes horaires du train aux heures DISPONIBLES (la vente d'une heure en rupture
est censurée, ce n'est pas la demande). Réglages fixés a priori, aucun ajustement fin.
Prévision glissante à J+1 : chaque jour D est prévu avec les données observées jusqu'à D-1,
le modèle n'étant jamais réentraîné. Le modèle n'est PAS figé ici : il sera choisi (ou non)
sur la validation, puis figé dans un milestone ultérieur.

Bruit de départage (M7b, décision utilisateur) : 71 % des cibles valent exactement 0, la valeur
initiale du boosting. HistGradientBoosting quantile reste alors bloqué à 0 (2 feuilles nulles
par arbre, reproduit sur un jouet). Un bruit uniforme [0, 1e-6), seed fixe, ajouté à la cible
d'ENTRAÎNEMENT uniquement, casse ces égalités sans changer la médiane de plus de 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from stockvisible.baselines import HOURS, HourlyForecast, check_no_future
from stockvisible.data import SERIES_KEY
from stockvisible.features import CATEGORICAL_FEATURES, FEATURES, hourly_features

MAX_CATEGORIES = 255  # limite de HistGradientBoosting pour une variable catégorielle


@dataclass(frozen=True)
class ModelSpec:
    """Seuls écarts aux valeurs par défaut de scikit-learn (fixés a priori, M7)."""

    loss: str = "quantile"
    quantile: float = 0.5
    early_stopping: bool = False  # le mode "auto" découperait le train au hasard
    random_state: int = 42
    target_jitter: float = 1e-6  # départage des cibles égales (M7b) ; pas un paramètre de HGB


MODEL_SPEC = ModelSpec()


class TemporalLeakError(ValueError):
    """Le modèle a été entraîné sur des dates >= aux dates qu'on lui demande de prévoir."""


@dataclass(frozen=True)
class FittedModel:
    estimator: HistGradientBoostingRegressor
    encoders: dict[str, dict[float, int]]
    train_start: str
    train_end: str
    n_train_rows: int
    sklearn_version: str


def fit_encoders(rows: pd.DataFrame) -> dict[str, dict[float, int]]:
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        values = np.sort(rows[col].dropna().unique())
        if len(values) > MAX_CATEGORIES:
            raise ValueError(f"{col} : {len(values)} catégories > {MAX_CATEGORIES}")
        encoders[col] = {float(v): i for i, v in enumerate(values)}
    return encoders


def design_matrix(feats: pd.DataFrame, encoders: dict[str, dict[float, int]]) -> np.ndarray:
    """Colonnes dans l'ordre de FEATURES ; catégorie inconnue du train → NaN (valeur manquante)."""
    cols = []
    for col in FEATURES:
        values = feats[col].astype(float)
        if col in encoders:
            values = values.map(encoders[col])
        cols.append(values.to_numpy(dtype=float))
    return np.column_stack(cols)


def fit_model(train: pd.DataFrame, spec: ModelSpec = MODEL_SPEC) -> FittedModel:
    feats = hourly_features(train)
    rows = feats[feats["available"] & feats["sale"].notna()]
    encoders = fit_encoders(rows)
    estimator = HistGradientBoostingRegressor(
        loss=spec.loss,
        quantile=spec.quantile,
        early_stopping=spec.early_stopping,
        random_state=spec.random_state,
        categorical_features=[c in CATEGORICAL_FEATURES for c in FEATURES],
    )
    y = rows["sale"].to_numpy(dtype=float)
    rng = np.random.default_rng(spec.random_state)
    y = y + rng.uniform(0.0, spec.target_jitter, size=len(y)) if spec.target_jitter > 0 else y
    estimator.fit(design_matrix(rows, encoders), y)
    return FittedModel(
        estimator=estimator,
        encoders=encoders,
        train_start=str(train["dt"].min()),
        train_end=str(train["dt"].max()),
        n_train_rows=len(rows),
        sklearn_version=sklearn.__version__,
    )


def ml_forecast(model: FittedModel, history: pd.DataFrame, targets: pd.DataFrame) -> HourlyForecast:
    """Prévision glissante à J+1 de chaque ligne de targets (ordre conservé)."""
    first_target = str(targets["dt"].min())
    if model.train_end >= first_target:
        raise TemporalLeakError(
            f"modèle entraîné jusqu'au {model.train_end}, cible à partir du {first_target}"
        )
    check_no_future(history, targets)
    keys = [*SERIES_KEY, "dt"]
    feats = hourly_features(pd.concat([history, targets], ignore_index=True))
    wanted = targets.loc[:, keys].reset_index(drop=True)
    grid = wanted.assign(_row=np.arange(len(wanted))).merge(feats, on=keys, how="left")
    grid = grid.sort_values(["_row", "hour"], kind="mergesort")
    pred = np.clip(model.estimator.predict(design_matrix(grid, model.encoders)), 0.0, None)
    shape = (len(wanted), HOURS)
    return HourlyForecast(
        name="ML",
        keys=wanted,
        pred=pred.reshape(shape),
        n_obs=np.full(shape, -1),
        abstained=np.zeros(shape, dtype=bool),
    )


def ml_forecaster(model: FittedModel):
    """Adaptateur (history, targets) -> HourlyForecast, même signature que b0/b1_forecast."""

    def forecast(history: pd.DataFrame, targets: pd.DataFrame) -> HourlyForecast:
        return ml_forecast(model, history, targets)

    return forecast
