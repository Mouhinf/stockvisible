"""Métriques et évaluation — F3 VERIFY. Protocole fixé avant tout résultat (M2).

Périmètre principal : heures déclarées disponibles dans la cible (statut 0), où la vente
observée est la demande non censurée. Les modèles sont comparés sur les mêmes cellules.
Périmètre secondaire, étiqueté : toutes les heures, avec une cible censurée pendant les ruptures.
Biais = moyenne(prédiction - réel) : négatif = sous-prévision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockvisible.baselines import (
    HourlyForecast,
    b0_forecast,
    b1_forecast,
    hourly_matrix,
)


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


def validation_table() -> pd.DataFrame:
    from stockvisible.data import load_raw
    from stockvisible.splits import chronological_split

    splits = chronological_split(load_raw("train"))
    forecasts = [
        b0_forecast(splits.train, splits.validation),
        b1_forecast(splits.train, splits.validation),
    ]
    return compare_baselines(splits.validation, forecasts, label="validation")


def main() -> None:
    table = validation_table()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
