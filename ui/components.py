"""Préparation des données de l'écran. Fonctions pures, sans Streamlit, testables."""

from __future__ import annotations

import pandas as pd

from stockvisible.baselines import HourlyForecast
from stockvisible.data import SERIES_KEY
from stockvisible.evaluation import compare_baselines
from stockvisible.splits import DevSplits
from stockvisible.validation import STOCK_WINDOW_HOURS

SeriesKey = tuple[int, int]

UI_SCOPES = ("heures disponibles, cellules communes", "heures disponibles, toute la couverture")
TABLE_COLUMNS = {
    "périmètre": "Périmètre",
    "modèle": "Modèle",
    "couverture": "Couverture (%)",
    "n": "Heures évaluées",
    "mae": "MAE",
    "bias": "Biais",
}


def series_keys(dev: DevSplits) -> list[SeriesKey]:
    keys = dev.train.loc[:, list(SERIES_KEY)].drop_duplicates().sort_values(list(SERIES_KEY))
    return [(int(s), int(p)) for s, p in keys.to_numpy()]


def series_label(key: SeriesKey) -> str:
    return f"magasin {key[0]} · produit {key[1]}"


def series_frames(dev: DevSplits, key: SeriesKey) -> tuple[pd.DataFrame, pd.DataFrame]:
    store, product = key

    def pick(df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["store_id"] == store) & (df["product_id"] == product)
        return df.loc[mask].sort_values("dt", kind="mergesort").reset_index(drop=True)

    return pick(dev.train), pick(dev.validation)


def series_summary(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float]:
    frame = pd.concat([train, validation])
    return {
        "jours": len(frame),
        "part_heures_rupture_6_22": float(frame["stock_hour6_22_cnt"].sum() / (STOCK_WINDOW_HOURS * len(frame))),
        "jours_avec_rupture_6_22": int((frame["stock_hour6_22_cnt"] > 0).sum()),
    }


def comparison_table(validation: pd.DataFrame, forecasts: list[HourlyForecast]) -> pd.DataFrame:
    """Périmètres « heures disponibles » uniquement : c'est là que la vente observée est la demande."""
    table = compare_baselines(validation, forecasts, label="validation")
    table = table[table["périmètre"].isin(UI_SCOPES)].drop(columns="split")
    table["couverture"] = 100 * table["couverture"]
    return table.rename(columns=TABLE_COLUMNS).reset_index(drop=True)
