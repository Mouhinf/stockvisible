"""Préparation des données de l'écran. Fonctions pures, sans Streamlit, testables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockvisible.allocation import Basket, Product
from stockvisible.baselines import HourlyForecast, b1_forecast, hourly_matrix
from stockvisible.data import SERIES_KEY
from stockvisible.evaluation import compare_baselines
from stockvisible.splits import DevSplits
from stockvisible.validation import STOCK_WINDOW, STOCK_WINDOW_HOURS

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


# ---------------------------------------------------------------- F4 DECIDE

DEFAULT_UNIT_COST = 1.0
DEFAULT_LOT_SIZE = 0.5
DEFAULT_STOCK = 0.0
HYPOTHESES_COLUMNS = ["Série", "Coût unitaire (hypothèse)", "Taille de lot", "Stock actuel (hypothèse)"]


def daily_demand_scenarios(dev: DevSplits, keys: list[SeriesKey]) -> tuple[list[str], np.ndarray]:
    """Un scénario = un jour de validation. Demande estimée = ventes observées aux heures
    disponibles + B1 aux heures en rupture de la fenêtre 6–22 (B1 absent → vente observée).
    Les heures de nuit ne sont jamais traitées comme demande perdue."""
    lo, hi = STOCK_WINDOW.start, STOCK_WINDOW.stop
    columns, dates = [], None
    for key in keys:
        train, validation = series_frames(dev, key)
        sales = hourly_matrix(validation, "hours_sale")
        stockout = hourly_matrix(validation, "hours_stock_status") == 1
        stockout[:, :lo] = False
        stockout[:, hi:] = False
        b1 = b1_forecast(train, validation).pred
        fill = stockout & ~np.isnan(b1)
        columns.append(np.where(fill, b1, sales).sum(axis=1))
        if dates is None:
            dates = validation["dt"].tolist()
        elif validation["dt"].tolist() != dates:
            raise ValueError("les séries n'ont pas les mêmes jours de validation")
    return dates, np.column_stack(columns)


def default_hypotheses(keys: list[SeriesKey]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Série": [series_label(k) for k in keys],
            "Coût unitaire (hypothèse)": [DEFAULT_UNIT_COST] * len(keys),
            "Taille de lot": [DEFAULT_LOT_SIZE] * len(keys),
            "Stock actuel (hypothèse)": [DEFAULT_STOCK] * len(keys),
        }
    )


def products_from_hypotheses(table: pd.DataFrame) -> list[Product]:
    products = []
    for row in table.itertuples(index=False):
        _, cost, lot, stock = row
        products.append(
            Product(
                name=row[0],
                unit_cost=None if pd.isna(cost) else float(cost),
                lot_size=float(lot),
                stock=0.0 if pd.isna(stock) else float(stock),
            )
        )
    return products


def budget_upper_bound(products: list[Product], scenarios: np.ndarray) -> float:
    """Budget au-delà duquel aucun achat supplémentaire ne peut couvrir plus de demande."""
    total = 0.0
    for j, p in enumerate(products):
        if p.unit_cost is None or np.isnan(p.unit_cost):
            continue
        need = max(float(scenarios[:, j].max()) - p.stock, 0.0)
        total += np.ceil(need / p.lot_size - 1e-12) * p.lot_size * p.unit_cost
    return float(total)


def basket_table(basket: Basket) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Série": basket.names,
            "Lots achetés": basket.lots,
            "Quantité (unités normalisées)": basket.quantities,
        }
    )
