"""Baselines horaires — F3 VERIFY. Fonctions pures : aucune I/O, aucune mutation, déterministes.

Créneau = (série, jour de semaine, heure). Jours comparables = même jour de semaine dans
l'historique fourni. B0 : médiane de toutes les ventes observées (ruptures incluses, donc
biaisée vers le bas). B1 : médiane restreinte aux heures déclarées disponibles
(hours_stock_status == 0), abstention si moins de `min_obs` observations comparables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockvisible.data import SERIES_KEY

HOURS = 24
B1_MIN_OBS = 3  # fixé a priori (M2), jamais ajusté sur validation ni test


@dataclass(frozen=True)
class HourlyForecast:
    """pred[i, h] = NaN et abstained[i, h] = True quand le modèle s'abstient."""

    name: str
    keys: pd.DataFrame
    pred: np.ndarray
    n_obs: np.ndarray
    abstained: np.ndarray

    @property
    def coverage(self) -> float:
        return float(1 - self.abstained.mean()) if self.abstained.size else 0.0


def hourly_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    if len(df) == 0:
        return np.empty((0, HOURS))
    return np.stack([np.asarray(v, dtype=float) for v in df[col]])


def _weekday(df: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(df["dt"], format="%Y-%m-%d").dt.dayofweek.to_numpy()


def _check_no_future(history: pd.DataFrame, targets: pd.DataFrame) -> None:
    keys = list(SERIES_KEY)
    last = history.assign(_d=pd.to_datetime(history["dt"])).groupby(keys)["_d"].max()
    first = targets.assign(_d=pd.to_datetime(targets["dt"])).groupby(keys)["_d"].min()
    both = pd.concat([last.rename("last"), first.rename("first")], axis=1, join="inner")
    leaked = both[both["last"] >= both["first"]]
    if len(leaked):
        raise ValueError(f"fuite temporelle : historique >= cible pour {leaked.index.tolist()[:5]}")


def _slot_median(
    history: pd.DataFrame, targets: pd.DataFrame, available_only: bool, min_obs: int, name: str
) -> HourlyForecast:
    _check_no_future(history, targets)
    keys = list(SERIES_KEY)

    sales = hourly_matrix(history, "hours_sale")
    if available_only:
        sales = np.where(hourly_matrix(history, "hours_stock_status") == 0, sales, np.nan)
    long = pd.DataFrame(
        {
            **{k: np.repeat(history[k].to_numpy(), HOURS) for k in keys},
            "weekday": np.repeat(_weekday(history), HOURS),
            "hour": np.tile(np.arange(HOURS), len(history)),
            "sale": sales.ravel(),
        }
    )
    stats = long.groupby([*keys, "weekday", "hour"])["sale"].agg(["median", "count"]).reset_index()

    wanted = pd.DataFrame(
        {
            **{k: np.repeat(targets[k].to_numpy(), HOURS) for k in keys},
            "weekday": np.repeat(_weekday(targets), HOURS),
            "hour": np.tile(np.arange(HOURS), len(targets)),
        }
    )
    got = wanted.merge(stats, on=[*keys, "weekday", "hour"], how="left", validate="many_to_one")
    n_obs = got["count"].fillna(0).to_numpy().reshape(len(targets), HOURS).astype(int)
    abstained = n_obs < min_obs
    pred = np.where(abstained, np.nan, got["median"].to_numpy().reshape(len(targets), HOURS))
    return HourlyForecast(
        name=name,
        keys=targets.loc[:, [*keys, "dt"]].reset_index(drop=True),
        pred=pred,
        n_obs=n_obs,
        abstained=abstained,
    )


def b0_forecast(history: pd.DataFrame, targets: pd.DataFrame) -> HourlyForecast:
    """B0 : médiane des ventes aux mêmes créneaux, ruptures ignorées (ventes censurées incluses)."""
    return _slot_median(history, targets, available_only=False, min_obs=1, name="B0")


def b1_forecast(
    history: pd.DataFrame, targets: pd.DataFrame, min_obs: int = B1_MIN_OBS
) -> HourlyForecast:
    """B1 : même calcul restreint aux heures disponibles ; abstention si < min_obs observations."""
    if min_obs < 1:
        raise ValueError("min_obs doit être >= 1")
    return _slot_median(history, targets, available_only=True, min_obs=min_obs, name="B1")
