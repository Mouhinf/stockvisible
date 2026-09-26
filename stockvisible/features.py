"""Variables horaires du modèle ML — strictement passées (M7).

Une ligne = (série, jour D, heure h). Chaque variable de D n'utilise que :
- les jours < D pour tout ce qui est observé (ventes, disponibilité, remise) ;
- le calendrier de D (jour de semaine, heure, jour férié : connus d'avance) ;
- la hiérarchie statique de la série.
Une vente observée pendant une heure en rupture est censurée : elle vaut NaN dans les lags et la
médiane glissante (seul hours_stock_status fait foi). La remise est celle de D-1 : celle de D
n'est pas vérifiablement connue d'avance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockvisible.baselines import HOURS, hourly_matrix
from stockvisible.data import SERIES_KEY


@dataclass(frozen=True)
class FeatureSpec:
    """Fixé avant tout entraînement (M7)."""

    lags_days: tuple[int, ...] = (1, 7)
    rolling_days: int = 7
    rolling_min_obs: int = 3


FEATURE_SPEC = FeatureSpec()

HIERARCHY = (
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "city_id",
)
CATEGORICAL_FEATURES = ("day_of_week", *HIERARCHY)
NUMERIC_FEATURES = ("hour", "lag_d1", "lag_d7", "rolling_median_7", "discount_d1", "holiday_flag")
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
LABEL_COLUMNS = ("sale", "available")

# Colonnes dont la valeur au jour D n'est connue qu'à la fin de D (ou après).
OUTCOME_COLUMNS = (
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
)


class FeatureLeakError(AssertionError):
    """Une variable du jour D dépend d'une donnée du jour D ou d'après."""


def _shift_days(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def hourly_features(frame: pd.DataFrame, spec: FeatureSpec = FEATURE_SPEC) -> pd.DataFrame:
    """Variables + étiquette (sale, available) pour chaque (série, dt, heure) de frame.

    Sortie triée par (store_id, product_id, dt, hour). Les décalages se font sur le calendrier
    (D-1 = veille réelle), jamais sur la position : un jour manquant donne NaN.
    """
    keys = list(SERIES_KEY)
    f = frame.assign(_date=pd.to_datetime(frame["dt"], format="%Y-%m-%d"))
    f = f.sort_values([*keys, "_date"], kind="mergesort").reset_index(drop=True)
    if f.duplicated([*keys, "_date"]).any():
        raise ValueError("(série, dt) dupliqué : variables ambiguës")

    series = f[keys].drop_duplicates().reset_index(drop=True)
    s_idx = f[keys].merge(series.reset_index(), on=keys, how="left")["index"].to_numpy()
    start = f["_date"].min()
    d_idx = (f["_date"] - start).dt.days.to_numpy()
    shape = (len(series), int(d_idx.max()) + 1)

    sales = np.full((*shape, HOURS), np.nan)
    status = np.full((*shape, HOURS), np.nan)
    sales[s_idx, d_idx] = hourly_matrix(f, "hours_sale")
    status[s_idx, d_idx] = hourly_matrix(f, "hours_stock_status")
    observed = np.where(status == 0, sales, np.nan)  # censuré ou absent → NaN

    discount = np.full(shape, np.nan)
    discount[s_idx, d_idx] = f["discount"].to_numpy(dtype=float)

    lags = {k: _shift_days(observed, k) for k in spec.lags_days}
    window = np.stack([_shift_days(observed, k) for k in range(1, spec.rolling_days + 1)])
    n_obs = (~np.isnan(window)).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # fenêtres vides → NaN
        rolling = np.nanmedian(window, axis=0)
    rolling = np.where(n_obs >= spec.rolling_min_obs, rolling, np.nan)
    discount_d1 = _shift_days(discount, 1)

    rows = np.repeat(np.arange(len(f)), HOURS)
    hours = np.tile(np.arange(HOURS), len(f))
    si, di = s_idx[rows], d_idx[rows]
    out = pd.DataFrame(
        {
            **{k: f[k].to_numpy()[rows] for k in keys},
            "dt": f["dt"].to_numpy()[rows],
            "hour": hours,
            "lag_d1": lags[1][si, di, hours],
            "lag_d7": lags[7][si, di, hours],
            "rolling_median_7": rolling[si, di, hours],
            "discount_d1": discount_d1[si, di],
            "holiday_flag": f["holiday_flag"].to_numpy(dtype=float)[rows],
            "day_of_week": f["_date"].dt.dayofweek.to_numpy()[rows],
            **{c: f[c].to_numpy()[rows] for c in HIERARCHY},
            "sale": sales[si, di, hours],
            "available": status[si, di, hours] == 0,
        }
    )
    return out


def perturb_outcomes(frame: pd.DataFrame, from_date: str, seed: int = 0) -> pd.DataFrame:
    """Copie de frame où toutes les colonnes d'issue des jours >= from_date sont modifiées."""
    rng = np.random.default_rng(seed)
    out = frame.copy(deep=True)
    hit = (out["dt"] >= from_date).to_numpy()
    for col in ("hours_sale", "hours_stock_status"):
        out[col] = out[col].astype(object)
    for i in np.flatnonzero(hit):
        sale = np.asarray(out.at[i, "hours_sale"], dtype=float)
        out.at[i, "hours_sale"] = list(sale * 1.7 + 0.3 + rng.random(HOURS))
        out.at[i, "hours_stock_status"] = [1 - int(s) for s in out.at[i, "hours_stock_status"]]
    out.loc[hit, "sale_amount"] = out.loc[hit, "hours_sale"].map(sum)
    out.loc[hit, "discount"] = 1.0 - 0.5 * out.loc[hit, "discount"].to_numpy(dtype=float)
    for col in ("activity_flag", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"):
        if col in out.columns:
            out.loc[hit, col] = out.loc[hit, col].to_numpy(dtype=float) + 1.0
    return out


def check_features_past_only(
    frame: pd.DataFrame, builder=hourly_features, cut_dates: list[str] | None = None
) -> None:
    """Pour chaque date de coupure C : modifier toutes les issues des jours >= C ne doit changer
    aucune variable des jours <= C. Lève FeatureLeakError sinon."""
    frame = frame.reset_index(drop=True)
    dates = sorted(frame["dt"].unique())
    cut_dates = cut_dates or [dates[len(dates) // 3], dates[len(dates) // 2], dates[-1]]
    reference = builder(frame)
    for cut in cut_dates:
        perturbed = builder(perturb_outcomes(frame, cut))
        mask = (reference["dt"] <= cut).to_numpy()
        for col in FEATURES:
            a = reference.loc[mask, col].to_numpy(dtype=float)
            b = perturbed.loc[mask, col].to_numpy(dtype=float)
            if not np.array_equal(a, b, equal_nan=True):
                raise FeatureLeakError(f"{col} des jours <= {cut} dépend des issues des jours >= {cut}")
