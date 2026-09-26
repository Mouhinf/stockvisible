"""Incertitude du moteur gelé (M10) — intervalle empirique « split conformal », VALIDATION seulement.

Le moteur gelé (ML, médiane) ne donne pas d'intervalle et model.py est intouchable. On construit
donc autour de sa prévision glissante J+1 un intervalle [pred + q10, pred + q90], où q10 et q90 sont
les quantiles des résidus (réel − prévision) observés, par heure de la journée, sur les jours de
CALIBRATION (jours 1–8 de la validation). La couverture est mesurée sur des jours jamais vus en
calibration (jours 9–15). Une strate avec moins de `min_support` résidus n'a pas d'intervalle :
elle est déclarée « intervalle non calibré ». Le test réservé n'est jamais lu.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stockvisible.baselines import HOURS, HourlyForecast, hourly_matrix
from stockvisible.data import PROJECT_ROOT, SERIES_KEY
from stockvisible.evaluation import ReservedPeriodError

REPORT_PATH = PROJECT_ROOT / "logs" / "interval_calibration.json"
NOT_CALIBRATED = "intervalle non calibré"


@dataclass(frozen=True)
class IntervalSpec:
    """Fixé avant tout calcul (M10)."""

    q_low: float = 0.10
    q_high: float = 0.90
    calibration_days: int = 8
    min_support: int = 200

    @property
    def nominal(self) -> float:
        return self.q_high - self.q_low


INTERVAL_SPEC = IntervalSpec()


def residuals(forecast: HourlyForecast, targets: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par heure DISPONIBLE prévue : la vente observée y est la demande."""
    actual = hourly_matrix(targets.reset_index(drop=True), "hours_sale")
    available = hourly_matrix(targets.reset_index(drop=True), "hours_stock_status") == 0
    keep = available & ~np.isnan(forecast.pred)
    rows, hours = np.nonzero(keep)
    return pd.DataFrame(
        {
            "dt": forecast.keys["dt"].to_numpy()[rows],
            "hour": hours,
            "pred": forecast.pred[rows, hours],
            "actual": actual[rows, hours],
        }
    ).assign(residual=lambda d: d["actual"] - d["pred"])


def split_days(res: pd.DataFrame, reserved_start: str, spec: IntervalSpec = INTERVAL_SPEC):
    """Jours de calibration puis jours d'évaluation ; refuse toute date de la période test."""
    if len(res) and str(res["dt"].max()) >= reserved_start:
        raise ReservedPeriodError(f"résidus datés jusqu'au {res['dt'].max()} : période test atteinte")
    days = sorted(res["dt"].unique())
    cal_days = set(days[: spec.calibration_days])
    is_cal = res["dt"].isin(cal_days)
    return res[is_cal], res[~is_cal]


def calibrate(cal: pd.DataFrame, spec: IntervalSpec = INTERVAL_SPEC) -> pd.DataFrame:
    """Par heure : décalages [q_low, q_high] des résidus et support ; strate faible → NaN."""
    table = pd.DataFrame({"hour": np.arange(HOURS)})
    stats = cal.groupby("hour")["residual"].agg(
        support="size",
        low=lambda r: float(np.quantile(r, spec.q_low)),
        high=lambda r: float(np.quantile(r, spec.q_high)),
    )
    table = table.merge(stats, on="hour", how="left").fillna({"support": 0})
    table["support"] = table["support"].astype(int)
    table["calibrated"] = table["support"] >= spec.min_support
    table.loc[~table["calibrated"], ["low", "high"]] = np.nan
    return table


def interval(pred: np.ndarray, hours: np.ndarray, calib: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Bornes pour des prévisions données ; NaN là où la strate n'est pas calibrée. Borne basse >= 0."""
    by_hour = calib.set_index("hour")
    low = pred + by_hour["low"].to_numpy()[hours]
    high = pred + by_hour["high"].to_numpy()[hours]
    return np.maximum(low, 0.0), high


def coverage(evaluation: pd.DataFrame, calib: pd.DataFrame) -> pd.DataFrame:
    """Couverture empirique par heure, mesurée hors calibration."""
    low, high = interval(evaluation["pred"].to_numpy(), evaluation["hour"].to_numpy(), calib)
    inside = (evaluation["actual"].to_numpy() >= low) & (evaluation["actual"].to_numpy() <= high)
    frame = evaluation.assign(inside=inside, calibrated=~np.isnan(low))
    per_hour = frame.groupby("hour").agg(n_eval=("inside", "size"), coverage=("inside", "mean"))
    out = calib.merge(per_hour, on="hour", how="left").fillna({"n_eval": 0})
    out.loc[~out["calibrated"], "coverage"] = np.nan
    out["status"] = np.where(out["calibrated"], "calibré", NOT_CALIBRATED)
    return out


def global_status(cov: pd.DataFrame, spec: IntervalSpec = INTERVAL_SPEC) -> dict:
    ok = cov[cov["calibrated"] & (cov["n_eval"] > 0)]
    if ok.empty:
        return {"statut": NOT_CALIBRATED, "couverture": None, "nominal": spec.nominal}
    weights = ok["n_eval"].to_numpy()
    value = float(np.average(ok["coverage"].to_numpy(), weights=weights))
    return {
        "statut": "calibré" if len(ok) == len(cov) else "partiellement calibré",
        "couverture": value,
        "nominal": spec.nominal,
        "heures_non_calibrées": cov.loc[~cov["calibrated"], "hour"].astype(int).tolist(),
        "n_eval": int(weights.sum()),
    }


def build_report(dev, spec: IntervalSpec = INTERVAL_SPEC) -> dict:
    """Moteur gelé (entraîné sur train) → résidus glissants sur la validation → calibration → couverture."""
    from stockvisible.model import fit_model, ml_forecast
    from stockvisible.selection import check_freeze, load_freeze
    from stockvisible.splits import reserved_period_start

    check_freeze(load_freeze())
    start = reserved_period_start(dev)
    model = fit_model(dev.train)
    res = residuals(ml_forecast(model, dev.train, dev.validation), dev.validation)
    cal, ev = split_days(res, start, spec)
    calib = calibrate(cal, spec)
    cov = coverage(ev, calib)
    return {
        "horodatage_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "moteur": "ML gelé (M8), entraîné sur le train seul",
        "spec": asdict(spec) | {"nominal": spec.nominal},
        "jours_calibration": [str(cal["dt"].min()), str(cal["dt"].max())],
        "jours_évaluation": [str(ev["dt"].min()), str(ev["dt"].max())],
        "global": global_status(cov, spec),
        "par_heure": json.loads(cov.to_json(orient="records")),
        "séries": int(dev.validation[list(SERIES_KEY)].drop_duplicates().shape[0]),
    }


def load_report(path: Path = REPORT_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main() -> None:
    from stockvisible.splits import load_dev

    report = build_report(load_dev())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["global"], indent=2, ensure_ascii=False))
    print(f"écrit : {REPORT_PATH}")


if __name__ == "__main__":
    main()
