"""IA responsable (M10) : abstention visible, intervalle empirique, jamais le test réservé."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockvisible.baselines import b1_forecast
from stockvisible.evaluation import ReservedPeriodError
from stockvisible.uncertainty import (
    INTERVAL_SPEC,
    NOT_CALIBRATED,
    REPORT_PATH,
    calibrate,
    coverage,
    global_status,
    interval,
    split_days,
)
from ui.components import abstention_summary, interval_summary

APP = Path(__file__).resolve().parents[1] / "app.py"
RESERVED = "2024-06-11"


def _day(date, sale=1.0, status=0):
    return {"store_id": 1, "product_id": 1, "dt": date,
            "hours_sale": [sale] * 24, "hours_stock_status": [status] * 24}


def _residuals(n_per_hour, days=15, seed=0):
    rng = np.random.default_rng(seed)
    dates = [str((pd.Timestamp("2024-05-27") + pd.Timedelta(days=d)).date()) for d in range(days)]
    rows = [(d, h) for d in dates for h in range(24) for _ in range(n_per_hour)]
    res = pd.DataFrame(rows, columns=["dt", "hour"]).assign(pred=5.0)  # réel toujours >= 0
    res["residual"] = rng.normal(0, 1, len(res))
    res["actual"] = res["pred"] + res["residual"]
    return res


def test_interval_spec_is_frozen():
    assert asdict(INTERVAL_SPEC) == {"q_low": 0.1, "q_high": 0.9, "calibration_days": 8, "min_support": 200}


# ---------------------------------------------------------------- abstention


def test_synthetic_series_without_enough_history_triggers_abstention():
    """SYNTHÉTIQUE : 2 jours d'historique (lundi, mardi), cible un lundi → < 3 observations."""
    history = pd.DataFrame([_day("2024-01-01"), _day("2024-01-02")])
    target = pd.DataFrame([_day("2024-01-08", sale=0.0)])
    summary = abstention_summary(b1_forecast(history, target))
    assert summary["niveau"] == "totale"
    assert "aucune période comparable" in summary["message"]


def test_partial_abstention_is_reported_not_filled():
    history = pd.DataFrame([_day(d) for d in ("2024-01-01", "2024-01-08", "2024-01-15")])
    target = pd.DataFrame([_day("2024-01-22"), _day("2024-01-23")])  # lundi couvert, mardi non
    f = b1_forecast(history, target)
    summary = abstention_summary(f)
    assert summary["niveau"] == "partielle" and "24 heures sur 48" in summary["message"]
    assert np.isnan(f.pred[1]).all()  # jamais remplacé par 0


def test_no_abstention_when_comparable_periods_exist():
    history = pd.DataFrame([_day(d) for d in ("2024-01-01", "2024-01-08", "2024-01-15")])
    summary = abstention_summary(b1_forecast(history, pd.DataFrame([_day("2024-01-22")])))
    assert summary == {"niveau": "aucune", "message": ""}


# ---------------------------------------------------------------- intervalle


def test_insufficient_support_is_declared_not_calibrated():
    cal, ev = split_days(_residuals(n_per_hour=2), RESERVED)  # 8 jours × 2 = 16 < 200
    cov = coverage(ev, calibrate(cal))
    assert not cov["calibrated"].any() and cov["low"].isna().all()
    assert set(cov["status"]) == {NOT_CALIBRATED}
    assert global_status(cov)["statut"] == NOT_CALIBRATED
    report = {"global": global_status(cov)}
    assert interval_summary(report)["niveau"] == "non_calibré"
    assert NOT_CALIBRATED in interval_summary(None)["message"]


def test_partially_supported_hours_are_listed():
    res = _residuals(n_per_hour=30)
    res = res[~((res["hour"] == 3) & (res.index % 10 != 0))]  # heure 3 : 3 résidus/jour
    cal, ev = split_days(res, RESERVED)
    status = global_status(coverage(ev, calibrate(cal)))
    assert status["statut"] == "partiellement calibré" and status["heures_non_calibrées"] == [3]


def test_calibrated_interval_covers_nominal_on_unseen_days():
    cal, ev = split_days(_residuals(n_per_hour=40), RESERVED)
    assert set(cal["dt"]).isdisjoint(ev["dt"]) and cal["dt"].nunique() == 8
    calib = calibrate(cal)
    assert calib["calibrated"].all()
    assert calib["low"].to_numpy() == pytest.approx(np.full(24, -1.2816), abs=0.2)
    assert global_status(coverage(ev, calib))["couverture"] == pytest.approx(0.8, abs=0.03)


def test_lower_bound_never_negative():
    calib = pd.DataFrame({"hour": range(24), "low": -5.0, "high": 1.0, "support": 500, "calibrated": True})
    low, _ = interval(np.zeros(24), np.arange(24), calib)
    assert (low >= 0).all()


def test_split_refuses_reserved_period_dates():
    res = _residuals(n_per_hour=1, days=16)  # 2024-05-27 + 15 jours = 2024-06-11
    with pytest.raises(ReservedPeriodError):
        split_days(res, RESERVED)


# ---------------------------------------------------------------- rapport réel et écran


@pytest.mark.skipif(not REPORT_PATH.exists(), reason="lancer `python -m stockvisible.uncertainty`")
def test_real_report_uses_validation_only_and_is_consistent():
    r = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert r["jours_calibration"] == ["2024-05-27", "2024-06-03"]
    assert r["jours_évaluation"] == ["2024-06-04", "2024-06-10"] and r["jours_évaluation"][1] < RESERVED
    hours = pd.DataFrame(r["par_heure"])
    ok = hours[hours["calibrated"]]
    weighted = float(np.average(ok["coverage"], weights=ok["n_eval"]))
    assert r["global"]["couverture"] == pytest.approx(weighted)
    assert r["global"]["n_eval"] == int(ok["n_eval"].sum())


def test_app_shows_abstention_and_interval_status(dev):
    from streamlit.testing.v1 import AppTest

    from ui.components import series_frames, series_keys

    key = next(
        k for k in series_keys(dev)
        if b1_forecast(*series_frames(dev, k)).abstained.any()
    )
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Couverture empirique de l'intervalle 80 %" for m in at.metric)  # Vérifier
    at.switch_page("ui/pages/comprendre.py").run()
    at.selectbox[0].set_value(key).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Abstention" in w.value for w in at.warning) or any("Abstention" in e.value for e in at.error)
