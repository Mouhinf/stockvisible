"""B0 / B1 et métriques — cas SYNTHÉTIQUES construits à la main."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockvisible.baselines import B1_MIN_OBS, b0_forecast, b1_forecast
from stockvisible.evaluation import compare_baselines, error_stats

MONDAY = pd.Timestamp("2024-01-01")  # lundi


def _day(date, sales, status, store=1, product=1) -> dict:
    return {
        "store_id": store,
        "product_id": product,
        "dt": str(pd.Timestamp(date).date()),
        "hours_sale": list(map(float, sales)),
        "hours_stock_status": list(map(int, status)),
    }


def _hours(value_at: dict[int, float], default: float = 0.0) -> list[float]:
    out = [default] * 24
    for h, v in value_at.items():
        out[h] = v
    return out


def _mondays(n: int, start: pd.Timestamp = MONDAY) -> list[pd.Timestamp]:
    return [start + pd.Timedelta(weeks=i) for i in range(n)]


def _history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_b0_is_median_of_same_weekday_slot():
    hist = _history([_day(d, _hours({10: v}), [0] * 24) for d, v in zip(_mondays(3), (1, 2, 10))])
    hist = pd.concat([hist, _history([_day(MONDAY + pd.Timedelta(days=1), _hours({10: 99}), [0] * 24)])])
    target = _history([_day(MONDAY + pd.Timedelta(weeks=3), [0] * 24, [0] * 24)])
    f = b0_forecast(hist, target)
    assert f.pred[0, 10] == 2.0  # le mardi à 99 n'est pas comparable
    assert f.n_obs[0, 10] == 3 and not f.abstained.any()


def test_b0_ignores_stockouts_b1_excludes_them():
    status = _hours({10: 1}, 0)
    rows = [_day(d, _hours({10: 0.0}), status) for d in _mondays(3)]
    rows += [_day(d, _hours({10: 4.0}), [0] * 24) for d in _mondays(3, MONDAY + pd.Timedelta(weeks=3))]
    hist = _history(rows)
    target = _history([_day(MONDAY + pd.Timedelta(weeks=6), [0] * 24, [0] * 24)])
    b0, b1 = b0_forecast(hist, target), b1_forecast(hist, target)
    assert b0.pred[0, 10] == 2.0  # médiane de [0,0,0,4,4,4] : les zéros censurés tirent vers le bas
    assert b1.pred[0, 10] == 4.0 and b1.n_obs[0, 10] == 3


def test_b1_abstains_without_comparable_period():
    """Cas SYNTHÉTIQUE : l'historique ne contient que des mardis, la cible est un lundi."""
    tuesdays = _mondays(5, MONDAY + pd.Timedelta(days=1))
    hist = _history([_day(d, [1.0] * 24, [0] * 24) for d in tuesdays])
    target = _history([_day(MONDAY + pd.Timedelta(weeks=5), [0] * 24, [0] * 24)])
    b1 = b1_forecast(hist, target)
    assert b1.abstained.all() and np.isnan(b1.pred).all()
    assert (b1.n_obs == 0).all() and b1.coverage == 0.0
    assert b0_forecast(hist, target).abstained.all()  # B0 n'invente rien non plus


def test_b1_abstains_when_comparable_days_are_all_stocked_out():
    status = _hours({10: 1}, 0)
    hist = _history([_day(d, _hours({10: 0.0}, 1.0), status) for d in _mondays(4)])
    target = _history([_day(MONDAY + pd.Timedelta(weeks=4), [0] * 24, [0] * 24)])
    b0, b1 = b0_forecast(hist, target), b1_forecast(hist, target)
    assert b1.abstained[0, 10] and np.isnan(b1.pred[0, 10])
    assert not b1.abstained[0, 9] and b1.pred[0, 9] == 1.0
    assert b0.pred[0, 10] == 0.0  # B0 prend la vente censurée pour la demande


@pytest.mark.parametrize("n_available, abstains", [(B1_MIN_OBS - 1, True), (B1_MIN_OBS, False)])
def test_b1_min_obs_threshold(n_available, abstains):
    days = _mondays(B1_MIN_OBS + 2)
    rows = [
        _day(d, [1.0] * 24, [0] * 24 if i < n_available else [1] * 24) for i, d in enumerate(days)
    ]
    target = _history([_day(days[-1] + pd.Timedelta(weeks=1), [0] * 24, [0] * 24)])
    b1 = b1_forecast(_history(rows), target)
    assert bool(b1.abstained.all()) is abstains
    with pytest.raises(ValueError):
        b1_forecast(_history(rows), target, min_obs=0)


def test_zero_sale_on_available_hour_is_an_observation_not_a_stockout():
    hist = _history([_day(d, [0.0] * 24, [0] * 24) for d in _mondays(3)])
    target = _history([_day(MONDAY + pd.Timedelta(weeks=3), [0] * 24, [0] * 24)])
    b1 = b1_forecast(hist, target)
    assert not b1.abstained.any() and (b1.pred == 0).all()


def test_series_are_not_mixed():
    hist = _history(
        [_day(d, [1.0] * 24, [0] * 24, store=1) for d in _mondays(3)]
        + [_day(d, [9.0] * 24, [0] * 24, store=2) for d in _mondays(3)]
    )
    target = _history([_day(MONDAY + pd.Timedelta(weeks=3), [0] * 24, [0] * 24, store=s) for s in (2, 1)])
    b1 = b1_forecast(hist, target)
    assert (b1.pred[0] == 9.0).all() and (b1.pred[1] == 1.0).all()
    assert b1.keys["store_id"].tolist() == [2, 1]


def test_future_history_is_refused():
    hist = _history([_day(d, [1.0] * 24, [0] * 24) for d in _mondays(4)])
    target = _history([_day(_mondays(4)[2], [0] * 24, [0] * 24)])
    for fn in (b0_forecast, b1_forecast):
        with pytest.raises(ValueError, match="fuite"):
            fn(hist, target)


def test_baselines_are_pure_and_order_independent():
    rng = np.random.default_rng(0)
    hist = _history([
        _day(MONDAY + pd.Timedelta(days=i), rng.random(24).round(2), (rng.random(24) < 0.3).astype(int))
        for i in range(28)
    ])
    target = _history([_day(MONDAY + pd.Timedelta(days=28 + i), [0] * 24, [0] * 24) for i in range(7)])
    before = hist.copy(deep=True)
    for fn in (b0_forecast, b1_forecast):
        a = fn(hist, target)
        b = fn(hist.sample(frac=1, random_state=3), target)
        np.testing.assert_array_equal(a.pred, b.pred)
        np.testing.assert_array_equal(a.abstained, b.abstained)
    pd.testing.assert_frame_equal(hist, before)


# ---------------------------------------------------------------- métriques


def test_error_stats_mae_and_bias():
    pred = np.array([[1.0, 2.0, np.nan, 5.0]])
    actual = np.array([[2.0, 2.0, 3.0, 1.0]])
    mask = np.array([[True, True, True, False]])
    assert error_stats(pred, actual, mask) == {"n": 2, "mae": 0.5, "bias": -0.5}
    assert error_stats(pred, actual, np.zeros_like(mask))["n"] == 0


def test_compare_uses_common_cells_and_available_hours():
    hist = _history(
        [_day(d, _hours({10: 2.0}), _hours({10: 1}) if i == 0 else [0] * 24) for i, d in enumerate(_mondays(3))]
    )
    target = _history([_day(MONDAY + pd.Timedelta(weeks=3), _hours({10: 2.0}), _hours({11: 1}))])
    b0, b1 = b0_forecast(hist, target), b1_forecast(hist, target)
    assert b1.abstained[0, 10]  # 2 observations disponibles < 3
    table = compare_baselines(target, [b0, b1], label="validation")
    common = table[table["périmètre"] == "heures disponibles, cellules communes"]
    assert common["n"].tolist() == [22, 22]  # 24 - heure 10 (abstention B1) - heure 11 (rupture cible)
    assert set(table["split"]) == {"validation"}
