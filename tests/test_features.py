"""Variables ML strictement passées — cas SYNTHÉTIQUES calculés à la main, anti-fuite, RÉEL dev."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from stockvisible.features import (
    CATEGORICAL_FEATURES,
    FEATURE_SPEC,
    FEATURES,
    FeatureLeakError,
    check_features_past_only,
    hourly_features,
)
from tests._synthetic import make_valid_frame

HIER = {
    "management_group_id": 1,
    "first_category_id": 2,
    "second_category_id": 3,
    "third_category_id": 4,
    "city_id": 5,
}


def _series(n_days=12, store=1, product=1, stockouts=(), skip_days=(), base=0.0):
    """Vente (d, h) = base + 100·d + h : chaque valeur est identifiable. SYNTHÉTIQUE."""
    rows = []
    for d in range(n_days):
        if d in skip_days:
            continue
        status = [1 if (d, h) in stockouts else 0 for h in range(24)]
        sale = [base + 100.0 * d + h for h in range(24)]
        rows.append(
            {
                "store_id": store,
                "product_id": product,
                "dt": str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).date()),
                "hours_sale": sale,
                "hours_stock_status": status,
                "sale_amount": float(sum(sale)),
                "discount": 0.5 + d / 100,
                "holiday_flag": d % 2,
                "activity_flag": 0,
                **HIER,
            }
        )
    return pd.DataFrame(rows)


def _at(feats, dt_index, hour, col, store=1):
    dt = str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=dt_index)).date())
    row = feats[(feats["store_id"] == store) & (feats["dt"] == dt) & (feats["hour"] == hour)]
    assert len(row) == 1
    return row[col].iloc[0]


def test_spec_and_features_are_frozen():
    assert asdict(FEATURE_SPEC) == {"lags_days": (1, 7), "rolling_days": 7, "rolling_min_obs": 3}
    assert FEATURES == (
        "hour", "lag_d1", "lag_d7", "rolling_median_7", "discount_d1", "holiday_flag",
        "day_of_week", "management_group_id", "first_category_id", "second_category_id",
        "third_category_id", "product_id", "city_id",
    )  # fmt: skip
    assert "store_id" not in FEATURES and "discount" not in FEATURES


def test_one_row_per_series_day_hour_sorted():
    frame = pd.concat([_series(store=2), _series(store=1)]).sample(frac=1, random_state=0)
    feats = hourly_features(frame)
    assert len(feats) == len(frame) * 24
    ordered = feats.sort_values(["store_id", "product_id", "dt", "hour"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(feats, ordered)


def test_calendar_features():
    feats = hourly_features(_series())
    assert _at(feats, 0, 13, "day_of_week") == 0  # 2024-01-01 = lundi
    assert _at(feats, 6, 13, "day_of_week") == 6
    assert _at(feats, 3, 13, "hour") == 13


def test_lags_by_hand():
    feats = hourly_features(_series())
    assert _at(feats, 8, 5, "lag_d1") == 705.0
    assert _at(feats, 8, 5, "lag_d7") == 105.0
    assert np.isnan(_at(feats, 0, 5, "lag_d1")) and np.isnan(_at(feats, 6, 5, "lag_d7"))


def test_rolling_median_uses_previous_7_days_with_min_3_obs():
    feats = hourly_features(_series())
    assert np.isnan(_at(feats, 2, 0, "rolling_median_7"))  # 2 observations seulement
    assert _at(feats, 3, 0, "rolling_median_7") == 100.0  # médiane de 0, 100, 200
    assert _at(feats, 7, 0, "rolling_median_7") == 300.0  # jours 0..6
    assert _at(feats, 10, 0, "rolling_median_7") == 600.0  # jours 3..9, jamais le jour 10


def test_censored_hours_are_nan_even_with_positive_sale():
    feats = hourly_features(_series(stockouts={(3, 10)}))  # vente 310 > 0 pendant la rupture
    assert np.isnan(_at(feats, 4, 10, "lag_d1"))
    assert np.isnan(_at(feats, 10, 10, "lag_d7"))
    assert _at(feats, 4, 10, "rolling_median_7") == 110.0  # médiane de 10, 110, 210
    assert not _at(feats, 3, 10, "available") and _at(feats, 3, 10, "sale") == 310.0


def test_discount_is_previous_day_holiday_is_same_day():
    feats = hourly_features(_series())
    assert _at(feats, 5, 12, "discount_d1") == pytest.approx(0.54)
    assert _at(feats, 5, 12, "holiday_flag") == 1 and _at(feats, 4, 12, "holiday_flag") == 0
    assert np.isnan(_at(feats, 0, 12, "discount_d1"))


def test_missing_day_uses_calendar_not_position():
    feats = hourly_features(_series(skip_days={4}))
    assert np.isnan(_at(feats, 5, 7, "lag_d1"))  # la veille réelle manque
    assert np.isnan(_at(feats, 5, 7, "discount_d1"))
    assert _at(feats, 6, 7, "lag_d1") == 507.0


def test_series_are_not_mixed():
    frame = pd.concat([_series(store=1), _series(store=2, base=10_000.0)])
    feats = hourly_features(frame)
    assert _at(feats, 8, 5, "lag_d1", store=1) == 705.0
    assert _at(feats, 8, 5, "lag_d1", store=2) == 10_705.0


def test_input_not_mutated_and_order_independent():
    frame = _series()
    before = frame.copy(deep=True)
    a = hourly_features(frame)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(a, hourly_features(frame.sample(frac=1, random_state=3)))


def test_duplicate_series_day_is_refused():
    frame = _series()
    with pytest.raises(ValueError, match="dupliqué"):
        hourly_features(pd.concat([frame, frame.iloc[[2]]]))


def test_hierarchy_values_are_static_identity():
    feats = hourly_features(_series())
    for col, value in HIER.items():
        assert (feats[col] == value).all()
    assert set(CATEGORICAL_FEATURES) >= set(HIER)


# ---------------------------------------------------------------- anti-fuite par perturbation


def _frame():
    return make_valid_frame(n_series=3, n_days=20, seed=5)


def test_clean_features_are_strictly_past():
    check_features_past_only(_frame())
    check_features_past_only(_series(stockouts={(3, 10), (9, 2)}))


def _leaky(transform):
    def builder(frame):
        return transform(hourly_features(frame), frame)

    return builder


def _same_day_discount(feats, frame):
    lookup = frame.set_index(["store_id", "product_id", "dt"])["discount"]
    idx = pd.MultiIndex.from_frame(feats[["store_id", "product_id", "dt"]])
    return feats.assign(discount_d1=lookup.reindex(idx).to_numpy())


LEAKS = {
    "lag_du_jour_meme": (_leaky(lambda f, _: f.assign(lag_d1=f["sale"])), "lag_d1"),
    "remise_du_jour_J": (_leaky(_same_day_discount), "discount_d1"),
    "mediane_incluant_J": (
        _leaky(lambda f, _: f.assign(rolling_median_7=f["rolling_median_7"].fillna(0) + f["sale"])),
        "rolling_median_7",
    ),
    "disponibilite_du_jour_J": (
        _leaky(lambda f, _: f.assign(holiday_flag=f["available"].astype(float))),
        "holiday_flag",
    ),
}


@pytest.mark.xfail(
    raises=FeatureLeakError,
    strict=True,
    reason="CONTRÔLE NÉGATIF : variable qui fuit le jour J, le test anti-fuite DOIT échouer",
)
@pytest.mark.parametrize("builder", [b for b, _ in LEAKS.values()], ids=LEAKS.keys())
def test_anti_leak_fails_on_injected_leaky_feature(builder):
    check_features_past_only(_frame(), builder=builder)


@pytest.mark.parametrize("builder, column", LEAKS.values(), ids=LEAKS.keys())
def test_each_leaky_feature_is_caught_for_the_right_column(builder, column):
    with pytest.raises(FeatureLeakError, match=column):
        check_features_past_only(_frame(), builder=builder)


def test_real_dev_features_are_strictly_past(dev):
    frame = pd.concat([dev.train, dev.validation], ignore_index=True)
    check_features_past_only(frame, cut_dates=["2024-04-15", "2024-05-27", "2024-06-05"])
