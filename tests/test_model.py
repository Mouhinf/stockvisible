"""Modèle ML, prévisions glissantes et règle de sélection. SYNTHÉTIQUE + RÉEL dev (sans test)."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from stockvisible.baselines import b1_forecast
from stockvisible.evaluation import (
    PRIMARY_SCOPE,
    MaskedPipeline,
    check_no_leakage,
    forecast_predictor,
    rolling_origin,
    rolling_validation_table,
    select_engine,
)
from stockvisible.features import FEATURES, perturb_outcomes
from stockvisible.model import (
    MODEL_SPEC,
    ModelSpec,
    TemporalLeakError,
    design_matrix,
    fit_encoders,
    fit_model,
    ml_forecast,
    ml_forecaster,
)
from stockvisible.splits import dev_split
from tests._synthetic import make_valid_frame


def _dev(n_series=4, seed=11):
    """SYNTHÉTIQUE : 75 jours par série, découpés comme le vrai dev (60 / 15)."""
    return dev_split(make_valid_frame(n_series=n_series, n_days=75, seed=seed))


@pytest.fixture(scope="module")
def synth():
    dev = _dev()
    return dev, fit_model(dev.train)


# ---------------------------------------------------------------- modèle


def test_model_spec_frozen_and_everything_else_default(synth):
    assert asdict(MODEL_SPEC) == {
        "loss": "quantile", "quantile": 0.5, "early_stopping": False, "random_state": 42,
        "target_jitter": 1e-6,
    }
    _, model = synth
    params = model.estimator.get_params()
    defaults = HistGradientBoostingRegressor().get_params()
    changed = {k for k in params if params[k] != defaults[k]}
    assert changed == {"loss", "quantile", "early_stopping", "random_state", "categorical_features"}


def test_trains_only_on_available_hours_of_train(synth):
    dev, model = synth
    status = np.stack(dev.train["hours_stock_status"].map(np.asarray).to_numpy())
    assert model.n_train_rows == int((status == 0).sum())
    assert (model.train_start, model.train_end) == (dev.train["dt"].min(), dev.train["dt"].max())


def test_censored_sales_never_reach_the_model():
    dev = _dev(seed=3)
    train = dev.train.copy()
    train["hours_sale"] = [
        [1000.0 if s == 1 else v for v, s in zip(sale, status)]
        for sale, status in zip(train["hours_sale"], train["hours_stock_status"])
    ]
    model = fit_model(train)
    pred = ml_forecast(model, train, dev.validation).pred
    assert pred.max() < 10  # jamais appris sur les ventes (fausses) des heures en rupture


def test_fit_is_deterministic(synth):
    dev, model = synth
    again = fit_model(dev.train)
    np.testing.assert_array_equal(
        ml_forecast(model, dev.train, dev.validation).pred,
        ml_forecast(again, dev.train, dev.validation).pred,
    )


def test_encoders_unknown_category_is_missing_and_cardinality_guard():
    rows = pd.DataFrame({c: [1, 2, 2] for c in FEATURES})
    enc = fit_encoders(rows)
    assert enc["product_id"] == {1.0: 0, 2.0: 1}
    unseen = pd.DataFrame({c: [1, 3] for c in FEATURES})
    col = design_matrix(unseen, enc)[:, FEATURES.index("product_id")]
    assert col[0] == 0 and np.isnan(col[1])
    too_many = pd.DataFrame({c: range(256) for c in FEATURES})
    with pytest.raises(ValueError, match="256 catégories > 255"):
        fit_encoders(too_many)


def test_forecast_shape_order_and_non_negative(synth):
    dev, model = synth
    shuffled = dev.validation.sample(frac=1, random_state=2)
    f = ml_forecast(model, dev.train, shuffled)
    assert f.pred.shape == (len(shuffled), 24) and (f.pred >= 0).all()
    assert not f.abstained.any()
    pd.testing.assert_frame_equal(f.keys, shuffled[["store_id", "product_id", "dt"]].reset_index(drop=True))
    ref = ml_forecast(model, dev.train, dev.validation)
    by_key = dict(zip(map(tuple, ref.keys.to_numpy()), ref.pred))
    for key, row in zip(map(tuple, f.keys.to_numpy()), f.pred):
        np.testing.assert_array_equal(row, by_key[key])


def test_temporal_guard_refuses_model_trained_on_target_dates(synth):
    dev, model = synth
    with pytest.raises(TemporalLeakError):
        ml_forecast(model, dev.train, dev.train)
    leaky = fit_model(pd.concat([dev.train, dev.validation], ignore_index=True))  # contrôle négatif
    with pytest.raises(TemporalLeakError, match="entraîné jusqu'au"):
        ml_forecast(leaky, dev.train, dev.validation)


def test_rolling_ml_uses_only_days_before_each_target(synth):
    dev, model = synth
    days = sorted(dev.validation["dt"].unique())
    cut = days[6]
    ref = ml_forecast(model, dev.train, dev.validation)
    moved = ml_forecast(model, dev.train, perturb_outcomes(dev.validation, cut))
    upto = (ref.keys["dt"] <= cut).to_numpy()
    np.testing.assert_array_equal(ref.pred[upto], moved.pred[upto])
    assert not np.array_equal(ref.pred[~upto], moved.pred[~upto])  # le glissant sert à quelque chose


# ---------------------------------------------------------------- B0/B1 glissants


def test_rolling_origin_equals_manual_day_by_day_loop():
    dev = _dev(n_series=3, seed=4)
    rolled = rolling_origin(b1_forecast, "B1 (glissant)")(dev.train, dev.validation)
    val = dev.validation.reset_index(drop=True)
    for day in sorted(val["dt"].unique()):
        idx = np.flatnonzero((val["dt"] == day).to_numpy())
        manual = b1_forecast(pd.concat([dev.train, val[val["dt"] < day]]), val.iloc[idx])
        np.testing.assert_array_equal(rolled.pred[idx], manual.pred)
        np.testing.assert_array_equal(rolled.abstained[idx], manual.abstained)
    assert rolled.name == "B1 (glissant)"


def test_rolling_baseline_never_sees_the_target_day():
    dev = _dev(n_series=3, seed=4)
    fn = rolling_origin(b1_forecast, "B1 (glissant)")
    days = sorted(dev.validation["dt"].unique())
    ref = fn(dev.train, dev.validation)
    moved = fn(dev.train, perturb_outcomes(dev.validation, days[3]))
    upto = (ref.keys["dt"] <= days[3]).to_numpy()
    np.testing.assert_array_equal(ref.pred[upto], moved.pred[upto])


# ---------------------------------------------------------------- règle de sélection


def _table(mae_ml, mae_b1, n_ml=100, n_b1=100):
    return pd.DataFrame(
        [
            {"périmètre": PRIMARY_SCOPE, "modèle": "ML (glissant)", "n": n_ml, "mae": mae_ml},
            {"périmètre": PRIMARY_SCOPE, "modèle": "B1 (glissant)", "n": n_b1, "mae": mae_b1},
            {"périmètre": "autre", "modèle": "ML (glissant)", "n": 1, "mae": 0.0},
        ]
    )


@pytest.mark.parametrize(
    "mae_ml, mae_b1, engine",
    [(0.040, 0.041, "ML"), (0.041, 0.041, "B1"), (0.042, 0.041, "B1"), (np.nan, 0.041, "B1")],
)
def test_selection_rule(mae_ml, mae_b1, engine):
    assert select_engine(_table(mae_ml, mae_b1))["moteur"] == engine


def test_selection_rule_refuses_unfair_comparisons():
    with pytest.raises(ValueError, match="cellules"):
        select_engine(_table(0.01, 0.02, n_ml=100, n_b1=90))
    with pytest.raises(ValueError, match="absents"):
        select_engine(_table(0.01, 0.02).query("modèle != 'B1 (glissant)'"))


# ---------------------------------------------------------------- RÉEL, dev uniquement


@pytest.fixture(scope="module")
def real_model(dev):
    return fit_model(dev.train)


def test_real_model_trained_on_train_only(dev, real_model):
    assert (real_model.train_start, real_model.train_end) == ("2024-03-28", "2024-05-26")
    status = np.stack(dev.train["hours_stock_status"].map(np.asarray).to_numpy())
    assert real_model.n_train_rows == int((status == 0).sum())


def test_real_ml_passes_canary_anti_leak(dev, real_model):
    predictor = forecast_predictor(ml_forecaster(real_model))
    check_no_leakage(dev.train, dev.validation, MaskedPipeline(predictor))


def test_real_negative_control_model_fitted_on_validation_is_refused(dev):
    leaky = fit_model(pd.concat([dev.train, dev.validation], ignore_index=True))
    with pytest.raises(TemporalLeakError):
        check_no_leakage(dev.train, dev.validation, MaskedPipeline(forecast_predictor(ml_forecaster(leaky))))


def test_real_rolling_table_compares_same_cells_and_applies_rule(dev, real_model):
    table = rolling_validation_table(dev, real_model)
    primary = table[table["périmètre"] == PRIMARY_SCOPE]
    assert set(primary["modèle"]) == {"B0 (glissant)", "B1 (glissant)", "ML (glissant)"}
    assert primary["n"].nunique() == 1 and primary["n"].iloc[0] > 100_000
    decision = select_engine(table)
    assert decision["moteur"] in {"ML", "B1"}
    expected = "ML" if decision["mae_ml"] < decision["mae_b1"] else "B1"
    assert decision["moteur"] == expected


# ---------------------------------------------------------------- départage des égalités (M7b)


def _mostly_zero_dev():
    """SYNTHÉTIQUE : la série 0 vend 1,0 chaque jour de 8 h à 12 h (médiane vraie = 1) ;
    tout le reste vaut 0 → ~97 % de cibles nulles, comme la réalité horaire (71 %)."""
    frame = make_valid_frame(n_series=4, n_days=75, seed=8)
    frame["hours_stock_status"] = [[0] * 24 for _ in range(len(frame))]
    frame["stock_hour6_22_cnt"] = 0
    sales = [[1.0 if (s == 10 and 8 <= h <= 12) else 0.0 for h in range(24)] for s in frame["store_id"]]
    frame["hours_sale"] = sales
    frame["sale_amount"] = [sum(x) for x in sales]
    return dev_split(frame)


def test_model_learns_a_positive_median_despite_massive_ties():
    dev = _mostly_zero_dev()
    f = ml_forecast(fit_model(dev.train), dev.train, dev.validation)
    busy = (f.keys["store_id"] == 10).to_numpy()
    assert (f.pred[busy][:, 8:13] > 0.9).all()
    assert (f.pred[busy][:, :8] < 0.1).all() and (f.pred[~busy] < 0.1).all()


def test_without_tie_breaking_hgb_quantile_stays_at_zero():
    """Contrôle négatif du correctif : sans bruit, sklearn reste bloqué à la valeur initiale 0.
    Si ce test casse après une mise à jour de sklearn, réexaminer la nécessité du bruit."""
    dev = _mostly_zero_dev()
    frozen = fit_model(dev.train, replace(ModelSpec(), target_jitter=0.0))
    assert (ml_forecast(frozen, dev.train, dev.validation).pred == 0).all()
