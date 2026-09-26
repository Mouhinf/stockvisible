"""Anti-fuite du test masqué.

Le même contrôle (`_assert_no_leak`) doit :
  - RÉUSSIR sur le pipeline propre ;
  - ÉCHOUER sur chaque fuite simulée injectée (contrôles négatifs, xfail strict :
    si une fuite injectée n'est plus détectée, la suite devient rouge).
Données SYNTHÉTIQUES, puis données RÉELLES dev (train + validation) uniquement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockvisible.baselines import b0_forecast, b1_forecast, hourly_matrix
from stockvisible.evaluation import (
    LeakageError,
    MaskedPipeline,
    MaskedTask,
    MaskSpec,
    build_masked_task,
    canonical,
    check_no_leakage,
    forecast_predictor,
    mask_candidates,
    materialize_task,
)
from tests._synthetic import make_valid_frame

SPEC = MaskSpec(n_per_stratum=5)
B0 = forecast_predictor(b0_forecast)
B1 = forecast_predictor(b1_forecast)


def _train_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = make_valid_frame(n_series=6, n_days=35, seed=2)
    cut = "2024-01-29"  # 28 jours d'historique (4 par jour de semaine), 7 jours cibles
    return df[df["dt"] < cut].reset_index(drop=True), df[df["dt"] >= cut].reset_index(drop=True)


# ---------------------------------------------------------------- fuites simulées


def leak_keeps_daily_total(frame, spec):
    task, vault = build_masked_task(frame, spec)
    leaky = task.frame.copy()
    leaky["sale_amount"] = canonical(frame)["sale_amount"].to_numpy()
    return MaskedTask(leaky, task.queries), vault


def leak_forgets_to_mask(frame, spec):
    task, vault = build_masked_task(frame, spec)
    leaky = task.frame.copy()
    leaky["hours_sale"] = canonical(frame)["hours_sale"].to_numpy()
    return MaskedTask(leaky, task.queries), vault


def leak_exposes_vault(frame, spec):
    task, vault = build_masked_task(frame, spec)
    leaky = task.frame.copy()
    leaky.attrs["vault"] = vault
    return MaskedTask(leaky, task.queries), vault


def peeking_predictor(history, task):
    vault = task.frame.attrs.get("vault")
    return np.array(vault._truth) if vault is not None else B1(history, task)


def leak_history_contains_targets(train, targets):
    return pd.concat([train, targets], ignore_index=True)


def leak_selects_on_sales(frame, spec):
    """Masque les heures candidates aux ventes les plus faibles : sélection sur l'issue."""
    frame = canonical(frame)
    cands = mask_candidates(frame, spec)
    value = hourly_matrix(frame, "hours_sale")[cands["row"], cands["hour"]]
    chosen = (
        cands.assign(_v=value)
        .sort_values(["_v", "row", "hour"], kind="mergesort")
        .head(20)
        .drop(columns="_v")
        .sort_values(["row", "hour"])
        .reset_index(drop=True)
    )
    return materialize_task(frame, chosen)


LEAKS = {
    "total_journalier_conserve": (MaskedPipeline(B1, build=leak_keeps_daily_total), "sale_amount"),
    "valeurs_non_masquees": (MaskedPipeline(B1, build=leak_forgets_to_mask), "visible dans : cible masquée"),
    "coffre_expose_au_predicteur": (
        MaskedPipeline(peeking_predictor, build=leak_exposes_vault), "prédictions contenant"
    ),
    "historique_contient_la_cible": (
        MaskedPipeline(B0, history_for=leak_history_contains_targets), "non strictement antérieur"
    ),
    "selection_sur_les_ventes": (MaskedPipeline(B1, build=leak_selects_on_sales), "biais de sélection"),
}
CLEAN = {"propre_B0": MaskedPipeline(B0), "propre_B1": MaskedPipeline(B1)}


def _assert_no_leak(pipeline: MaskedPipeline, train=None, targets=None) -> None:
    if train is None:
        train, targets = _train_targets()
    check_no_leakage(train, targets, pipeline, SPEC)


# ---------------------------------------------------------------- le test anti-fuite


@pytest.mark.parametrize("pipeline", CLEAN.values(), ids=CLEAN.keys())
def test_anti_leak_passes_on_clean_pipeline(pipeline):
    _assert_no_leak(pipeline)


@pytest.mark.xfail(
    raises=LeakageError,
    strict=True,
    reason="CONTRÔLE NÉGATIF : fuite injectée volontairement, le test anti-fuite DOIT échouer",
)
@pytest.mark.parametrize("pipeline", [p for p, _ in LEAKS.values()], ids=LEAKS.keys())
def test_anti_leak_fails_on_injected_leak(pipeline):
    _assert_no_leak(pipeline)


@pytest.mark.parametrize("pipeline, reason", LEAKS.values(), ids=LEAKS.keys())
def test_each_injected_leak_is_caught_for_the_right_reason(pipeline, reason):
    with pytest.raises(LeakageError, match=reason):
        _assert_no_leak(pipeline)


def test_clean_pipeline_predictions_are_real_forecasts():
    train, targets = _train_targets()
    task, vault = build_masked_task(targets, SPEC)
    pred = B1(train, task)
    assert pred.shape == (len(vault),) and np.nanmax(pred) < 100
    assert vault.score(pred)["n"] > 0


def test_detector_does_not_hide_genuine_bugs():
    def broken(history, task):
        raise RuntimeError("bug sans fuite")

    with pytest.raises(RuntimeError, match="bug sans fuite"):
        _assert_no_leak(MaskedPipeline(broken))


# ---------------------------------------------------------------- RÉEL, dev uniquement


@pytest.mark.parametrize("pipeline", CLEAN.values(), ids=CLEAN.keys())
def test_real_dev_clean_pipeline_has_no_leak(dev, pipeline):
    check_no_leakage(dev.train, dev.validation, pipeline)


def test_real_dev_negative_control_is_caught(dev):
    pipeline, reason = LEAKS["total_journalier_conserve"]
    with pytest.raises(LeakageError, match=reason):
        check_no_leakage(dev.train, dev.validation, pipeline)


def test_real_dev_never_reaches_the_test_period(dev):
    for frame in (dev.train, dev.validation):
        assert frame["dt"].max() < "2024-06-11"
    assert set(vars(dev)) == {"train", "validation"}
