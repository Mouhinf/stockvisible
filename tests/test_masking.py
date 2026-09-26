"""Masquage stratifié + coffre de vérité. Cas SYNTHÉTIQUES, puis données RÉELLES dev uniquement."""

from __future__ import annotations

import gc
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from stockvisible import data
from stockvisible.baselines import b1_forecast, hourly_matrix
from stockvisible.evaluation import (
    MASK_SPEC,
    QUERY_COLUMNS,
    MaskSpec,
    TruthVault,
    build_masked_task,
    canonical,
    forecast_predictor,
    mask_candidates,
)
from tests._synthetic import make_valid_frame

SMALL = MaskSpec(n_per_stratum=5)


def _targets(seed: int = 1) -> pd.DataFrame:
    return make_valid_frame(n_series=6, n_days=10, seed=seed)


def _row(status_hours: list[int], sale: float = 1.0, dt: str = "2024-01-01", store: int = 1) -> dict:
    status = [0] * 24
    for h in status_hours:
        status[h] = 1
    sales = [0.0 if s else sale for s in status]
    return {
        "store_id": store, "product_id": 1, "dt": dt, "sale_amount": float(sum(sales)),
        "hours_sale": sales, "hours_stock_status": status,
        "stock_hour6_22_cnt": int(sum(status[6:22])),
    }


def test_mask_spec_is_frozen():
    assert asdict(MASK_SPEC) == {
        "n_per_stratum": 250, "max_distance": 3, "window": (6, 22), "seed": 42
    }


def test_same_seed_same_positions_whatever_the_row_order():
    a, _ = build_masked_task(_targets(), SMALL)
    b, _ = build_masked_task(_targets().sample(frac=1, random_state=9), SMALL)
    pd.testing.assert_frame_equal(a.queries, b.queries)
    pd.testing.assert_frame_equal(a.frame, b.frame)


def test_different_seed_different_positions():
    a, _ = build_masked_task(_targets(), SMALL)
    b, _ = build_masked_task(_targets(), MaskSpec(n_per_stratum=5, seed=7))
    assert not a.queries[["row", "hour"]].equals(b.queries[["row", "hour"]])


def test_positions_are_available_hours_close_to_a_real_stockout():
    task, _ = build_masked_task(_targets(), SMALL)
    st = hourly_matrix(canonical(_targets()), "hours_stock_status")
    for q in task.queries.itertuples():
        assert st[q.row, q.hour] == 0 and 6 <= q.hour < 22
        stock_hours = [h for h in range(6, 22) if st[q.row, h] == 1]
        nearest = min(abs(h - q.hour) for h in stock_hours)
        assert nearest == q.distance <= 3


def test_days_without_stockout_or_with_night_stockout_only_are_never_candidates():
    frame = pd.DataFrame([
        _row([], dt="2024-01-01"),
        _row([0, 1, 2, 23], dt="2024-01-02"),  # ruptures de nuit uniquement
        _row([12], dt="2024-01-03"),
    ])
    cands = mask_candidates(frame)
    assert set(cands["row"]) == {2}
    assert sorted(cands["hour"]) == [9, 10, 11, 13, 14, 15]
    with pytest.raises(ValueError, match="aucune position"):
        build_masked_task(frame.iloc[:2])


def test_strata_distance_and_side():
    cands = mask_candidates(pd.DataFrame([_row([10, 16])]))
    by_hour = cands.set_index("hour")
    assert by_hour.loc[9, "side"] == "avant_rupture" and by_hour.loc[9, "distance"] == 1
    assert by_hour.loc[11, "side"] == "apres_rupture" and by_hour.loc[11, "distance"] == 1
    assert by_hour.loc[13, "side"] == "entre_ruptures" and by_hour.loc[13, "distance"] == 3


def test_sampling_is_stratified():
    task, _ = build_masked_task(_targets(), SMALL)
    available = mask_candidates(canonical(_targets())).groupby(["distance", "side"]).size()
    taken = task.queries.groupby(["distance", "side"]).size()
    expected = available.clip(upper=SMALL.n_per_stratum)
    pd.testing.assert_series_equal(taken, expected, check_names=False)
    assert (taken == SMALL.n_per_stratum).sum() >= 3


def test_positions_never_depend_on_sales():
    rng = np.random.default_rng(3)
    shuffled = _targets().copy()
    shuffled["hours_sale"] = [list(rng.random(24) * 50) for _ in range(len(shuffled))]
    a, _ = build_masked_task(_targets(), SMALL)
    b, _ = build_masked_task(shuffled, SMALL)
    pd.testing.assert_frame_equal(a.queries, b.queries)


def test_masked_frame_hides_exactly_the_positions_and_the_daily_total():
    targets = _targets()
    before = targets.copy(deep=True)
    task, _ = build_masked_task(targets, SMALL)
    pd.testing.assert_frame_equal(targets, before)  # entrée intacte

    ref = canonical(targets)
    masked = hourly_matrix(task.frame, "hours_sale")
    original = hourly_matrix(ref, "hours_sale")
    hidden = np.zeros_like(masked, dtype=bool)
    hidden[task.queries["row"], task.queries["hour"]] = True
    assert np.isnan(masked[hidden]).all()
    np.testing.assert_array_equal(masked[~hidden], original[~hidden])

    rows_hidden = hidden.any(axis=1)
    assert task.frame.loc[rows_hidden, "sale_amount"].isna().all()
    np.testing.assert_array_equal(task.frame.loc[~rows_hidden, "sale_amount"], ref.loc[~rows_hidden, "sale_amount"])
    others = [c for c in ref.columns if c not in ("hours_sale", "sale_amount")]
    pd.testing.assert_frame_equal(task.frame[others], ref[others])


def test_queries_carry_no_values():
    task, _ = build_masked_task(_targets(), SMALL)
    assert list(task.queries.columns) == QUERY_COLUMNS
    assert "hours_sale" not in task.queries and "sale_amount" not in task.queries


def test_vault_is_physically_isolated_and_read_only():
    targets = _targets()
    task, vault = build_masked_task(targets, SMALL)
    truth = vault._truth
    assert not truth.flags.writeable
    with pytest.raises(ValueError):
        truth[0] = 0.0
    for cell in [*task.frame["hours_sale"], *targets["hours_sale"]]:
        assert not np.shares_memory(truth, np.asarray(cell))
    assert {id(x) for x in task.frame["hours_sale"]}.isdisjoint({id(x) for x in targets["hours_sale"]})
    assert all(isinstance(v, pd.DataFrame) for v in vars(task).values())
    assert not any(o is vault for o in gc.get_referents(task.frame)) and not task.frame.attrs
    assert "valeurs masquées scellées" in repr(vault)
    with pytest.raises(AttributeError):
        vault.extra = 1  # __slots__ : aucun attribut ajouté après coup


def test_vault_scores_without_exposing_values():
    targets = _targets()
    task, vault = build_masked_task(targets, SMALL)
    true = hourly_matrix(canonical(targets), "hours_sale")[task.queries["row"], task.queries["hour"]]
    assert vault.score(true)["mae"] == 0.0
    shifted = vault.score(true + 1.0)
    assert shifted["mae"] == pytest.approx(1.0) and shifted["bias"] == pytest.approx(1.0)
    half = true.copy()
    half[::2] = np.nan
    assert vault.score(half)["couverture"] == pytest.approx(0.5, abs=0.05)
    assert len(vault.score_by_stratum(true)) == task.queries.groupby(["distance", "side"]).ngroups
    with pytest.raises(ValueError):
        vault.score(true[:-1])
    assert isinstance(vault, TruthVault)


# ---------------------------------------------------------------- RÉEL, dev uniquement

RAW_PRESENT = (data.RAW_DIR / "train.parquet").exists()


@pytest.mark.skipif(not RAW_PRESENT, reason="lancer `python -m stockvisible.data`")
def test_real_dev_masking(dev):
    assert dev.train["dt"].max() == "2024-05-26" and dev.validation["dt"].min() == "2024-05-27"
    assert dev.validation["dt"].max() == "2024-06-10"  # < 2024-06-11, début du test réservé
    task, vault = build_masked_task(dev.validation)
    per_stratum = task.queries.groupby(["distance", "side"]).size()
    assert per_stratum.max() == MASK_SPEC.n_per_stratum and len(vault) == per_stratum.sum()
    assert task.queries["dt"].between("2024-05-27", "2024-06-10").all()
    pred = forecast_predictor(b1_forecast)(dev.train, task)
    assert pred.shape == (len(vault),)
