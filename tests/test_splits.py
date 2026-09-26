"""Split chronologique — cas SYNTHÉTIQUES + contrôle sur le sous-ensemble RÉEL local."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest

from stockvisible import data
from stockvisible.data import SERIES_KEY, SubsetSpec, load_raw
from stockvisible.splits import (
    SPLIT_SPEC,
    UNSEAL_PHRASE,
    SealedFrame,
    chronological_split,
    date_ranges,
)
from tests._synthetic import make_valid_frame

KEYS = list(SERIES_KEY)


def _frame(n_series: int = 3, n_days: int = 90) -> pd.DataFrame:
    return make_valid_frame(n_series=n_series, n_days=n_days)


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([*KEYS, "dt"]).reset_index(drop=True)


def test_split_spec_is_frozen_60_15_15():
    assert asdict(SPLIT_SPEC) == {"train_days": 60, "validation_days": 15, "test_days": 15}
    with pytest.raises(AttributeError):
        SPLIT_SPEC.train_days = 70  # type: ignore[misc]


def test_sizes_per_series():
    s = chronological_split(_frame())
    assert s.train.groupby(KEYS).size().eq(60).all()
    assert s.validation.groupby(KEYS).size().eq(15).all()
    assert s.test.open(UNSEAL_PHRASE).groupby(KEYS).size().eq(15).all()


def test_strictly_chronological_per_series_no_overlap():
    s = chronological_split(_frame())
    test = s.test.open(UNSEAL_PHRASE)
    for key, tr in s.train.groupby(KEYS):
        va = s.validation.set_index(KEYS).loc[key]
        te = test.set_index(KEYS).loc[key]
        assert tr["dt"].max() < va["dt"].min() <= va["dt"].max() < te["dt"].min()
    union = pd.concat([s.train, s.validation, test])
    pd.testing.assert_frame_equal(_canon(union), _canon(_frame()))


@pytest.mark.parametrize("seed", [0, 1, 42, 2024])
def test_split_is_reproducible_whatever_the_row_order_seed(seed):
    ref = chronological_split(_frame())
    shuffled = chronological_split(_frame().sample(frac=1, random_state=seed))
    pd.testing.assert_frame_equal(ref.train, shuffled.train)
    pd.testing.assert_frame_equal(ref.validation, shuffled.validation)
    pd.testing.assert_frame_equal(
        ref.test.open(UNSEAL_PHRASE), shuffled.test.open(UNSEAL_PHRASE)
    )


def test_same_seed_same_split_twice():
    a, b = chronological_split(_frame()), chronological_split(_frame())
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.validation, b.validation)


def test_input_is_not_mutated():
    df = _frame()
    before = df.copy()
    chronological_split(df)
    pd.testing.assert_frame_equal(df, before)


def test_test_partition_is_sealed():
    s = chronological_split(_frame())
    assert isinstance(s.test, SealedFrame) and "scellé" in repr(s.test)
    for wrong in ("", "test", "TEST FINAL"):
        with pytest.raises(PermissionError):
            s.test.open(wrong)
    assert s.test.n_rows == 3 * 15


def test_rejects_short_series():
    with pytest.raises(ValueError, match="longueur"):
        chronological_split(_frame(n_days=89))


def test_rejects_gap_or_duplicate_date_in_one_series():
    gap = _frame()
    gap.loc[89, "dt"] = "2024-03-31"  # série (10, 100) : 90 jours mais un trou le 30/03
    dup = _frame()
    dup.loc[1, "dt"] = dup.loc[0, "dt"]  # série (10, 100) : 90 lignes mais un doublon
    for broken in (gap, dup):
        with pytest.raises(ValueError) as exc:
            chronological_split(broken)
        assert "longueur []" in str(exc.value) and "[10, 100]" in str(exc.value)


RAW_PRESENT = (data.RAW_DIR / "train.parquet").exists()


@pytest.mark.skipif(not RAW_PRESENT, reason="lancer `python -m stockvisible.data`")
def test_real_subset_split_boundaries():
    assert asdict(SubsetSpec()) == {"n_series": 500, "seed": 42}
    s = chronological_split(load_raw("train"))
    assert date_ranges(s) == {
        "train": ("2024-03-28", "2024-05-26"),
        "validation": ("2024-05-27", "2024-06-10"),
        "test": ("2024-06-11", "2024-06-25"),
    }
    assert (len(s.train), len(s.validation), s.test.n_rows) == (30_000, 7_500, 7_500)
