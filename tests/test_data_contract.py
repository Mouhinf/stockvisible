"""Contrat de données F1. Tests unitaires sur données SYNTHÉTIQUES + contrôles sur le sous-ensemble RÉEL local."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from stockvisible import data
from stockvisible.data import (
    CONTRACT_COLUMNS,
    DATASET_REVISION,
    SERIES_KEY,
    SOURCE_SHA256,
    SubsetSpec,
    classify_split,
    load_raw,
    select_series,
    sha256_file,
    source_url,
    verify_sha256,
)
from stockvisible.validation import validate
from tests._synthetic import SYNTHETIC_LABEL, make_valid_frame

OFFICIAL_SCHEMA = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "dt", "sale_amount",
    "hours_sale", "stock_hour6_22_cnt", "hours_stock_status", "discount",
    "holiday_flag", "activity_flag", "precpt", "avg_temperature", "avg_humidity",
    "avg_wind_level",
]  # fmt: skip


def _keys(n_store: int = 20, n_prod: int = 15) -> pd.DataFrame:
    s, p = np.meshgrid(np.arange(n_store), np.arange(n_prod))
    return pd.DataFrame({"store_id": s.ravel(), "product_id": p.ravel()})


def _frame(keys: list[tuple[int, int]], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(s, p, d) for s, p in keys for d in dates], columns=[*SERIES_KEY, "dt"]
    )


# ---------------------------------------------------------------- schéma


def test_contract_matches_official_schema_exactly():
    assert list(CONTRACT_COLUMNS) == OFFICIAL_SCHEMA


def test_no_currency_naming_in_contract():
    assert not any("fcfa" in c.lower() or "xof" in c.lower() for c in CONTRACT_COLUMNS)


def test_load_raw_keeps_names_and_normalised_values(tmp_path):
    df = make_valid_frame()
    df.to_parquet(tmp_path / "train.parquet", index=False)
    loaded = load_raw("train", raw_dir=tmp_path)
    assert list(loaded.columns) == list(CONTRACT_COLUMNS)
    np.testing.assert_array_equal(loaded["sale_amount"], df["sale_amount"])


def test_load_raw_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="stockvisible.data"):
        load_raw("eval", raw_dir=tmp_path)
    with pytest.raises(ValueError):
        load_raw("test", raw_dir=tmp_path)


def test_synthetic_helper_is_labelled():
    assert SYNTHETIC_LABEL == "SYNTHETIC-TEST-ONLY"


# ---------------------------------------------------------------- source figée


def test_source_url_is_pinned_https():
    for split in ("train", "eval"):
        url = source_url(split)
        assert url.startswith("https://huggingface.co/datasets/")
        assert DATASET_REVISION in url and "/main/" not in url
    with pytest.raises(ValueError):
        source_url("test")


def test_verify_sha256(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"stockvisible")
    verify_sha256(path, sha256_file(path))
    with pytest.raises(ValueError, match="SHA-256"):
        verify_sha256(path, "0" * 64)


# ---------------------------------------------------------------- sélection déterministe


def test_subset_criteria_are_frozen():
    assert asdict(SubsetSpec()) == {"n_series": 500, "seed": 42}


def test_select_series_is_deterministic_and_order_independent():
    keys = _keys()
    a = select_series(keys, 30, seed=42)
    b = select_series(keys.sample(frac=1, random_state=7), 30, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 30 and not a.duplicated().any()
    assert set(map(tuple, a.to_numpy())) <= set(map(tuple, keys.to_numpy()))


def test_select_series_ignores_outcome_columns():
    keys = _keys()
    rng = np.random.default_rng(0)
    with_outcomes = keys.assign(sale_amount=rng.random(len(keys)), stock=rng.integers(0, 2, len(keys)))
    shuffled = with_outcomes.assign(sale_amount=rng.permutation(with_outcomes["sale_amount"].to_numpy()))
    pd.testing.assert_frame_equal(
        select_series(with_outcomes, 30, 42), select_series(shuffled, 30, 42)
    )


def test_select_series_seed_matters_and_bounds():
    keys = _keys()
    assert not select_series(keys, 30, 1).equals(select_series(keys, 30, 2))
    for bad in (0, len(keys) + 1):
        with pytest.raises(ValueError):
            select_series(keys, bad, 42)


def test_duplicate_key_rows_count_once():
    keys = pd.concat([_keys(3, 2)] * 5)
    assert len(select_series(keys, 6, 0)) == 6


# ---------------------------------------------------------------- nature du split


def test_classify_temporal_same_series():
    k = [(1, 1), (1, 2), (2, 1)]
    out = classify_split(_frame(k, ["2024-01-01", "2024-01-02"]), _frame(k, ["2024-01-03"]))
    assert out["kind"] == "temporal_same_series"
    assert out["n_series_shared"] == 3 and not out["date_overlap"]


def test_classify_series_disjoint():
    out = classify_split(
        _frame([(1, 1), (1, 2)], ["2024-01-01"]), _frame([(9, 9)], ["2024-01-01"])
    )
    assert out["kind"] == "series_disjoint" and out["n_series_shared"] == 0


def test_classify_mixed_partial_overlap_or_date_overlap():
    partial = classify_split(
        _frame([(1, 1), (1, 2)], ["2024-01-01"]), _frame([(1, 1), (9, 9)], ["2024-01-02"])
    )
    overlap = classify_split(
        _frame([(1, 1)], ["2024-01-01", "2024-01-02"]), _frame([(1, 1)], ["2024-01-02"])
    )
    assert partial["kind"] == "mixed"
    assert overlap["kind"] == "mixed" and overlap["date_overlap"]


# ---------------------------------------------------------------- sous-ensemble RÉEL local

RAW_PRESENT = (data.RAW_DIR / "manifest.json").exists()
CACHE_PRESENT = all((data.CACHE_DIR / f"{s}.parquet").exists() for s in data.SPLITS)
needs_raw = pytest.mark.skipif(not RAW_PRESENT, reason="lancer `python -m stockvisible.data`")
needs_cache = pytest.mark.skipif(not CACHE_PRESENT, reason="fichiers officiels non téléchargés")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((data.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real() -> dict[str, pd.DataFrame]:
    return {s: load_raw(s) for s in data.SPLITS}


@needs_raw
def test_manifest_traces_real_pinned_source(manifest):
    assert manifest["data_nature"].startswith("REAL")
    assert manifest["revision"] == DATASET_REVISION
    assert manifest["source_sha256"] == SOURCE_SHA256
    assert manifest["subset_spec"] == asdict(SubsetSpec())
    for split, out in manifest["outputs"].items():
        assert sha256_file(data.RAW_DIR / out["path"]) == out["sha256"], split


@needs_raw
def test_official_split_verified_as_temporal_same_series(manifest):
    split = manifest["official_split"]
    assert split["kind"] == "temporal_same_series"
    assert split["n_series_train"] == split["n_series_eval"] == split["n_series_shared"] == 50_000
    assert not split["date_overlap"]


@needs_raw
def test_subset_shape_and_split_nature(real):
    n = SubsetSpec().n_series
    tr, ev = real["train"], real["eval"]
    assert list(tr.columns) == list(CONTRACT_COLUMNS)
    assert tr.groupby(list(SERIES_KEY)).size().eq(90).all() and len(tr) == n * 90
    assert ev.groupby(list(SERIES_KEY)).size().eq(7).all() and len(ev) == n * 7
    assert classify_split(tr, ev)["kind"] == "temporal_same_series"


@needs_raw
def test_validator_has_no_false_positive_on_real_data(real):
    for split, df in real.items():
        report = validate(df)
        assert report.ok, (split, report.errors[:3])


@needs_raw
def test_real_data_shows_zero_sale_is_not_stockout(real):
    tr = real["train"]
    no_stockout = tr["hours_stock_status"].map(lambda s: int(np.sum(s)) == 0)
    assert ((tr["sale_amount"] == 0) & no_stockout).any()
    assert "SALE_DURING_STOCKOUT" in validate(tr).codes("info")


@needs_raw
@needs_cache
def test_subset_is_reproducible_and_unaltered_from_source(real):
    keys = pq.read_table(data.CACHE_DIR / "train.parquet", columns=list(SERIES_KEY)).to_pandas()
    expected = select_series(keys, SubsetSpec().n_series, SubsetSpec().seed)
    got = real["train"].loc[:, list(SERIES_KEY)].drop_duplicates().reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)

    store, product = (int(v) for v in expected.iloc[0])
    for split, df in real.items():
        src = pq.read_table(
            data.CACHE_DIR / f"{split}.parquet",
            filters=[("store_id", "=", store), ("product_id", "=", product)],
        ).to_pandas().sort_values("dt").reset_index(drop=True)
        mine = df[(df.store_id == store) & (df.product_id == product)].reset_index(drop=True)
        pd.testing.assert_frame_equal(mine, src)
