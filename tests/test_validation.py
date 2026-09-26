"""validate() et lecture sûre — cas SYNTHÉTIQUES volontairement corrompus."""

from __future__ import annotations

import copy
import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from stockvisible.validation import (
    ERROR,
    INFO,
    InputRejected,
    read_user_bytes,
    read_user_file,
    validate,
)
from tests._synthetic import make_valid_frame

ROW = 5  # ligne ciblée par les corruptions (série 1, jour 1)


def _set(df: pd.DataFrame, col: str, value, row: int = ROW) -> pd.DataFrame:
    df[col] = df[col].astype(object)
    df.at[row, col] = value
    return df


def _list_with(col_values: list, idx: int, value) -> list:
    out = list(col_values)
    out[idx] = value
    return out


def _dup_dt(df):
    return _set(df, "dt", df.at[ROW - 1, "dt"])


def _dup_row(df):
    return pd.concat([df, df.iloc[[ROW]]], ignore_index=True)


def _hours_sale_negative(df):
    hs = _list_with(df.at[ROW, "hours_sale"], 3, -1.0)
    df = _set(df, "hours_sale", hs)
    return _set(df, "sale_amount", float(sum(hs)))


def _status_flip_in_window(df):
    st = list(df.at[ROW, "hours_stock_status"])
    st[10] = 1 - st[10]
    return _set(df, "hours_stock_status", st)


def _cnt_plus_one(df):
    df = _set(df, "hours_stock_status", [0] * 24)
    df = _set(df, "hours_sale", [0.1] * 24)
    df = _set(df, "sale_amount", float(sum([0.1] * 24)))
    return _set(df, "stock_hour6_22_cnt", 1)


def _hours_sale_changed_only(df):
    return _set(df, "hours_sale", _list_with(df.at[ROW, "hours_sale"], 8, 99.0))


# (identifiant, corruption, code attendu, ligne attendue ou None si globale)
CORRUPTIONS = [
    ("dt_duplique_meme_serie", _dup_dt, "DUPLICATE_DATE", ROW),
    ("ligne_dupliquee", _dup_row, "DUPLICATE_DATE", 12),
    ("dt_mois_13", lambda d: _set(d, "dt", "2024-13-01"), "BAD_DATE", ROW),
    ("dt_format_fr", lambda d: _set(d, "dt", "01/02/2024"), "BAD_DATE", ROW),
    ("dt_manquant", lambda d: _set(d, "dt", None), "NULL_VALUE", ROW),
    ("sale_amount_negatif", lambda d: _set(d, "sale_amount", -2.0), "OUT_OF_RANGE", ROW),
    ("sale_amount_nan", lambda d: _set(d, "sale_amount", np.nan), "NULL_VALUE", ROW),
    ("sale_amount_texte", lambda d: _set(d, "sale_amount", "abc"), "NOT_NUMERIC", ROW),
    ("sale_amount_inf", lambda d: _set(d, "sale_amount", np.inf), "OUT_OF_RANGE", ROW),
    ("hours_sale_negatif", _hours_sale_negative, "SEQ_OUT_OF_RANGE", ROW),
    ("hours_sale_23h", lambda d: _set(d, "hours_sale", [0.0] * 23), "BAD_SEQUENCE", ROW),
    ("hours_sale_nan", lambda d: _set(d, "hours_sale", [np.nan] * 24), "BAD_SEQUENCE", ROW),
    ("hours_sale_none", lambda d: _set(d, "hours_sale", None), "BAD_SEQUENCE", ROW),
    ("hours_sale_scalaire", lambda d: _set(d, "hours_sale", 3.0), "BAD_SEQUENCE", ROW),
    ("hours_sale_texte", lambda d: _set(d, "hours_sale", ["a"] * 24), "BAD_SEQUENCE", ROW),
    ("status_25h", lambda d: _set(d, "hours_stock_status", [0] * 25), "BAD_SEQUENCE", ROW),
    ("status_valeur_2", lambda d: _set(d, "hours_stock_status", [2] * 24), "SEQ_OUT_OF_RANGE", ROW),
    ("status_negatif", lambda d: _set(d, "hours_stock_status", [-1] * 24), "SEQ_OUT_OF_RANGE", ROW),
    ("status_demi", lambda d: _set(d, "hours_stock_status", [0.5] * 24), "SEQ_OUT_OF_RANGE", ROW),
    ("cnt_17", lambda d: _set(d, "stock_hour6_22_cnt", 17), "OUT_OF_RANGE", ROW),
    ("cnt_negatif", lambda d: _set(d, "stock_hour6_22_cnt", -1), "OUT_OF_RANGE", ROW),
    ("cnt_non_entier", lambda d: _set(d, "stock_hour6_22_cnt", 2.5), "NON_INTEGER", ROW),
    ("cnt_incoherent_status", _cnt_plus_one, "STOCK_COUNT_MISMATCH", ROW),
    ("status_modifie_sans_cnt", _status_flip_in_window, "STOCK_COUNT_MISMATCH", ROW),
    ("sale_amount_sans_hours", lambda d: _set(d, "sale_amount", d.at[ROW, "sale_amount"] + 1),
     "SALE_SUM_MISMATCH", ROW),
    ("hours_sans_sale_amount", _hours_sale_changed_only, "SALE_SUM_MISMATCH", ROW),
    ("discount_1_5", lambda d: _set(d, "discount", 1.5), "OUT_OF_RANGE", ROW),
    ("discount_negatif", lambda d: _set(d, "discount", -0.1), "OUT_OF_RANGE", ROW),
    ("holiday_flag_2", lambda d: _set(d, "holiday_flag", 2), "OUT_OF_RANGE", ROW),
    ("activity_flag_demi", lambda d: _set(d, "activity_flag", 0.5), "NON_INTEGER", ROW),
    ("precpt_negatif", lambda d: _set(d, "precpt", -1.0), "OUT_OF_RANGE", ROW),
    ("humidite_120", lambda d: _set(d, "avg_humidity", 120.0), "OUT_OF_RANGE", ROW),
    ("temperature_80", lambda d: _set(d, "avg_temperature", 80.0), "OUT_OF_RANGE", ROW),
    ("vent_negatif", lambda d: _set(d, "avg_wind_level", -1.0), "OUT_OF_RANGE", ROW),
    ("store_id_negatif", lambda d: _set(d, "store_id", -3), "OUT_OF_RANGE", ROW),
    ("product_id_decimal", lambda d: _set(d, "product_id", 1.5), "NON_INTEGER", ROW),
    ("city_id_nan", lambda d: _set(d, "city_id", np.nan), "NULL_VALUE", ROW),
    ("colonne_absente", lambda d: d.drop(columns=["hours_stock_status"]), "MISSING_COLUMN", None),
]


def _detected(report, code: str, row: int | None) -> bool:
    hits = [i for i in report.errors if i.code == code]
    return bool(hits) and (row is None or any(row in i.rows for i in hits))


def test_valid_synthetic_frame_has_no_error():
    report = validate(make_valid_frame())
    assert report.ok, report.errors


@pytest.mark.parametrize(
    "corrupt, code, row", [c[1:] for c in CORRUPTIONS], ids=[c[0] for c in CORRUPTIONS]
)
def test_corruption_is_detected_on_the_right_row(corrupt, code, row):
    report = validate(corrupt(make_valid_frame()))
    assert not report.ok
    assert _detected(report, code, row), report.issues


def test_detection_rate_is_100_percent():
    missed = [
        name
        for name, corrupt, code, row in CORRUPTIONS
        if not _detected(validate(corrupt(make_valid_frame())), code, row)
    ]
    rate = 1 - len(missed) / len(CORRUPTIONS)
    assert rate == 1.0, f"taux de détection {rate:.0%}, manqués : {missed}"


def test_corruptions_do_not_leak_to_untouched_rows():
    both_copies = {"dt_duplique_meme_serie": {ROW - 1, ROW}, "ligne_dupliquee": {ROW, 12}}
    for name, corrupt, _code, row in CORRUPTIONS:
        if row is None:
            continue
        flagged = {r for i in validate(corrupt(make_valid_frame())).errors for r in i.rows}
        assert flagged == both_copies.get(name, {row}), (name, flagged)


def test_zero_sale_without_stockout_is_not_an_error():
    df = make_valid_frame()
    assert df.at[0, "sale_amount"] == 0 and sum(df.at[0, "hours_stock_status"]) == 0
    report = validate(df)
    assert not any(0 in i.rows for i in report.issues)


def test_sale_during_stockout_is_info_not_error():
    report = validate(make_valid_frame())
    info = [i for i in report.issues if i.code == "SALE_DURING_STOCKOUT"]
    assert info and info[0].severity == INFO and 1 in info[0].rows
    assert all(i.code != "SALE_DURING_STOCKOUT" for i in report.errors)


def test_stockout_outside_6_22_window_does_not_touch_count():
    df = make_valid_frame()
    st = list(df.at[ROW, "hours_stock_status"])
    st[2], st[22], st[23] = 1, 1, 1
    df = _set(df, "hours_stock_status", st)
    df = _set(df, "hours_sale", _list_with(_list_with(_list_with(df.at[ROW, "hours_sale"], 2, 0.0), 22, 0.0), 23, 0.0))
    df = _set(df, "sale_amount", float(sum(df.at[ROW, "hours_sale"])))
    assert validate(df).ok


def test_validate_does_not_mutate_input():
    df = _dup_row(make_valid_frame())
    before = copy.deepcopy(df)
    validate(df)
    pd.testing.assert_frame_equal(df, before)


def test_rows_are_positions_even_with_custom_index():
    df = _set(make_valid_frame(), "discount", 2.0)
    df.index = df.index + 1000
    issue = next(i for i in validate(df).errors if i.code == "OUT_OF_RANGE")
    assert issue.rows == (ROW,)


def test_all_error_severities_are_error():
    report = validate(_dup_row(make_valid_frame()))
    assert all(i.severity == ERROR for i in report.errors)


# ---------------------------------------------------------------- sécurité d'entrée


def _csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for col in ("hours_sale", "hours_stock_status"):
        out[col] = out[col].map(json.dumps)
    return out.to_csv(index=False).encode("utf-8")


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class _Payload:
    def __init__(self, target: str):
        self.target = target

    def __reduce__(self):
        return (os.makedirs, (self.target,))


def test_valid_csv_roundtrip_is_accepted():
    df = read_user_bytes("ventes.csv", _csv_bytes(make_valid_frame()))
    assert validate(df).ok


def test_valid_parquet_file_roundtrip_is_accepted(tmp_path):
    path = tmp_path / "ventes.PARQUET"
    path.write_bytes(_parquet_bytes(make_valid_frame()))
    assert validate(read_user_file(path)).ok


@pytest.mark.parametrize(
    "name", ["data.txt", "data.pkl", "data.json", "data.xlsx", "data.csv.exe", "data", "data.py"]
)
def test_only_csv_and_parquet_extensions(name):
    with pytest.raises(InputRejected, match="extension"):
        read_user_bytes(name, b"a,b\n1,2\n")


def test_empty_file_is_rejected():
    with pytest.raises(InputRejected, match="vide"):
        read_user_bytes("x.csv", b"")


def test_oversized_file_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "big.csv"
    path.write_bytes(b"a\n" + b"1\n" * 100)
    with pytest.raises(InputRejected, match="taille"):
        read_user_file(path, max_bytes=50)
    with pytest.raises(InputRejected, match="taille"):
        read_user_bytes("big.csv", path.read_bytes(), max_bytes=50)


def test_directory_and_symlink_are_rejected(tmp_path):
    with pytest.raises(InputRejected):
        read_user_file(tmp_path)
    target = tmp_path / "real.csv"
    target.write_bytes(_csv_bytes(make_valid_frame()))
    link = tmp_path / "link.csv"
    link.symlink_to(target)
    with pytest.raises(InputRejected):
        read_user_file(link)


def test_parquet_extension_with_other_content_is_rejected():
    with pytest.raises(InputRejected, match="PAR1"):
        read_user_bytes("x.parquet", _csv_bytes(make_valid_frame()))


def test_fake_parquet_with_magic_only_is_rejected():
    with pytest.raises(InputRejected, match="illisible"):
        read_user_bytes("x.parquet", b"PAR1" + b"\x00" * 32 + b"PAR1")


def test_pickle_disguised_as_parquet_is_never_executed(tmp_path):
    sentinel = tmp_path / "pwned"
    payload = pickle.dumps(_Payload(str(sentinel)))
    for name in ("x.parquet", "x.csv"):
        with pytest.raises(InputRejected):
            read_user_bytes(name, payload)
    assert not sentinel.exists()


def test_csv_with_nul_byte_or_non_utf8_is_rejected():
    with pytest.raises(InputRejected, match="binaire"):
        read_user_bytes("x.csv", b"a,b\n1,\x002\n")
    with pytest.raises(InputRejected, match="UTF-8"):
        read_user_bytes("x.csv", b"a,b\n\xff\xfe,2\n")


@pytest.mark.parametrize(
    "cell",
    [
        "__import__('os').makedirs('{sentinel}')",
        "[__import__('os').makedirs('{sentinel}')]",
        "{{'a': 1}}",
        "[1, true]",
        '["1", "2"]',
        "[[1], [2]]",
    ],
)
def test_csv_list_cells_are_parsed_as_json_never_evaluated(tmp_path, cell):
    sentinel = tmp_path / "pwned"
    df = make_valid_frame()
    raw = df.copy()
    raw["hours_sale"] = raw["hours_sale"].map(json.dumps)
    raw["hours_stock_status"] = raw["hours_stock_status"].map(json.dumps)
    raw.at[ROW, "hours_sale"] = cell.format(sentinel=sentinel)
    with pytest.raises(InputRejected, match="cellule"):
        read_user_bytes("x.csv", raw.to_csv(index=False).encode("utf-8"))
    assert not sentinel.exists()
