"""Test final (M9) — écrit AVANT l'ouverture du test réservé.

- gardes : une seule exécution, gel vérifié avant toute lecture ;
- pipeline éprouvé sur SYNTHÉTIQUE, puis répétition générale sur la VALIDATION réelle (doit
  retrouver exactement les métriques gelées en M8) ;
- vérification du scellé : active seulement une fois logs/final_test_result.json écrit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stockvisible import final_test
from stockvisible.data import RAW_DIR, sha256_file
from stockvisible.evaluation import PRIMARY_SCOPE
from stockvisible.final_test import (
    PROTOCOL,
    RESULT_PATH,
    FinalTestAlreadyRun,
    evaluate_sealed,
    run_final_test,
)
from stockvisible.model import fit_model
from stockvisible.selection import (
    FREEZE_PATH,
    FreezeViolation,
    check_freeze,
    load_freeze,
)
from stockvisible.splits import UNSEAL_PHRASE, chronological_split
from tests._synthetic import make_valid_frame


def _forbid(*_args, **_kwargs):
    raise AssertionError("lecture de données interdite dans ce test")


def test_protocol_fixed_before_opening():
    assert PROTOCOL["entraînement"].startswith("modèle gelé M8, train seul")
    assert PROTOCOL["périmètre_principal"] == PRIMARY_SCOPE


def test_refuses_a_second_run(tmp_path, monkeypatch):
    done = tmp_path / "final_test_result.json"
    done.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(final_test, "load_raw", _forbid)
    with pytest.raises(FinalTestAlreadyRun):
        run_final_test(result_path=done)
    assert done.read_text(encoding="utf-8") == "{}"


def test_aborts_before_reading_if_freeze_is_broken(tmp_path, monkeypatch):
    def broken(_record):
        raise FreezeViolation("stockvisible/model.py modifié après le gel")

    monkeypatch.setattr(final_test, "check_freeze", broken)
    monkeypatch.setattr(final_test, "load_raw", _forbid)
    target = tmp_path / "final_test_result.json"
    with pytest.raises(FreezeViolation):
        run_final_test(result_path=target)
    assert not target.exists()


def test_evaluate_sealed_on_synthetic_periods():
    splits = chronological_split(make_valid_frame(n_series=3, n_days=90, seed=12))
    test = splits.test.open(UNSEAL_PHRASE)  # SYNTHÉTIQUE : aucune donnée réelle
    history = pd.concat([splits.train, splits.validation], ignore_index=True)
    table = evaluate_sealed(history, test, fit_model(splits.train), label="test")
    primary = table[table["périmètre"] == PRIMARY_SCOPE]
    assert set(primary["modèle"]) == {"B0 (glissant)", "B1 (glissant)", "ML (glissant)"}
    assert primary["n"].nunique() == 1 and primary["n"].iloc[0] > 0
    assert set(table["split"]) == {"test"} and table["dt_min"].iloc[0] == test["dt"].min()


def test_dress_rehearsal_on_validation_reproduces_frozen_metrics(dev):
    """Répétition générale : même fonction, validation en guise de test → chiffres M8 exacts."""
    freeze = load_freeze()
    table = evaluate_sealed(dev.train, dev.validation, fit_model(dev.train), label="validation")
    primary = table[table["périmètre"] == PRIMARY_SCOPE].set_index("modèle")
    for name, m in freeze["métriques_validation_périmètre_principal"].items():
        assert primary.at[name, "mae"] == pytest.approx(m["mae"], abs=1e-9), name
        assert primary.at[name, "bias"] == pytest.approx(m["bias"], abs=1e-9), name
        assert int(primary.at[name, "n"]) == m["n"]


# ---------------------------------------------------------------- scellé (après l'exécution unique)

sealed = pytest.mark.skipif(not RESULT_PATH.exists(), reason="test final pas encore exécuté")


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


@sealed
def test_result_is_a_single_opening_of_the_frozen_engine(result):
    freeze = load_freeze()
    assert result["ouvertures_du_test"] == 1
    assert result["moteur_gelé"] == freeze["moteur"]
    assert result["période_test"] == ["2024-06-11", "2024-06-25"]
    assert result["protocole"] == PROTOCOL
    assert {r["split"] for r in result["table_test"]} == {"test"}
    ns = {result["résultat_moteur_test"]["n"], *(m["n"] for m in result["contexte_test"].values())}
    assert len(ns) == 1  # moteur et contexte sur les mêmes cellules


@sealed
def test_nothing_changed_after_the_final_test(result):
    prov = result["provenance"]
    check_freeze(load_freeze())
    assert sha256_file(FREEZE_PATH) == prov["engine_freeze_sha256"]
    assert sha256_file(Path(final_test.__file__)) == prov["final_test_py_sha256"]
    assert sha256_file(RAW_DIR / "train.parquet") == prov["train_parquet_sha256"]


@sealed
def test_final_test_cannot_run_twice():
    with pytest.raises(FinalTestAlreadyRun):
        run_final_test()
