"""Sélection du moteur et gel (M8).

1. select_engine() refuse toute table qui n'est pas de la validation ou qui atteint la période test.
2. Sur les données RÉELLES, toute la chaîne de sélection tourne sous garde-fous (test scellé
   inconstructible) et sous espions : aucune donnée datée >= début du test n'atteint le calcul.
3. Le gel (engine_freeze.json) correspond au recalcul et au code actuel.
"""

from __future__ import annotations

import copy
import shutil

import pandas as pd
import pytest

from stockvisible import baselines, evaluation, features, selection
from stockvisible.evaluation import PRIMARY_SCOPE, ReservedPeriodError, select_engine
from stockvisible.selection import (
    FREEZE_PATH,
    FROZEN_FILES,
    FreezeViolation,
    check_freeze,
    load_freeze,
)
from stockvisible.splits import DevSplits, dev_split, reserved_period_start
from tests._synthetic import make_valid_frame

TEST_START = "2024-06-11"


def _table(split="validation", dt_max="2024-06-10", mae_ml=0.040, mae_b1=0.050):
    rows = [
        {"périmètre": PRIMARY_SCOPE, "modèle": "ML (glissant)", "n": 100, "mae": mae_ml},
        {"périmètre": PRIMARY_SCOPE, "modèle": "B1 (glissant)", "n": 100, "mae": mae_b1},
    ]
    return pd.DataFrame(rows).assign(split=split, dt_min="2024-05-27", dt_max=dt_max)


# ---------------------------------------------------------------- garde-fous de select_engine


def test_select_engine_accepts_a_validation_table():
    assert select_engine(_table(), test_start=TEST_START)["moteur"] == "ML"
    assert select_engine(_table(mae_ml=0.06), test_start=TEST_START)["moteur"] == "B1"


@pytest.mark.parametrize(
    "table, reason",
    [
        (_table(split="test"), "autre chose que la validation"),
        (pd.concat([_table(), _table(split="test")]), "autre chose que la validation"),
        (_table(dt_max="2024-06-11"), "période test"),
        (_table(dt_max="2024-06-25"), "période test"),
        (_table().drop(columns=["dt_max"]), "provenance absente"),
        (_table().drop(columns=["split"]), "provenance absente"),
    ],
    ids=["étiquette_test", "mélange_validation_test", "premier_jour_test", "dernier_jour_test",
         "sans_dates", "sans_étiquette"],
)
def test_select_engine_refuses_test_data(table, reason):
    with pytest.raises(ReservedPeriodError, match=reason):
        select_engine(table, test_start=TEST_START)


def test_reserved_period_start_is_derived_from_dev_only():
    dev = dev_split(make_valid_frame(n_series=2, n_days=75, seed=1))  # 2024-01-01 + 75 jours
    assert not hasattr(dev, "test") and set(vars(dev)) == {"train", "validation"}
    assert reserved_period_start(dev) == "2024-03-16"
    overflow = DevSplits(train=dev.train, validation=dev.validation.assign(dt="2024-03-16"))
    with pytest.raises(ValueError, match="déborde"):
        reserved_period_start(overflow)


# ---------------------------------------------------------------- chaîne réelle sous espions


@pytest.fixture(scope="module")
def selection_run(dev):
    """run_selection sur le dev RÉEL. `dev` (conftest) interdit déjà de construire le test scellé ;
    on espionne en plus chaque frame qui atteint un calcul et la table remise à select_engine."""
    seen_dates: list[str] = []
    seen_tables: list[tuple[pd.DataFrame, str]] = []
    real_matrix = baselines.hourly_matrix
    real_select = selection.select_engine

    def spy_matrix(df, col):
        if "dt" in df.columns and len(df):
            seen_dates.append(str(df["dt"].max()))
        return real_matrix(df, col)

    def spy_select(table, **kwargs):
        seen_tables.append((table.copy(), kwargs["test_start"]))
        return real_select(table, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        for module in (baselines, features, evaluation):
            mp.setattr(module, "hourly_matrix", spy_matrix)
        mp.setattr(selection, "select_engine", spy_select)
        decision, table, model, start = selection.run_selection(dev)
    return {"decision": decision, "table": table, "model": model, "start": start,
            "dates": seen_dates, "tables": seen_tables}


def test_select_engine_never_receives_test_data(selection_run):
    assert selection_run["start"] == TEST_START
    assert len(selection_run["tables"]) == 1
    table, start = selection_run["tables"][0]
    assert start == TEST_START
    assert set(table["split"]) == {"validation"}
    assert table["dt_min"].min() == "2024-05-27" and table["dt_max"].max() == "2024-06-10"


def test_no_frame_dated_in_the_test_period_reaches_any_computation(selection_run):
    dates = selection_run["dates"]
    assert len(dates) > 50  # l'espion a bien vu passer les calculs (variables, B0/B1, métriques)
    assert max(dates) == "2024-06-10" < TEST_START


def test_model_used_for_selection_was_trained_on_train_only(selection_run):
    model = selection_run["model"]
    assert (model.train_start, model.train_end) == ("2024-03-28", "2024-05-26")


# ---------------------------------------------------------------- gel


@pytest.fixture(scope="module")
def frozen():
    if not FREEZE_PATH.exists():
        pytest.fail("engine_freeze.json absent : lancer `python -m stockvisible.selection`")
    return load_freeze()


def test_freeze_matches_a_fresh_selection_run(frozen, selection_run):
    decision = selection_run["decision"]
    assert frozen["moteur"] == decision["moteur"]
    assert frozen["décision"]["n"] == decision["n"]
    assert frozen["décision"]["mae_ml"] == pytest.approx(decision["mae_ml"], abs=1e-9)
    assert frozen["décision"]["mae_b1"] == pytest.approx(decision["mae_b1"], abs=1e-9)
    primary = selection_run["table"].query("périmètre == @PRIMARY_SCOPE").set_index("modèle")
    for name, m in frozen["métriques_validation_périmètre_principal"].items():
        assert m["mae"] == pytest.approx(primary.at[name, "mae"], abs=1e-9)
        assert m["bias"] == pytest.approx(primary.at[name, "bias"], abs=1e-9)


def test_frozen_engine_follows_the_rule(frozen):
    d = frozen["décision"]
    assert frozen["moteur"] == ("ML" if d["mae_ml"] < d["mae_b1"] else "B1")
    assert frozen["périodes"] == {
        "train": ["2024-03-28", "2024-05-26"],
        "validation": ["2024-05-27", "2024-06-10"],
        "début_test_réservé": TEST_START,
    }


def test_frozen_code_and_specs_are_unchanged(frozen):
    check_freeze(frozen)


def test_freeze_detects_a_modified_engine_file(frozen, tmp_path):
    for rel in FROZEN_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(selection.PROJECT_ROOT / rel, tmp_path / rel)
    check_freeze(frozen, root=tmp_path)
    with open(tmp_path / "stockvisible/features.py", "a", encoding="utf-8") as fh:
        fh.write("\n# modification après gel\n")
    with pytest.raises(FreezeViolation, match="features.py"):
        check_freeze(frozen, root=tmp_path)


def test_freeze_detects_a_changed_spec(frozen):
    altered = copy.deepcopy(frozen)
    altered["modèle"]["spec"]["quantile"] = 0.6
    with pytest.raises(FreezeViolation, match="ModelSpec"):
        check_freeze(altered)


def test_frozen_training_data_is_unchanged(frozen):
    from stockvisible.data import RAW_DIR, sha256_file

    assert sha256_file(RAW_DIR / "train.parquet") == frozen["données"]["train_parquet_sha256"]
