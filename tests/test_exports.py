"""F5 ACT (validation + export du panier) et import utilisateur — M12."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from stockvisible.allocation import Product, optimal_basket
from stockvisible.exports import (
    EXPORT_COLUMNS,
    basket_fingerprint,
    export_csv,
    export_frame,
    validate_basket,
)
from stockvisible.validation import read_user_bytes, validate
from ui.components import import_report

FIXTURES = Path(__file__).parent / "e2e" / "fixtures"
APP = Path(__file__).resolve().parents[1] / "app.py"
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _case(names=("A", "B"), budget=3.0):
    products = [Product(n, unit_cost=1.0, lot_size=0.5, stock=0.0) for n in names]
    basket = optimal_basket(products, budget, [[2.0] * len(names), [1.0] * len(names)])
    return products, basket, budget


def test_validation_records_time_and_fingerprint():
    products, basket, budget = _case()
    v = validate_basket(basket, products, budget, now=NOW)
    assert v.validated_at == "2026-09-27T12:00:00+00:00"
    assert v.fingerprint == basket_fingerprint(basket, products, budget)


def test_fingerprint_changes_with_budget_hypotheses_or_lots():
    products, basket, budget = _case()
    ref = basket_fingerprint(basket, products, budget)
    assert basket_fingerprint(basket, products, budget + 1) != ref
    changed = [Product("A", 2.0, 0.5, 0.0), products[1]]
    assert basket_fingerprint(basket, changed, budget) != ref
    other = optimal_basket(products, 1.0, [[2.0, 2.0], [1.0, 1.0]])
    assert basket_fingerprint(other, products, budget) != ref


def test_over_budget_basket_cannot_be_validated():
    products, basket, _ = _case(budget=3.0)
    with pytest.raises(ValueError, match="budget"):
        validate_basket(basket, products, budget=basket.cost - 0.5)


def test_export_csv_round_trip_carries_hypotheses_and_provenance():
    products, basket, budget = _case()
    frame = pd.read_csv(io.BytesIO(export_csv(validate_basket(basket, products, budget, now=NOW))))
    assert list(frame.columns) == EXPORT_COLUMNS
    assert frame["lots"].tolist() == list(basket.lots)
    assert frame["cout_total_panier"].iloc[0] == pytest.approx(basket.cost)
    assert (frame["cout_ligne"].sum()) == pytest.approx(basket.cost)
    assert set(frame["source_demande"]) == {"B1 estimé (ventes disponibles + B1 aux heures en rupture 6-22 h)"}
    assert set(frame["valide_le_utc"]) == {"2026-09-27T12:00:00+00:00"}


@pytest.mark.parametrize("name", ["=HYPERLINK(\"x\")", "+1", "-2", "@SUM(A1)", "\tcmd"])
def test_export_neutralises_spreadsheet_formulas(name):
    products = [Product(name, 1.0, 0.5, 0.0)]
    basket = optimal_basket(products, 1.0, [[1.0]])
    cell = export_frame(validate_basket(basket, products, 1.0, now=NOW))["serie"].iloc[0]
    assert cell == "'" + name


def test_unknown_cost_is_exported_empty_not_zero():
    products = [Product("A", None, 0.5, 0.0), Product("B", 1.0, 0.5, 0.0)]
    basket = optimal_basket(products, 1.0, [[1.0, 1.0]])
    frame = export_frame(validate_basket(basket, products, 1.0, now=NOW)).set_index("serie")
    assert frame.at["A", "cout_unitaire_hypothese"] == "" and frame.at["A", "lots"] == 0


# ---------------------------------------------------------------- import


def test_import_fixtures_are_labelled_synthetic_and_behave():
    ok = read_user_bytes("f.csv", (FIXTURES / "import_conforme_SYNTHETIQUE.csv").read_bytes())
    bad = read_user_bytes("f.csv", (FIXTURES / "import_corrompu_SYNTHETIQUE.csv").read_bytes())
    good = import_report(ok, validate(ok))
    assert good["conforme"] and good["séries"] == 2 and good["lignes"] == 10
    broken = import_report(bad, validate(bad))
    assert not broken["conforme"]
    assert {"DUPLICATE_DATE", "OUT_OF_RANGE"} <= set(broken["erreurs"]["Code"])


# ---------------------------------------------------------------- écran


def test_acheter_validation_then_export_and_invalidation(dev):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("ui/pages/acheter.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.get("download_button")  # pas d'export avant validation
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Panier validé" in s.value for s in at.success)
    assert len(at.get("download_button")) == 1
    at.slider[0].set_value(0.0).run()  # le panier change → validation caduque
    assert not at.get("download_button")
    assert any("a changé" in i.value for i in at.info)
