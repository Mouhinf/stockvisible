"""F5 ACT (validation + export CSV/JSON du panier) et import utilisateur — M12, M14."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from stockvisible.allocation import Product, optimal_basket
from stockvisible.exports import (
    CSV_COLUMNS,
    DEMAND_SOURCE,
    basket_fingerprint,
    basket_view,
    export_csv,
    export_frame,
    export_json,
    validate_basket,
    validated_view,
)
from stockvisible.validation import read_user_bytes, validate
from ui.components import import_report, view_table

FIXTURES = Path(__file__).parent / "e2e" / "fixtures"
APP = Path(__file__).resolve().parents[1] / "app.py"
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
CONTEXT = {"moteur_gele_projet": "ML (gelé)", "donnees": "jeu X", "scenarios": "15 jours"}


def _case(names=("A", "B"), budget=3.0):
    products = [Product(n, unit_cost=1.0, lot_size=0.5, stock=0.0) for n in names]
    basket = optimal_basket(products, budget, [[2.0] * len(names), [1.0] * len(names)])
    return products, basket, budget


def _validated(**kw):
    products, basket, budget = _case(**kw)
    return validate_basket(basket, products, budget, now=NOW, context=CONTEXT)


# ---------------------------------------------------------------- validation


def test_validation_records_time_fingerprint_and_context():
    v = _validated()
    assert v.validated_at == "2026-09-27T12:00:00+00:00"
    assert v.fingerprint == basket_fingerprint(v.basket, v.products, v.budget)
    assert v.context == CONTEXT


def test_fingerprint_changes_with_budget_hypotheses_or_lots():
    products, basket, budget = _case()
    ref = basket_fingerprint(basket, products, budget)
    assert basket_fingerprint(basket, products, budget + 1) != ref
    assert basket_fingerprint(basket, [Product("A", 2.0, 0.5, 0.0), products[1]], budget) != ref
    other = optimal_basket(products, 1.0, [[2.0, 2.0], [1.0, 1.0]])
    assert basket_fingerprint(other, products, budget) != ref


def test_over_budget_basket_cannot_be_validated():
    products, basket, _ = _case(budget=3.0)
    with pytest.raises(ValueError, match="budget"):
        validate_basket(basket, products, budget=basket.cost - 0.5)


# ---------------------------------------------------------------- une seule source : basket_view


def test_view_carries_hypotheses_engines_timestamp_and_provenance():
    view = validated_view(_validated())
    assert view["horodatage_utc"] == "2026-09-27T12:00:00+00:00" and len(view["empreinte_panier"]) == 16
    assert view["moteur"] == {"demande_panier": DEMAND_SOURCE, "moteur_gele_projet": "ML (gelé)"}
    assert [h["serie"] for h in view["hypotheses"]] == ["A", "B"]
    assert view["provenance"]["donnees"] == "jeu X" and view["provenance"]["scenarios"] == "15 jours"
    assert "hypothèses" in view["provenance"]["hypotheses_saisies"]


def test_unvalidated_view_has_no_timestamp():
    products, basket, budget = _case()
    view = basket_view(basket, products, budget, CONTEXT)
    assert view["horodatage_utc"] is None and view["empreinte_panier"] is None


def test_csv_and_json_are_the_same_view():
    v = _validated()
    view = validated_view(v)
    as_json = json.loads(export_json(v))
    assert as_json == json.loads(json.dumps(view))
    csv = pd.read_csv(io.BytesIO(export_csv(v)), keep_default_na=False)
    assert list(csv.columns) == CSV_COLUMNS
    assert csv["serie"].tolist() == [line["serie"] for line in view["panier"]]
    assert csv["lots"].tolist() == [line["lots"] for line in view["panier"]]
    for key, value in view["indicateurs"].items():
        assert set(csv[key]) == {value}
    assert set(csv["horodatage_utc"]) == {view["horodatage_utc"]}
    assert set(csv["moteur_gele_projet"]) == {"ML (gelé)"}


@pytest.mark.parametrize("name", ['=HYPERLINK("x")', "+1", "-2", "@SUM(A1)", "\tcmd", "  =1+1"])
def test_export_neutralises_spreadsheet_formulas(name):
    products = [Product(name, 1.0, 0.5, 0.0)]
    basket = optimal_basket(products, 1.0, [[1.0]])
    cell = export_frame(validate_basket(basket, products, 1.0, now=NOW))["serie"].iloc[0]
    assert cell == "'" + name


def test_unknown_cost_is_exported_empty_not_zero():
    products = [Product("A", None, 0.5, 0.0), Product("B", 1.0, 0.5, 0.0)]
    basket = optimal_basket(products, 1.0, [[1.0, 1.0]])
    v = validate_basket(basket, products, 1.0, now=NOW)
    frame = export_frame(v).set_index("serie")
    assert frame.at["A", "cout_unitaire_hypothese"] == "" and frame.at["A", "lots"] == 0
    assert json.loads(export_json(v))["hypotheses"][0]["cout_unitaire_hypothese"] is None


# ---------------------------------------------------------------- import


def test_import_fixtures_are_labelled_synthetic_and_behave():
    ok = read_user_bytes("f.csv", (FIXTURES / "import_conforme_SYNTHETIQUE.csv").read_bytes())
    bad = read_user_bytes("f.csv", (FIXTURES / "import_corrompu_SYNTHETIQUE.csv").read_bytes())
    good = import_report(ok, validate(ok))
    assert good["conforme"] and good["séries"] == 2 and good["lignes"] == 10
    broken = import_report(bad, validate(bad))
    assert not broken["conforme"]
    assert {"DUPLICATE_DATE", "OUT_OF_RANGE"} <= set(broken["erreurs"]["Code"])


# ---------------------------------------------------------------- écran = export


def _screen(at) -> dict:
    metrics = {m.label: m.value for m in at.metric}
    return {
        "metrics": metrics,
        "delta": next(m.proto.delta for m in at.metric if m.label == "Demande couverte espérée / jour"),
        "table": at.dataframe[-1].value.reset_index(drop=True),
        "success": next(s.value for s in at.success if "Panier validé" in s.value),
    }


def test_export_is_strictly_what_the_screen_shows(dev):
    """Sur les données RÉELLES : chaque valeur affichée sur Acheter se retrouve à l'identique dans
    les exports CSV et JSON du panier validé (et inversement pour le tableau du panier)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("ui/pages/acheter.py").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    validated = at.session_state["panier_valide"]
    screen = _screen(at)
    view = json.loads(export_json(validated))
    csv = pd.read_csv(io.BytesIO(export_csv(validated)), keep_default_na=False)
    ind = view["indicateurs"]

    assert screen["metrics"]["Coût du panier"] == f"{ind['cout_panier']:.2f}"
    assert screen["metrics"]["Demande couverte espérée / jour"] == f"{ind['demande_couverte_esperee_jour']:.2f}"
    assert screen["delta"] == f"{ind['gain_vs_sans_achat']:+.2f} vs sans achat"
    assert screen["metrics"]["Manque espéré / jour"] == f"{ind['manque_espere_jour']:.2f}"
    pd.testing.assert_frame_equal(screen["table"], view_table(view), check_dtype=False)
    assert csv["serie"].tolist() == screen["table"]["Série"].tolist()
    assert csv["lots"].tolist() == screen["table"]["Lots achetés"].tolist()
    assert csv["quantite_unites_normalisees"].tolist() == screen["table"]["Quantité (unités normalisées)"].tolist()
    assert view["horodatage_utc"] in screen["success"] and view["empreinte_panier"] in screen["success"]
    assert view["moteur"]["moteur_gele_projet"].startswith("ML")
    assert "FreshRetailNet-50K" in view["provenance"]["donnees"]
    hyp = at.dataframe[0].value.reset_index(drop=True)  # tableau éditable des hypothèses
    assert hyp["Série"].tolist() == [h["serie"] for h in view["hypotheses"]]
    assert hyp["Coût unitaire (hypothèse)"].tolist() == [h["cout_unitaire_hypothese"] for h in view["hypotheses"]]
    assert hyp["Taille de lot"].tolist() == [h["taille_lot"] for h in view["hypotheses"]]
    assert hyp["Stock actuel (hypothèse)"].tolist() == [h["stock_actuel_hypothese"] for h in view["hypotheses"]]
    assert csv["taille_lot"].tolist() == hyp["Taille de lot"].tolist()


def test_acheter_validation_then_export_and_invalidation(dev):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.switch_page("ui/pages/acheter.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.get("download_button")  # pas d'export avant validation
    at.button[0].click().run()
    assert any("Panier validé" in s.value for s in at.success)
    assert len(at.get("download_button")) == 2  # CSV + JSON
    at.slider[0].set_value(0.0).run()  # le panier change → validation caduque
    assert not at.get("download_button")
    assert any("a changé" in i.value for i in at.info)
