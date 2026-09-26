"""Écran minimal : builders purs sur données SYNTHÉTIQUES + exécution réelle de app.py (dev uniquement)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockvisible.baselines import b0_forecast, b1_forecast
from stockvisible.splits import DevSplits, dev_split
from tests._synthetic import make_valid_frame
from ui.charts import hourly_index, sales_availability_figure
from ui.components import (
    UI_SCOPES,
    comparison_table,
    series_frames,
    series_keys,
    series_label,
    series_summary,
)

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def synthetic_dev() -> DevSplits:
    return dev_split(make_valid_frame(n_series=3, n_days=75, seed=4))


def _one_series(dev: DevSplits):
    key = series_keys(dev)[1]
    train, validation = series_frames(dev, key)
    return key, train, validation, [b0_forecast(train, validation), b1_forecast(train, validation)]


def test_series_selection_helpers(synthetic_dev):
    keys = series_keys(synthetic_dev)
    assert keys == [(10, 100), (11, 101), (12, 102)]
    assert series_label(keys[0]) == "magasin 10 · produit 100"
    _, train, validation, _ = _one_series(synthetic_dev)
    assert len(train) == 60 and len(validation) == 15
    assert set(train["store_id"]) == {11} and train["dt"].is_monotonic_increasing
    assert train["dt"].max() < validation["dt"].min()


def test_series_summary(synthetic_dev):
    _, train, validation, _ = _one_series(synthetic_dev)
    s = series_summary(train, validation)
    assert s["jours"] == 75 and 0 <= s["part_heures_rupture_6_22"] <= 1


def test_hourly_index():
    idx = hourly_index(pd.DataFrame({"dt": ["2024-01-01", "2024-01-02"]}))
    assert len(idx) == 48 and idx[0] == pd.Timestamp("2024-01-01 00:00")
    assert idx[-1] == pd.Timestamp("2024-01-02 23:00")


def test_figure_traces_and_single_y_axis_per_panel(synthetic_dev):
    _, train, validation, forecasts = _one_series(synthetic_dev)
    fig = sales_availability_figure(train, validation, forecasts)
    names = [t.name for t in fig.data]
    assert names == ["Ventes observées", "B0 (prévision validation)", "B1 (prévision validation)", "Disponible"]
    layout = fig.to_plotly_json()["layout"]
    assert not any("overlaying" in v for k, v in layout.items() if k.startswith("yaxis"))
    obs, b0, b1, avail = fig.data
    assert len(obs.x) == 75 * 24 and set(np.unique(avail.y)) <= {0, 1}
    assert pd.Timestamp(b0.x[0]) == pd.Timestamp(validation["dt"].min())
    assert len(b1.y) == 15 * 24


def test_b1_abstention_is_a_gap_not_a_zero(synthetic_dev):
    _, train, validation, forecasts = _one_series(synthetic_dev)
    b1 = forecasts[1]
    b1_trace = sales_availability_figure(train, validation, forecasts).data[2]
    y = np.asarray(b1_trace.y, dtype=float)
    np.testing.assert_array_equal(np.isnan(y), b1.abstained.ravel())
    assert b1_trace.connectgaps is False


def test_comparison_table_shows_only_uncensored_scopes(synthetic_dev):
    _, _, validation, forecasts = _one_series(synthetic_dev)
    table = comparison_table(validation, forecasts)
    assert list(table.columns) == ["Périmètre", "Modèle", "Couverture (%)", "Heures évaluées", "MAE", "Biais"]
    assert set(table["Périmètre"]) == set(UI_SCOPES)
    assert table["Couverture (%)"].between(0, 100).all()


# ---------------------------------------------------------------- app.py réel, dev uniquement


def test_app_runs_without_exception_on_real_dev_data(dev):
    """`dev` (conftest) garde actifs les garde-fous : construire le test scellé fait échouer l'app."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert "RÉELLES" in at.caption[0].value and "ni chargée ni affichée" in at.caption[0].value
    assert len(at.dataframe) == 4 and len(at.selectbox[0].options) == 500  # 2 B0/B1 + hypothèses + panier
    assert len(at.get("plotly_chart")) == 1

    # set_value (valeur brute) et non select_index : AppTest re-formate le libellé déjà formaté.
    at.selectbox[0].set_value(series_keys(dev)[250]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == series_keys(dev)[250]
    shown = at.dataframe[0].value
    assert set(shown["Modèle"]) == {"B0", "B1"} and (shown["Heures évaluées"] > 0).all()


def test_budget_slider_really_recomputes_the_basket(dev):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    slider = at.slider[0]
    upper = slider.max

    def basket_state(budget):
        at.slider[0].set_value(budget).run()
        assert not at.exception, [e.value for e in at.exception]
        cost = float(at.metric[0].value)
        lots = tuple(at.dataframe[3].value["Lots achetés"])
        return cost, lots

    full_cost, full_lots = basket_state(upper)
    half_cost, half_lots = basket_state(round(upper / 2, 1))
    zero_cost, zero_lots = basket_state(0.0)
    assert full_cost <= upper + 1e-9 and half_cost <= upper / 2 + 1e-9
    assert full_lots != half_lots and sum(half_lots) < sum(full_lots)
    assert zero_cost == 0.0 and set(zero_lots) == {0}


def test_scenarios_fill_only_daytime_stockouts_with_b1(synthetic_dev):
    from stockvisible.baselines import b1_forecast, hourly_matrix
    from ui.components import daily_demand_scenarios

    key = series_keys(synthetic_dev)[0]
    train, validation = series_frames(synthetic_dev, key)
    days, scen = daily_demand_scenarios(synthetic_dev, [key])
    assert days == validation["dt"].tolist() and scen.shape == (15, 1)
    sales = hourly_matrix(validation, "hours_sale")
    status = hourly_matrix(validation, "hours_stock_status")
    b1 = b1_forecast(train, validation).pred
    expected = sales.copy()
    day = np.zeros(24, dtype=bool)
    day[6:22] = True
    fill = (status == 1) & day & ~np.isnan(b1)
    expected[fill] = b1[fill]
    np.testing.assert_allclose(scen[:, 0], expected.sum(axis=1))
    assert (scen[:, 0] >= sales.sum(axis=1) - 1e-12).all()  # jamais moins que l'observé


def test_hypotheses_to_products_and_budget_bound():
    from ui.components import (
        budget_upper_bound,
        default_hypotheses,
        products_from_hypotheses,
    )

    table = default_hypotheses([(1, 2), (3, 4)])
    table.loc[1, "Coût unitaire (hypothèse)"] = np.nan
    products = products_from_hypotheses(table)
    assert products[0].unit_cost == 1.0 and products[1].unit_cost is None
    # produit 1 : besoin max 1.2 → 3 lots de 0.5 à 1.0 = 1.5 ; produit 2 sans coût : 0
    assert budget_upper_bound(products, np.array([[1.2, 9.0], [0.4, 1.0]])) == pytest.approx(1.5)
