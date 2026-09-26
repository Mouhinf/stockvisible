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
    assert len(at.dataframe) == 2 and len(at.selectbox[0].options) == 500
    assert len(at.get("plotly_chart")) == 1

    # set_value (valeur brute) et non select_index : AppTest re-formate le libellé déjà formaté.
    at.selectbox[0].set_value(series_keys(dev)[250]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == series_keys(dev)[250]
    shown = at.dataframe[0].value
    assert set(shown["Modèle"]) == {"B0", "B1"} and (shown["Heures évaluées"] > 0).all()
