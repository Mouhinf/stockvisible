"""Écran 2 — Comprendre : ventes, disponibilité et prévisions de référence d'une série."""

import streamlit as st

from stockvisible.baselines import B1_MIN_OBS, b0_forecast, b1_forecast
from ui.charts import sales_availability_figure
from ui.common import SELECTED_SERIES, TABLE_FORMAT, require_dev
from ui.components import (
    abstention_summary,
    comparison_table,
    series_frames,
    series_keys,
    series_label,
    series_summary,
)

st.title("Comprendre")
dev = require_dev()
keys = series_keys(dev)
st.caption(
    f"Données RÉELLES FreshRetailNet-50K, {len(keys)} séries, ventes normalisées sans unité. "
    f"Train {dev.train['dt'].min()} → {dev.train['dt'].max()}, validation "
    f"{dev.validation['dt'].min()} → {dev.validation['dt'].max()}. Période test non affichée."
)

current = st.session_state.get(SELECTED_SERIES, keys[0])
key = st.selectbox("Série", keys, index=keys.index(current), format_func=series_label)
st.session_state[SELECTED_SERIES] = key
train, validation = series_frames(dev, key)
forecasts = [b0_forecast(train, validation), b1_forecast(train, validation)]
summary = series_summary(train, validation)

st.subheader("Ventes et disponibilité")
st.write(
    f"{summary['jours']} jours · heures en rupture (6 h–22 h) : "
    f"{summary['part_heures_rupture_6_22']:.1%} · jours avec au moins une rupture : "
    f"{summary['jours_avec_rupture_6_22']}"
)
st.plotly_chart(sales_availability_figure(train, validation, forecasts))
abstention = abstention_summary(forecasts[1])
if abstention["niveau"] == "totale":
    st.error(abstention["message"])
elif abstention["niveau"] == "partielle":
    st.warning(abstention["message"])

st.subheader("B0 et B1 sur la validation, pour cette série")
st.dataframe(comparison_table(validation, forecasts), hide_index=True, column_config=TABLE_FORMAT)
st.caption(
    "B0 : médiane des ventes au même créneau (même jour de semaine, même heure) sur le train, "
    "ruptures comprises. B1 : même calcul limité aux heures déclarées disponibles ; B1 s'abstient "
    f"(trou dans la courbe) s'il a moins de {B1_MIN_OBS} observations comparables. Le moteur gelé "
    "(ML) est évalué sur l'écran Vérifier."
)
