"""StockVisible — écran unique (F2 REVEAL + F3 VERIFY). Données dev uniquement : le test réservé
n'est jamais chargé ni affiché."""

import streamlit as st

from stockvisible.baselines import B1_MIN_OBS, b0_forecast, b1_forecast
from stockvisible.splits import DevSplits, load_dev
from ui.charts import sales_availability_figure
from ui.components import (
    comparison_table,
    series_frames,
    series_keys,
    series_label,
    series_summary,
)

TABLE_FORMAT = {
    "Couverture (%)": st.column_config.NumberColumn(format="%.1f"),
    "Heures évaluées": st.column_config.NumberColumn(format="%d"),
    "MAE": st.column_config.NumberColumn(format="%.4f"),
    "Biais": st.column_config.NumberColumn(format="%.4f"),
}


@st.cache_resource(show_spinner="Chargement des données dev…")
def dev_data() -> DevSplits:
    return load_dev()


@st.cache_resource(show_spinner="Calcul B0 / B1 sur toutes les séries…")
def all_series_table():
    dev = dev_data()
    forecasts = [b0_forecast(dev.train, dev.validation), b1_forecast(dev.train, dev.validation)]
    return comparison_table(dev.validation, forecasts)


st.set_page_config(page_title="StockVisible", layout="wide")
st.title("StockVisible")

try:
    dev = dev_data()
except FileNotFoundError as exc:
    st.error(f"Données locales absentes : {exc}")
    st.stop()

keys = series_keys(dev)
st.caption(
    f"Données RÉELLES FreshRetailNet-50K (CC BY 4.0), sous-ensemble de {len(keys)} séries. "
    "Ventes normalisées par le fournisseur, sans unité monétaire. "
    f"Période affichée : train {dev.train['dt'].min()} → {dev.train['dt'].max()}, "
    f"validation {dev.validation['dt'].min()} → {dev.validation['dt'].max()}. "
    "La période test réservée n'est ni chargée ni affichée."
)

key = st.selectbox("Série", keys, format_func=series_label)
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

st.subheader("B0 vs B1 sur la validation — cette série")
st.dataframe(comparison_table(validation, forecasts), hide_index=True, column_config=TABLE_FORMAT)

st.subheader(f"B0 vs B1 sur la validation — les {len(keys)} séries")
st.dataframe(all_series_table(), hide_index=True, column_config=TABLE_FORMAT)

st.caption(
    "B0 : médiane des ventes au même créneau (même jour de semaine, même heure) sur le train, "
    "ruptures comprises. B1 : même calcul limité aux heures déclarées disponibles ; B1 s'abstient "
    f"(trou dans la courbe) s'il a moins de {B1_MIN_OBS} observations comparables. Erreurs mesurées "
    "uniquement sur les heures disponibles de la validation, où la vente observée est la demande. "
    "MAE = erreur absolue moyenne ; biais = moyenne(prévision − réel), négatif = sous-prévision."
)
