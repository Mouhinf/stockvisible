"""StockVisible — écran unique (F2 REVEAL + F3 VERIFY). Données dev uniquement : le test réservé
n'est jamais chargé ni affiché."""

import streamlit as st

from stockvisible.allocation import MAX_PRODUCTS, optimal_basket
from stockvisible.baselines import B1_MIN_OBS, b0_forecast, b1_forecast
from stockvisible.splits import DevSplits, load_dev
from ui.charts import sales_availability_figure
from ui.components import (
    basket_table,
    budget_upper_bound,
    comparison_table,
    daily_demand_scenarios,
    default_hypotheses,
    products_from_hypotheses,
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

# ---------------------------------------------------------------- F4 DECIDE

st.subheader("Panier sous budget")
chosen = st.multiselect(
    "Produits du panier (3 au maximum)",
    keys,
    default=keys[keys.index(key): keys.index(key) + 3],
    format_func=series_label,
    max_selections=MAX_PRODUCTS,
)
if not chosen:
    st.info("Choisir au moins un produit pour calculer un panier.")
    st.stop()

st.caption(
    "Le jeu de données ne contient ni coût ni niveau de stock : les valeurs ci-dessous sont des "
    "HYPOTHÈSES à saisir (valeurs par défaut arbitraires). Une cellule de coût vide = coût inconnu, "
    "le produit n'est alors jamais acheté."
)
hypotheses = st.data_editor(
    default_hypotheses(chosen),
    hide_index=True,
    disabled=["Série"],
    key="hypotheses-" + "|".join(series_label(k) for k in chosen),
    column_config={
        "Coût unitaire (hypothèse)": st.column_config.NumberColumn(min_value=0.01, format="%.2f"),
        "Taille de lot": st.column_config.NumberColumn(min_value=0.1, format="%.2f", required=True),
        "Stock actuel (hypothèse)": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
    },
)
scenario_days, scenarios = daily_demand_scenarios(dev, chosen)
try:
    products = products_from_hypotheses(hypotheses)
    upper = max(budget_upper_bound(products, scenarios), 1.0)
    budget = st.slider("Budget (mêmes unités que les coûts saisis)", 0.0, upper, upper / 2, step=0.1)
    basket = optimal_basket(products, budget, scenarios)
except ValueError as exc:
    st.error(f"Hypothèses invalides : {exc}")
    st.stop()

for name, reason in basket.excluded.items():
    st.warning(f"{name} : {reason}")
cols = st.columns(3)
cols[0].metric("Coût du panier", f"{basket.cost:.2f}", help=f"Budget : {budget:.2f}")
cols[1].metric(
    "Demande couverte espérée / jour",
    f"{basket.expected_covered:.2f}",
    delta=f"{basket.gain:+.2f} vs sans achat",
)
cols[2].metric("Manque espéré / jour", f"{basket.expected_shortfall:.2f}")
st.dataframe(basket_table(basket), hide_index=True)
st.caption(
    f"{len(scenario_days)} scénarios équiprobables = les jours de validation "
    f"{scenario_days[0]} → {scenario_days[-1]}. Demande journalière ESTIMÉE (unités normalisées) : "
    "ventes observées aux heures disponibles, estimation B1 aux heures en rupture entre 6 h et 22 h. "
    "Le panier maximise la demande couverte moyenne ; à valeur égale, le moins cher. "
    f"{basket.combinations_evaluated} combinaisons de lots évaluées exhaustivement."
)
