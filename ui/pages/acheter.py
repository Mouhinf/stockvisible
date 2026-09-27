"""Écran 3 — Acheter : panier optimal sous budget, hypothèses explicites."""

import streamlit as st

from stockvisible.allocation import MAX_PRODUCTS, optimal_basket
from ui.common import SELECTED_SERIES, require_dev
from ui.components import (
    basket_table,
    budget_upper_bound,
    daily_demand_scenarios,
    default_hypotheses,
    products_from_hypotheses,
    series_keys,
    series_label,
)

st.title("Acheter")
dev = require_dev()
keys = series_keys(dev)
start = keys.index(st.session_state.get(SELECTED_SERIES, keys[0]))
chosen = st.multiselect(
    "Produits du panier (3 au maximum)",
    keys,
    default=keys[start : start + MAX_PRODUCTS],
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
    f"{scenario_days[0]} → {scenario_days[-1]}. Demande journalière ESTIMÉE (unités normalisées) "
    "avec B1, pas avec le moteur ML gelé : ventes observées aux heures disponibles, estimation B1 "
    "aux heures en rupture entre 6 h et 22 h. Le panier maximise la demande couverte moyenne ; à "
    f"valeur égale, le moins cher. {basket.combinations_evaluated} combinaisons de lots évaluées."
)
