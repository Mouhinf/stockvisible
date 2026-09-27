"""Écran 1 — Vérifier : la preuve d'abord. Tout chiffre est lu dans un artefact versionné."""

import streamlit as st

from stockvisible.selection import FreezeViolation, check_freeze
from stockvisible.uncertainty import load_report
from ui.common import FINAL_FILE, FREEZE_FILE, data_manifest, dev_data, read_json
from ui.components import (
    guarantee_label,
    interval_summary,
    problem_banner,
    problem_summary,
    proof_guarantees,
    proof_headline,
    proof_limits,
    proof_table,
)

# Bandeau d'accueil : le problème en une phrase, chiffré sur les données (jamais tapé).
try:
    summary = problem_summary(dev_data())
except FileNotFoundError:
    summary = None
with st.container(border=True):
    st.markdown("**StockVisible** — voir la demande que les ruptures cachent, puis décider quoi racheter.")
    if summary is not None:
        st.markdown(problem_banner(summary))

st.title("Vérifier")
freeze, final = read_json(FREEZE_FILE), read_json(FINAL_FILE)
if freeze is None or final is None:
    st.warning("Preuve incomplète : engine_freeze.json ou logs/final_test_result.json absent.")
    st.stop()

st.markdown(f"**{proof_headline(freeze, final)}**")

st.subheader("Erreurs mesurées, validation puis test final")
st.dataframe(
    proof_table(freeze, final),
    hide_index=True,
    column_config={
        c: st.column_config.NumberColumn(format="%.4f")
        for c in ("MAE validation", "Biais validation", "MAE test final", "Biais test final")
    }
    | {"Couverture test (%)": st.column_config.NumberColumn(format="%.1f")},
)
st.caption(
    f"Prévisions horaires glissantes à J+1, heures déclarées disponibles, mêmes heures pour les "
    f"trois modèles. Validation {freeze['périodes']['validation'][0]} → "
    f"{freeze['périodes']['validation'][1]} ; test final {final['période_test'][0]} → "
    f"{final['période_test'][1]}. MAE = erreur absolue moyenne ; biais = moyenne(prévision − réel), "
    "négatif = sous-prévision. Règle de choix fixée avant tout résultat : "
    f"{freeze['règle']}"
)

st.subheader("Incertitude du moteur")
uncertainty = interval_summary(load_report())
if uncertainty["niveau"] == "non_calibré":
    st.warning(uncertainty["message"])
else:
    st.metric(
        "Couverture empirique de l'intervalle 80 %",
        f"{uncertainty['couverture']:.1%}",
        delta=f"{uncertainty['couverture'] - uncertainty['nominal']:+.1%} vs nominal",
        delta_color="off",
    )
    st.caption(uncertainty["message"])

st.subheader("Garanties")
try:
    check_freeze(freeze)
    intact = True
except FreezeViolation:
    intact = False
for i, (ok, text) in enumerate(proof_guarantees(freeze, final, intact)):
    st.markdown(f"- **{guarantee_label(i, ok)}** — {text}")

st.subheader("Limites")
for text in proof_limits(freeze, final, load_report()):
    st.markdown(f"- {text}")

manifest = data_manifest()
if manifest:
    st.caption(
        f"Données : {manifest['dataset']} (révision {manifest['revision'][:8]}, licence "
        f"{manifest['license']}), sous-ensemble déterministe de {manifest['subset_spec']['n_series']} "
        "séries RÉELLES."
    )
