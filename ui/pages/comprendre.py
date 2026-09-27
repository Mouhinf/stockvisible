"""Écran 2 — Comprendre : ventes, disponibilité et prévisions de référence d'une série."""

import streamlit as st

from stockvisible.baselines import B1_MIN_OBS, b0_forecast, b1_forecast
from stockvisible.validation import MAX_BYTES, InputRejected, read_user_bytes, validate
from ui.charts import sales_availability_figure
from ui.common import SELECTED_SERIES, TABLE_FORMAT, require_dev
from ui.components import (
    abstention_summary,
    comparison_table,
    import_report,
    observed_vs_estimated,
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
gap = observed_vs_estimated(dev, key)
left, right = st.columns(2)
left.metric("Ventes observées / jour (validation)", f"{gap['ventes_observées']:.2f}")
right.metric(
    "Demande estimée / jour (validation)",
    f"{gap['demande_estimée']:.2f}",
    delta=None if gap["écart"] is None else f"{gap['écart']:+.0%} vs ventes observées",
    delta_color="off",
)
st.caption(
    f"Moyenne sur les {gap['jours']} jours de validation, unités normalisées. Demande ESTIMÉE = ventes "
    "des heures disponibles + prévision B1 aux heures en rupture de 6 h à 22 h : c'est une "
    "estimation, pas une mesure de la demande perdue. C'est elle qu'utilise l'écran Acheter."
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

st.subheader("Importer vos données")
st.caption(
    "CSV ou Parquet au schéma FreshRetailNet-50K (listes horaires en JSON dans un CSV), "
    f"{MAX_BYTES // (1024 * 1024)} Mo au plus. Le fichier est lu sans aucune exécution de contenu, "
    "puis contrôlé par le contrat de données. Les données importées sont celles de l'utilisateur : "
    "elles ne sont pas mélangées au jeu de référence."
)
upload = st.file_uploader("Fichier à contrôler", type=["csv", "parquet"])
if upload is not None:
    try:
        imported = read_user_bytes(upload.name, upload.getvalue())
        result = import_report(imported, validate(imported))
    except InputRejected as exc:
        st.error("Fichier refusé avant lecture. Détail :")
        st.text(str(exc))  # texte brut : un fragment du fichier n'est jamais interprété en markdown
    except (TypeError, ValueError, RecursionError):
        st.error("Fichier refusé : contenu inattendu, impossible à contrôler par le contrat de données.")
    else:
        if not result["conforme"]:
            st.error(f"Fichier non conforme au contrat de données ({len(result['erreurs'])} type(s) d'erreur) :")
            st.table(result["erreurs"].set_index("Code"))  # tableau HTML : lisible au lecteur d'écran
        else:
            st.success(
                f"Fichier conforme : {result['lignes']} lignes, {result['séries']} série(s), du "
                f"{result['dates'][0]} au {result['dates'][1]}. Données IMPORTÉES par l'utilisateur."
            )
            if result["infos"]:
                st.caption(f"{result['infos']} ligne(s) avec vente pendant une heure en rupture (attendu).")
            st.dataframe(
                result["par_série"],
                hide_index=True,
                column_config={"Heures en rupture 6–22 h (%)": st.column_config.NumberColumn(format="%.1f")},
            )
