"""Socle partagé par les 3 écrans : données dev en cache, formats de tableaux, artefacts JSON."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from stockvisible.data import PROJECT_ROOT, RAW_DIR
from stockvisible.splits import DevSplits, load_dev

FREEZE_FILE = PROJECT_ROOT / "engine_freeze.json"
FINAL_FILE = PROJECT_ROOT / "logs" / "final_test_result.json"
SELECTED_SERIES = "serie_choisie"  # clé de session non liée à un widget : survit aux changements d'écran

TABLE_FORMAT = {
    "Couverture (%)": st.column_config.NumberColumn(format="%.1f"),
    "Heures évaluées": st.column_config.NumberColumn(format="%d"),
    "MAE": st.column_config.NumberColumn(format="%.4f"),
    "Biais": st.column_config.NumberColumn(format="%.4f"),
}


@st.cache_resource(show_spinner="Chargement des données dev…")
def dev_data() -> DevSplits:
    return load_dev()


def require_dev() -> DevSplits:
    try:
        return dev_data()
    except FileNotFoundError as exc:
        st.error(f"Données locales absentes : {exc}")
        st.stop()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def data_manifest() -> dict | None:
    return read_json(RAW_DIR / "manifest.json")
