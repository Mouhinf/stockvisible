"""StockVisible — routeur des 3 écrans, dans l'ordre : Vérifier → Comprendre → Acheter.
Données dev uniquement ; le test final n'est affiché que via son résultat scellé."""

from pathlib import Path

import streamlit as st

from ui.theme import css

FAVICON = Path(__file__).parent / "ui" / "assets" / "galsen-favicon.png"  # icône officielle de galsentechnologie.com

st.set_page_config(page_title="StockVisible", page_icon=str(FAVICON), layout="wide")
st.markdown(css(), unsafe_allow_html=True)

pages = [
    st.Page("ui/pages/verifier.py", title="Vérifier", default=True),  # servi à la racine "/"
    st.Page("ui/pages/comprendre.py", title="Comprendre", url_path="comprendre"),
    st.Page("ui/pages/acheter.py", title="Acheter", url_path="acheter"),
]
st.navigation(pages, position="top").run()
