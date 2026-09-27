"""StockVisible — routeur des 3 écrans, dans l'ordre : Vérifier → Comprendre → Acheter.
Données dev uniquement ; le test final n'est affiché que via son résultat scellé."""

import streamlit as st

from ui.theme import css

st.set_page_config(page_title="StockVisible", layout="wide")
st.markdown(css(), unsafe_allow_html=True)

pages = [
    st.Page("ui/pages/verifier.py", title="Vérifier", default=True),  # servi à la racine "/"
    st.Page("ui/pages/comprendre.py", title="Comprendre", url_path="comprendre"),
    st.Page("ui/pages/acheter.py", title="Acheter", url_path="acheter"),
]
st.navigation(pages, position="top").run()
