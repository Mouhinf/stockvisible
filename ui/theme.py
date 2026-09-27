"""Jetons de design — traduction de design-system/MASTER.md (source de vérité).

Thème natif : .streamlit/config.toml (mêmes valeurs, vérifiées par tests/test_design.py).
CSS injecté : uniquement ce que le thème natif ne couvre pas (largeur, chiffres tabulaires,
police système, espacements).
"""

# ---------------------------------------------------------------- données (graphiques, M4)
# Slots 1–3 de la palette de référence dataviz, validés (clair + sombre, toutes paires).
# L'aqua est < 3:1 sur fond clair : relief assuré par le tableau B0/B1 et par un tracé différent
# pour chaque prévision (tirets / pointillés).
OBSERVED = "#2a78d6"
B0 = "#eb6834"
B1 = "#1baf7a"
NEUTRAL = "#52514e"

FORECAST_STYLE = {"B0": {"color": B0, "dash": "dash"}, "B1": {"color": B1, "dash": "dot"}}

# ---------------------------------------------------------------- interface (M11)
COLORS = {
    "background": "#FFFFFF",
    "surface": "#F5F6F8",
    "border": "#D9DDE3",
    "text": "#1B1F24",
    "text_secondary": "#4B5563",
    "accent": "#1F4E8C",
    "success": "#1E7B4A",
    "warning": "#A15C00",
    "critical": "#B42318",
}
# Fonds teintés des alertes (texte d'état posé dessus : contraste >= 4,5:1 vérifié).
TINTS = {"success": "#E8F4EE", "warning": "#FDF3E4", "critical": "#FCEBEA", "accent": "#EAF0F8"}
TEXT_COLORS = ("text", "text_secondary", "accent", "success", "warning", "critical")

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
SPACING = (4, 8, 16, 24, 32, 48)
MAX_WIDTH_PX = 1200
# La barre de navigation Streamlit (≈ 60 px) recouvre le haut du contenu : le premier bloc (bandeau
# d'accueil encadré) doit commencer en dessous. 48 + 32 px, sur l'échelle d'espacement.
HEADER_CLEARANCE_PX = SPACING[5] + SPACING[4]
RADIUS_PX = 6


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(a: str, b: str) -> float:
    """Contraste WCAG 2.1 entre deux couleurs hexadécimales."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def css() -> str:
    """CSS injecté : complète le thème natif, sans dégradé, ombre ni animation."""
    return f"""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] {{ font-family: {FONT_STACK}; }}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: {MAX_WIDTH_PX}px; padding-top: {HEADER_CLEARANCE_PX}px;
}}
h2, [data-testid="stHeading"] h2 {{ margin-top: {SPACING[4]}px; }}
[data-testid="stDataFrame"], [data-testid="stMetricValue"], [data-testid="stTable"] {{
  font-variant-numeric: tabular-nums;
}}
[data-testid="stCaptionContainer"] {{ color: {COLORS["text_secondary"]}; font-size: 13px; }}
</style>
"""
