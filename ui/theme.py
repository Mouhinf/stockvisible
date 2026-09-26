"""Jetons de couleur. Slots 1–3 de la palette de référence dataviz, validés (clair + sombre,
toutes paires). L'aqua est < 3:1 sur fond clair : relief assuré par le tableau B0/B1 et par un
tracé différent pour chaque prévision (tirets / pointillés)."""

OBSERVED = "#2a78d6"
B0 = "#eb6834"
B1 = "#1baf7a"
NEUTRAL = "#52514e"

FORECAST_STYLE = {"B0": {"color": B0, "dash": "dash"}, "B1": {"color": B1, "dash": "dot"}}
