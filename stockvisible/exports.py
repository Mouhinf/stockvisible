"""F5 ACT — valider puis exporter le panier (M12, M14).

Une seule source de vérité : `basket_view()` produit les valeurs du panier, arrondies comme à
l'écran. L'écran Acheter les affiche telles quelles, et les exports CSV/JSON les sérialisent sans
recalcul : ce qui est exporté est strictement ce qui est affiché.

Le moteur gelé du projet (ML) n'est pas celui qui estime la demande du panier (B1) : l'export
porte les deux, explicitement. Toute cellule texte commençant (après espaces) par =, +, -, @,
tabulation ou retour chariot est préfixée d'une apostrophe, pour empêcher l'injection de formules à
l'ouverture dans un tableur.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from stockvisible.allocation import Basket, Product

DECIMALS = 2  # arrondi des indicateurs, identique à l'écran
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
DEMAND_SOURCE = "B1 estimé (ventes disponibles + B1 aux heures en rupture 6-22 h)"
EXPORT_VERSION = 2

CSV_COLUMNS = [
    "serie",
    "lots",
    "quantite_unites_normalisees",
    "cout_unitaire_hypothese",
    "taille_lot",
    "stock_actuel_hypothese",
    "budget",
    "cout_panier",
    "demande_couverte_esperee_jour",
    "gain_vs_sans_achat",
    "manque_espere_jour",
    "moteur_demande_panier",
    "moteur_gele_projet",
    "scenarios",
    "donnees",
    "horodatage_utc",
    "empreinte_panier",
]


@dataclass(frozen=True)
class ValidatedBasket:
    basket: Basket
    products: tuple[Product, ...]
    budget: float
    validated_at: str
    fingerprint: str
    context: dict = field(default_factory=dict)


def basket_fingerprint(basket: Basket, products: list[Product] | tuple[Product, ...], budget: float) -> str:
    """Identifie exactement ce qui a été validé : tout changement l'invalide."""
    payload = {
        "names": basket.names,
        "lots": basket.lots,
        "budget": round(float(budget), 6),
        "products": [(p.name, p.unit_cost, p.lot_size, p.stock) for p in products],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def validate_basket(
    basket: Basket,
    products: list[Product],
    budget: float,
    now: datetime | None = None,
    context: dict | None = None,
) -> ValidatedBasket:
    if basket.cost > budget + 1e-9:
        raise ValueError("panier au-dessus du budget : validation refusée")
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    return ValidatedBasket(
        basket, tuple(products), float(budget), stamp, basket_fingerprint(basket, products, budget),
        dict(context or {}),
    )


def basket_view(
    basket: Basket,
    products: list[Product] | tuple[Product, ...],
    budget: float,
    context: dict | None = None,
    validated: ValidatedBasket | None = None,
) -> dict:
    """Tout ce que l'écran affiche du panier, et rien d'autre, dans les valeurs affichées."""
    ctx = context or {}
    return {
        "version_export": EXPORT_VERSION,
        "horodatage_utc": validated.validated_at if validated else None,
        "empreinte_panier": validated.fingerprint if validated else None,
        "budget": round(float(budget), DECIMALS),
        "indicateurs": {
            "cout_panier": round(basket.cost, DECIMALS),
            "demande_couverte_esperee_jour": round(basket.expected_covered, DECIMALS),
            "gain_vs_sans_achat": round(basket.gain, DECIMALS),
            "manque_espere_jour": round(basket.expected_shortfall, DECIMALS),
        },
        "panier": [
            {"serie": n, "lots": int(k), "quantite_unites_normalisees": float(q)}
            for n, k, q in zip(basket.names, basket.lots, basket.quantities, strict=True)
        ],
        "hypotheses": [
            {
                "serie": p.name,
                "cout_unitaire_hypothese": None if p.unit_cost is None else float(p.unit_cost),
                "taille_lot": float(p.lot_size),
                "stock_actuel_hypothese": float(p.stock),
            }
            for p in products
        ],
        "exclus": dict(basket.excluded),
        "moteur": {
            "demande_panier": DEMAND_SOURCE,
            "moteur_gele_projet": ctx.get("moteur_gele_projet", "inconnu"),
        },
        "provenance": {
            "donnees": ctx.get("donnees", "inconnu"),
            "scenarios": ctx.get("scenarios", "inconnu"),
            "combinaisons_evaluees": int(basket.combinations_evaluated),
            "hypotheses_saisies": "coûts et stocks = hypothèses de l'utilisateur (absents du jeu de données)",
        },
    }


def validated_view(validated: ValidatedBasket) -> dict:
    return basket_view(
        validated.basket, validated.products, validated.budget, validated.context, validated
    )


def _safe(value: object) -> object:
    if isinstance(value, str) and value.lstrip(" ").startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def export_frame(validated: ValidatedBasket) -> pd.DataFrame:
    view = validated_view(validated)
    hyp = {h["serie"]: h for h in view["hypotheses"]}
    ind = view["indicateurs"]
    rows = []
    for line in view["panier"]:
        h = hyp[line["serie"]]
        rows.append(
            {
                **line,
                "cout_unitaire_hypothese": "" if h["cout_unitaire_hypothese"] is None else h["cout_unitaire_hypothese"],
                "taille_lot": h["taille_lot"],
                "stock_actuel_hypothese": h["stock_actuel_hypothese"],
                "budget": view["budget"],
                **ind,
                "moteur_demande_panier": view["moteur"]["demande_panier"],
                "moteur_gele_projet": view["moteur"]["moteur_gele_projet"],
                "scenarios": view["provenance"]["scenarios"],
                "donnees": view["provenance"]["donnees"],
                "horodatage_utc": view["horodatage_utc"],
                "empreinte_panier": view["empreinte_panier"],
            }
        )
    return pd.DataFrame(rows, columns=CSV_COLUMNS).map(_safe)


def export_csv(validated: ValidatedBasket) -> bytes:
    buffer = io.StringIO()
    export_frame(validated).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def export_json(validated: ValidatedBasket) -> bytes:
    return (json.dumps(validated_view(validated), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
