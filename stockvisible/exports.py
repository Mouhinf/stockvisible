"""F5 ACT — valider puis exporter le panier (M12).

Le CSV exporté porte ses hypothèses et sa provenance : chaque ligne rappelle le budget, la date de
validation, l'origine de la demande (B1 estimé) et le fait que coûts et stocks sont des hypothèses
saisies. Toute cellule texte commençant par =, +, -, @, tabulation ou retour chariot est préfixée
d'une apostrophe, pour empêcher l'injection de formules à l'ouverture dans un tableur.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from stockvisible.allocation import Basket, Product

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
EXPORT_COLUMNS = [
    "serie",
    "lots",
    "quantite_unites_normalisees",
    "cout_unitaire_hypothese",
    "taille_lot",
    "stock_actuel_hypothese",
    "cout_ligne",
    "budget",
    "cout_total_panier",
    "demande_couverte_esperee_jour",
    "source_demande",
    "valide_le_utc",
    "empreinte_panier",
]


@dataclass(frozen=True)
class ValidatedBasket:
    basket: Basket
    products: tuple[Product, ...]
    budget: float
    validated_at: str
    fingerprint: str


def basket_fingerprint(basket: Basket, products: list[Product] | tuple[Product, ...], budget: float) -> str:
    """Identifie exactement ce qui a été validé : tout changement l'invalide."""
    payload = {
        "names": basket.names,
        "lots": basket.lots,
        "budget": round(float(budget), 6),
        "products": [(p.name, p.unit_cost, p.lot_size, p.stock) for p in products],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def validate_basket(basket: Basket, products: list[Product], budget: float, now: datetime | None = None) -> ValidatedBasket:
    if basket.cost > budget + 1e-9:
        raise ValueError("panier au-dessus du budget : validation refusée")
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    return ValidatedBasket(basket, tuple(products), float(budget), stamp, basket_fingerprint(basket, products, budget))


def _safe(value: object) -> object:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def export_frame(validated: ValidatedBasket) -> pd.DataFrame:
    b = validated.basket
    by_name = {p.name: p for p in validated.products}
    rows = []
    for name, lots, qty in zip(b.names, b.lots, b.quantities, strict=True):
        p = by_name[name]
        unit = p.unit_cost
        rows.append(
            {
                "serie": name,
                "lots": int(lots),
                "quantite_unites_normalisees": float(qty),
                "cout_unitaire_hypothese": "" if unit is None else float(unit),
                "taille_lot": float(p.lot_size),
                "stock_actuel_hypothese": float(p.stock),
                "cout_ligne": 0.0 if unit is None else round(float(qty) * float(unit), 6),
                "budget": validated.budget,
                "cout_total_panier": round(b.cost, 6),
                "demande_couverte_esperee_jour": round(b.expected_covered, 6),
                "source_demande": "B1 estimé (ventes disponibles + B1 aux heures en rupture 6-22 h)",
                "valide_le_utc": validated.validated_at,
                "empreinte_panier": validated.fingerprint,
            }
        )
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS).map(_safe)


def export_csv(validated: ValidatedBasket) -> bytes:
    buffer = io.StringIO()
    export_frame(validated).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
