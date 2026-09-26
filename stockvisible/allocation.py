"""F4 DECIDE — panier optimal sous budget, par énumération exhaustive bornée (≤ 3 produits).

Objectif : maximiser la demande couverte espérée, moyenne sur des scénarios équiprobables de
Σ_p min(stock_p + q_p, demande_p). Départage : à valeur égale, le panier le moins cher, puis le plus
petit en ordre lexicographique. Conséquence : si le stock couvre déjà tous les scénarios, acheter
n'apporte rien et le panier optimal est vide.

Les quantités s'achètent par lots entiers. Un produit sans coût connu n'est jamais acheté ; il est
signalé explicitement, jamais traité comme gratuit. Aucun solveur : on évalue toutes les
combinaisons utiles, ce qui se vérifie à la main.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAX_PRODUCTS = 3
MAX_COMBINATIONS = 2_000_000
VALUE_TOL = 1e-9


@dataclass(frozen=True)
class Product:
    name: str
    unit_cost: float | None  # None / NaN = coût inconnu → produit non achetable
    lot_size: float = 1.0
    stock: float = 0.0


@dataclass(frozen=True)
class Basket:
    names: tuple[str, ...]
    lots: tuple[int, ...]
    quantities: tuple[float, ...]
    cost: float
    expected_covered: float
    expected_covered_without_purchase: float
    expected_shortfall: float
    combinations_evaluated: int
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def gain(self) -> float:
        return self.expected_covered - self.expected_covered_without_purchase


def _cost_known(p: Product) -> bool:
    return p.unit_cost is not None and not math.isnan(p.unit_cost)


def _validate(products: list[Product], budget: float, demand: np.ndarray) -> None:
    if not 1 <= len(products) <= MAX_PRODUCTS:
        raise ValueError(f"entre 1 et {MAX_PRODUCTS} produits attendus, reçu {len(products)}")
    if len({p.name for p in products}) != len(products):
        raise ValueError("noms de produits dupliqués")
    if not (isinstance(budget, (int, float)) and math.isfinite(budget) and budget >= 0):
        raise ValueError(f"budget invalide : {budget!r} (fini et >= 0 attendu)")
    for p in products:
        if not (math.isfinite(p.lot_size) and p.lot_size > 0):
            raise ValueError(f"{p.name} : taille de lot > 0 attendue")
        if not (math.isfinite(p.stock) and p.stock >= 0):
            raise ValueError(f"{p.name} : stock fini et >= 0 attendu")
        if _cost_known(p) and not (math.isfinite(p.unit_cost) and p.unit_cost > 0):
            raise ValueError(f"{p.name} : coût unitaire > 0 attendu (ou None si inconnu)")
    if demand.ndim != 2 or demand.shape[1] != len(products) or demand.shape[0] == 0:
        raise ValueError(f"scénarios de forme (n_scénarios, {len(products)}) attendus, reçu {demand.shape}")
    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("demandes des scénarios finies et >= 0 attendues")


def _useful_lots(p: Product, budget: float, max_demand: float) -> int:
    """Au-delà du pic de demande des scénarios, un lot de plus ne couvre rien : borne naturelle."""
    if not _cost_known(p):
        return 0
    by_budget = math.floor(budget / (p.unit_cost * p.lot_size) + 1e-12)
    by_demand = math.ceil(max(max_demand - p.stock, 0.0) / p.lot_size - 1e-12)
    return max(0, min(by_budget, by_demand))


def optimal_basket(
    products: list[Product], budget: float, scenarios: np.ndarray | list[list[float]]
) -> Basket:
    """scenarios[s, p] = demande du produit p dans le scénario s (scénarios équiprobables)."""
    demand = np.asarray(scenarios, dtype=float)
    _validate(products, budget, demand)

    axes_value, axes_cost, axes_lots = [], [], []
    for j, p in enumerate(products):
        lots = np.arange(_useful_lots(p, budget, float(demand[:, j].max())) + 1)
        supply = p.stock + lots * p.lot_size
        axes_value.append(np.minimum(supply[:, None], demand[None, :, j]).mean(axis=1))
        axes_cost.append(lots * (p.unit_cost * p.lot_size if _cost_known(p) else 0.0))
        axes_lots.append(lots)

    shape = tuple(len(a) for a in axes_lots)
    n_comb = int(np.prod(shape))
    if n_comb > MAX_COMBINATIONS:
        raise ValueError(f"énumération trop grande ({n_comb} > {MAX_COMBINATIONS}) : augmenter les lots")

    value = np.zeros(shape)
    cost = np.zeros(shape)
    for j in range(len(products)):
        view = [1] * len(products)
        view[j] = shape[j]
        value = value + axes_value[j].reshape(view)
        cost = cost + axes_cost[j].reshape(view)

    feasible = cost <= budget + 1e-9
    best_value = value[feasible].max()
    candidates = feasible & (value >= best_value - VALUE_TOL)
    best_cost = cost[candidates].min()
    winners = np.argwhere(candidates & (cost <= best_cost + 1e-9))
    chosen = tuple(int(i) for i in winners[0])  # argwhere : ordre lexicographique

    quantities = tuple(float(chosen[j] * p.lot_size) for j, p in enumerate(products))
    stock = np.array([p.stock for p in products])
    covered_now = np.minimum(stock[None, :], demand).sum(axis=1).mean()
    final = stock + np.array(quantities)
    return Basket(
        names=tuple(p.name for p in products),
        lots=chosen,
        quantities=quantities,
        cost=float(cost[chosen]),
        expected_covered=float(value[chosen]),
        expected_covered_without_purchase=float(covered_now),
        expected_shortfall=float(np.maximum(demand - final[None, :], 0).sum(axis=1).mean()),
        combinations_evaluated=n_comb,
        excluded={p.name: "coût unitaire inconnu : non achetable" for p in products if not _cost_known(p)},
    )
