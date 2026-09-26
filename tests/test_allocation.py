"""optimal_basket — cas limites SYNTHÉTIQUES + recoupement avec une énumération naïve."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from stockvisible.allocation import MAX_PRODUCTS, Basket, Product, optimal_basket


def _p(name, cost=1.0, lot=1.0, stock=0.0):
    return Product(name=name, unit_cost=cost, lot_size=lot, stock=stock)


def _naive(products, budget, demand):
    """Référence indépendante : itertools sur tous les lots, mêmes règles de départage."""
    demand = np.asarray(demand, float)
    ranges = []
    for j, p in enumerate(products):
        if p.unit_cost is None or math.isnan(p.unit_cost):
            ranges.append([0])
        else:
            top = int(budget // (p.unit_cost * p.lot_size)) if budget > 0 else 0
            ranges.append(range(min(top, 50) + 1))
    best = None
    for lots in itertools.product(*ranges):
        cost = sum(k * p.unit_cost * p.lot_size for k, p in zip(lots, products) if k)
        if cost > budget + 1e-9:
            continue
        value = sum(
            np.minimum(p.stock + k * p.lot_size, demand[:, j]).mean()
            for j, (k, p) in enumerate(zip(lots, products))
        )
        key = (-round(value, 9), round(cost, 9), lots)
        if best is None or key < best[0]:
            best = (key, lots, value, cost)
    return best


# ---------------------------------------------------------------- cas limites


def test_zero_budget_buys_nothing():
    b = optimal_basket([_p("A"), _p("B"), _p("C")], 0.0, [[5, 5, 5]])
    assert b.lots == (0, 0, 0) and b.cost == 0.0
    assert b.expected_covered == b.expected_covered_without_purchase == 0.0
    assert b.expected_shortfall == 15.0


def test_sufficient_stock_means_zero_purchase_even_with_budget():
    products = [_p("A", stock=10), _p("B", stock=4), _p("C", stock=7)]
    scenarios = [[3, 4, 2], [10, 1, 7], [0, 0, 0]]
    b = optimal_basket(products, 1_000.0, scenarios)
    assert b.lots == (0, 0, 0) and b.cost == 0.0 and b.gain == 0.0
    assert b.expected_shortfall == 0.0


def test_partially_sufficient_stock_only_buys_where_useful():
    products = [_p("A", stock=10), _p("B", stock=0)]
    b = optimal_basket(products, 100.0, [[5, 3], [8, 3]])
    assert b.quantities == (0.0, 3.0)


def test_missing_cost_is_excluded_not_free():
    for missing in (None, float("nan")):
        products = [_p("A", cost=missing), _p("B", cost=2.0)]
        b = optimal_basket(products, 10.0, [[4, 4]])
        assert b.lots[0] == 0 and b.excluded == {"A": "coût unitaire inconnu : non achetable"}
        assert b.quantities[1] == 4.0 and b.cost == 8.0


def test_all_costs_missing_gives_empty_basket():
    b = optimal_basket([_p("A", cost=None), _p("B", cost=None)], 50.0, [[1, 1]])
    assert b.lots == (0, 0) and set(b.excluded) == {"A", "B"}


def test_integer_lots_and_budget_not_divisible():
    b = optimal_basket([_p("A", cost=1.5, lot=4)], 10.0, [[20]])  # lot = 6 ; 10 // 6 = 1
    assert b.lots == (1,) and b.quantities == (4.0,) and b.cost == 6.0
    b2 = optimal_basket([_p("A", cost=1.5, lot=4)], 11.99, [[20]])
    assert b2.lots == (1,)
    b3 = optimal_basket([_p("A", cost=1.5, lot=4)], 12.0, [[20]])
    assert b3.lots == (2,) and b3.cost == 12.0


def test_fractional_lot_sizes_stay_whole_lots():
    b = optimal_basket([_p("A", cost=2.0, lot=0.5)], 3.0, [[1.2]])
    assert b.lots == (3,) and b.quantities == (1.5,) and b.cost == 3.0


def test_no_useless_overbuy_beyond_peak_demand():
    b = optimal_basket([_p("A", cost=1.0)], 1_000.0, [[2], [3], [1]])
    assert b.quantities == (3.0,)


def test_budget_goes_to_best_marginal_coverage():
    # A : demande certaine 2 ; B : demande 2 une fois sur deux. Budget pour 2 unités → A.
    b = optimal_basket([_p("A"), _p("B")], 2.0, [[2, 2], [2, 0]])
    assert b.quantities == (2.0, 0.0) and b.expected_covered == 2.0


def test_tie_prefers_cheapest_then_lexicographic():
    b = optimal_basket([_p("A", cost=3.0), _p("B", cost=1.0)], 3.0, [[1, 1], [0, 0]])
    # A seul ou B seul couvrent 0,5 en moyenne ; A+B (4) dépasse le budget → le moins cher, B.
    assert b.quantities == (0.0, 1.0) and b.cost == 1.0
    tie = optimal_basket([_p("A"), _p("B")], 1.0, [[1, 1]])
    assert tie.lots == (0, 1)  # même valeur, même coût : départage lexicographique déterministe


def test_value_is_monotone_in_budget():
    rng = np.random.default_rng(0)
    products = [_p("A", 1.3, 1), _p("B", 0.7, 2), _p("C", 2.1, 0.5, stock=1)]
    scenarios = rng.gamma(2.0, 1.5, size=(15, 3)).round(1)
    values = [optimal_basket(products, b, scenarios).expected_covered for b in np.linspace(0, 20, 21)]
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(values))


@pytest.mark.parametrize("seed", range(200))
def test_matches_naive_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, MAX_PRODUCTS + 1))
    products = [
        _p(f"P{j}",
           cost=None if rng.random() < 0.15 else float(rng.choice([0.5, 1.0, 1.5, 2.0])),
           lot=float(rng.choice([0.5, 1.0, 2.0])),
           stock=float(rng.choice([0.0, 0.5, 2.0])))
        for j in range(n)
    ]
    scenarios = rng.integers(0, 6, size=(int(rng.integers(1, 6)), n)).astype(float)
    budget = float(rng.choice([0.0, 1.0, 2.5, 4.0, 7.0]))
    b = optimal_basket(products, budget, scenarios)
    _, lots, value, cost = _naive(products, budget, scenarios)
    assert b.expected_covered == pytest.approx(value, abs=1e-9)
    assert b.lots == lots and b.cost == pytest.approx(cost)
    assert b.cost <= budget + 1e-9


def test_result_invariants():
    products = [_p("A", 1.0, 2.0, 1.0), _p("B", 3.0, 1.0)]
    b = optimal_basket(products, 7.0, [[5, 2], [1, 3]])
    assert isinstance(b, Basket) and b.names == ("A", "B")
    assert all(q == k * p.lot_size for q, k, p in zip(b.quantities, b.lots, products))
    assert b.expected_covered >= b.expected_covered_without_purchase
    assert b.combinations_evaluated >= 1


# ---------------------------------------------------------------- entrées invalides


@pytest.mark.parametrize(
    "products, budget, scenarios, match",
    [
        ([], 1.0, [[1]], "entre 1 et 3"),
        ([_p(c) for c in "ABCD"], 1.0, [[1, 1, 1, 1]], "entre 1 et 3"),
        ([_p("A"), _p("A")], 1.0, [[1, 1]], "dupliqués"),
        ([_p("A")], -1.0, [[1]], "budget"),
        ([_p("A")], float("nan"), [[1]], "budget"),
        ([_p("A")], float("inf"), [[1]], "budget"),
        ([_p("A", cost=0.0)], 1.0, [[1]], "coût"),
        ([_p("A", cost=-2.0)], 1.0, [[1]], "coût"),
        ([_p("A", lot=0)], 1.0, [[1]], "lot"),
        ([_p("A", stock=-1)], 1.0, [[1]], "stock"),
        ([_p("A")], 1.0, [[1, 2]], "forme"),
        ([_p("A")], 1.0, np.empty((0, 1)), "forme"),
        ([_p("A")], 1.0, [[-1]], "finies"),
        ([_p("A")], 1.0, [[float("nan")]], "finies"),
    ],
)
def test_invalid_inputs_are_refused(products, budget, scenarios, match):
    with pytest.raises(ValueError, match=match):
        optimal_basket(products, budget, scenarios)


def test_enumeration_bound_is_explicit():
    products = [_p(c, cost=0.01, lot=0.01) for c in "ABC"]
    with pytest.raises(ValueError, match="énumération trop grande"):
        optimal_basket(products, 1e6, [[100, 100, 100]])
