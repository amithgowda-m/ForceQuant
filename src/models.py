"""Equations 1-13 (see CLAUDE.md). Pure functions, no I/O.

Conventions for filings-based proxies:
  - missing input (None / NaN / "n/a")      -> returns N_A ("n/a")
  - not meaningful (e.g. OpInc <= 0 in DOL)  -> returns N_M ("n/m")
Theoretical equations raise ValueError on invalid inputs instead.
"""
from __future__ import annotations

import math
from typing import Mapping

N_A = "n/a"
N_M = "n/m"


def _missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return True  # "n/a", "n/m", "Not Disclosed", ...
    try:
        return math.isnan(x)
    except TypeError:
        return True


# 1. Fixed costs ---------------------------------------------------------------
def amortized_rd(i_rd: float, n_life: float) -> float:
    """A_RD = I_RD / N_life."""
    if n_life <= 0:
        raise ValueError("n_life must be > 0")
    return i_rd / n_life


def fixed_costs(fc_ops: float, i_rd: float, n_life: float, d_hw: float = 0.0) -> float:
    """FC = FC_ops + A_RD + D_hw."""
    return fc_ops + amortized_rd(i_rd, n_life) + d_hw


# 2. Total cost / revenue ------------------------------------------------------
def total_cost(fc: float, vc: float, q: float) -> float:
    """TC = FC + VC*Q."""
    return fc + vc * q


def total_revenue(p: float, q: float) -> float:
    """TR = P*Q."""
    return p * q


# 3. Break-even quantity -------------------------------------------------------
def breakeven_quantity(fc: float, p: float, vc: float) -> float:
    """Q_BE = FC / (P - VC). Requires P > VC."""
    if p <= vc:
        raise ValueError("price must exceed variable cost")
    return fc / (p - vc)


# 4. Degree of operating leverage ----------------------------------------------
def dol(q: float, p: float, vc: float, fc: float) -> float:
    """DOL = Q(P-VC) / (Q(P-VC) - FC)."""
    contribution = q * (p - vc)
    denom = contribution - fc
    if denom == 0:
        raise ValueError("DOL undefined at break-even (Q(P-VC) == FC)")
    return contribution / denom


def dol_empirical(revenue, cogs, op_income):
    """(Revenue - COGS) / OperatingIncome; n/m if OpInc <= 0, n/a if missing."""
    if _missing(revenue) or _missing(cogs) or _missing(op_income):
        return N_A
    if op_income <= 0:
        return N_M
    return (revenue - cogs) / op_income


# 5. Net revenue retention -----------------------------------------------------
def nrr(c: float, g: float) -> float:
    """NRR = 1 - c + g (c = churn rate, g = expansion rate)."""
    return 1 - c + g


# 6. Customer lifetime value ---------------------------------------------------
def ltv(p: float, vc: float, r: float, c: float, g: float) -> float:
    """LTV = (P-VC)/(r+c-g); valid only if r + c > g."""
    if r + c <= g:
        raise ValueError("LTV requires r + c > g")
    return (p - vc) / (r + c - g)


def ltv_proxy(arpu, gross_margin, churn):
    """Filings proxy LTV = ARPU*GM / c; n/a unless churn (and inputs) disclosed."""
    if _missing(arpu) or _missing(gross_margin) or _missing(churn):
        return N_A
    if churn <= 0:
        raise ValueError("churn must be > 0")
    return arpu * gross_margin / churn


# 7. LTV / CAC -----------------------------------------------------------------
def cac(sm_expense, new_customers):
    """CAC = S&M / new customers; n/a unless both disclosed."""
    if _missing(sm_expense) or _missing(new_customers) or new_customers <= 0:
        return N_A
    return sm_expense / new_customers


def ltv_cac(ltv_value, cac_value):
    """LTV/CAC; n/a if either input is missing."""
    if _missing(ltv_value) or _missing(cac_value) or cac_value == 0:
        return N_A
    return ltv_value / cac_value


# 8. Switching cost ------------------------------------------------------------
def switching_cost(c_mig: float, n_emp: float, h_train: float, w_emp: float,
                   c_int: float, rho: float, c_down: float) -> float:
    """SC = C_mig + N_emp*H_train*W_emp + C_int + rho*C_down."""
    return c_mig + n_emp * h_train * w_emp + c_int + rho * c_down


# 9. NPV of switching ----------------------------------------------------------
def pvaf(r: float, t: int) -> float:
    """Present value annuity factor: sum_{t=1..T} 1/(1+r)^t."""
    if t < 1:
        raise ValueError("T must be >= 1")
    return sum(1 / (1 + r) ** k for k in range(1, t + 1))


def annual_benefit(p_inc: float, p_new: float, d_e: float) -> float:
    """(P_inc - P_new) + dE."""
    return (p_inc - p_new) + d_e


def npv(p_inc: float, p_new: float, d_e: float, r: float, t: int, sc: float) -> float:
    """NPV = sum_{t=1..T} ((P_inc - P_new) + dE)/(1+r)^t - SC."""
    return annual_benefit(p_inc, p_new, d_e) * pvaf(r, t) - sc


def breakeven_price_cut(npv_value: float, r: float, t: int) -> float:
    """Extra annual price reduction by the new vendor that brings NPV to 0.

    = -NPV / PVAF when NPV < 0, else 0.
    """
    return max(0.0, -npv_value) / pvaf(r, t)


# 10. ROI ----------------------------------------------------------------------
def roi(npv_value: float, sc: float) -> float:
    """ROI = NPV / SC * 100."""
    if sc == 0:
        raise ValueError("SC must be non-zero")
    return npv_value / sc * 100


# 11. Price elasticity ---------------------------------------------------------
def price_elasticity(pct_dq: float, pct_dp: float) -> float:
    """E_d = |%dQ / %dP|."""
    if pct_dp == 0:
        raise ValueError("%dP must be non-zero")
    return abs(pct_dq / pct_dp)


# 12. Lerner index -------------------------------------------------------------
def lerner(p: float, mc: float) -> float:
    """L = (P - MC) / P."""
    if p == 0:
        raise ValueError("price must be non-zero")
    return (p - mc) / p


def lerner_from_elasticity(e_d: float) -> float:
    """L = 1 / E_d."""
    if e_d == 0:
        raise ValueError("elasticity must be non-zero")
    return 1 / e_d


def lerner_proxy(revenue, cogs):
    """Filings proxy (Revenue - COGS) / Revenue; n/a if missing or revenue <= 0."""
    if _missing(revenue) or _missing(cogs) or revenue <= 0:
        return N_A
    return (revenue - cogs) / revenue


# 13. Market Attractiveness Score ----------------------------------------------
def mas(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """MAS = 100 * (1 - sum_i w_i * (S_i - 1)/9); sum w_i = 1, S_i in [1, 10]."""
    if set(scores) != set(weights):
        raise ValueError("scores and weights must cover the same forces")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("weights must sum to 1")
    for k, s in scores.items():
        if not 1 <= s <= 10:
            raise ValueError(f"S_i for {k} must be in [1, 10], got {s}")
    return 100 * (1 - sum(weights[k] * (scores[k] - 1) / 9 for k in scores))
