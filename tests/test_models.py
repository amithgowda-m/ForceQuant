"""Edge-case rules for src/models.py (n/m, n/a, ValueError)."""
import math

import pytest

from src import models as m


def test_dol_theoretical():
    # Q=100, P=40k, VC=8k, FC=2.4M -> 3.2M / 0.8M = 4
    assert m.dol(100, 40_000, 8_000, 2_400_000) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        m.dol(75, 40_000, 8_000, 2_400_000)


def test_dol_empirical_rules():
    assert m.dol_empirical(100, 30, 35) == pytest.approx(2.0)
    assert m.dol_empirical(100, 30, 0) == m.N_M
    assert m.dol_empirical(100, 30, -5) == m.N_M
    assert m.dol_empirical(100, None, 10) == m.N_A
    assert m.dol_empirical(float("nan"), 30, 10) == m.N_A


def test_nrr():
    assert m.nrr(0.05, 0.10) == pytest.approx(1.05)


def test_ltv_requires_r_plus_c_gt_g():
    assert m.ltv(100, 20, 0.10, 0.05, 0.05) == pytest.approx(800)
    with pytest.raises(ValueError):
        m.ltv(100, 20, 0.05, 0.05, 0.10)
    with pytest.raises(ValueError):
        m.ltv(100, 20, 0.05, 0.05, 0.10)  # r + c == g


def test_ltv_proxy_needs_churn():
    assert m.ltv_proxy(1000, 0.8, 0.1) == pytest.approx(8000)
    assert m.ltv_proxy(1000, 0.8, None) == m.N_A
    assert m.ltv_proxy(1000, 0.8, "Not Disclosed") == m.N_A


def test_cac_and_ltv_cac():
    assert m.cac(1_000_000, 100) == 10_000
    assert m.cac(1_000_000, None) == m.N_A
    assert m.cac(m.N_A, 100) == m.N_A
    assert m.ltv_cac(30_000, 10_000) == 3
    assert m.ltv_cac(m.N_A, 10_000) == m.N_A
    assert m.ltv_cac(30_000, m.N_A) == m.N_A


def test_elasticity_and_lerner():
    assert m.price_elasticity(-20, 10) == 2
    assert m.lerner_from_elasticity(2) == 0.5
    assert m.lerner(40_000, 8_000) == pytest.approx(0.8)
    assert m.lerner_proxy(100, 25) == pytest.approx(0.75)
    assert m.lerner_proxy(0, 25) == m.N_A
    assert m.lerner_proxy(None, 25) == m.N_A
    with pytest.raises(ValueError):
        m.price_elasticity(5, 0)


def test_mas():
    forces = ["a", "b", "c", "d", "e", "f"]
    w = {f: 1 / 6 for f in forces}
    assert m.mas({f: 1 for f in forces}, w) == pytest.approx(100)
    assert m.mas({f: 10 for f in forces}, w) == pytest.approx(0)
    assert m.mas({f: 5.5 for f in forces}, w) == pytest.approx(50)
    with pytest.raises(ValueError):
        m.mas({f: 11 for f in forces}, w)
    with pytest.raises(ValueError):
        m.mas({f: 5 for f in forces}, {f: 0.2 for f in forces})
