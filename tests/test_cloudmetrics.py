"""Case 4 (CloudMetrics, hypothetical): exact targets from CLAUDE.md."""
import pytest

from src import models as m

P, VC = 40_000, 8_000
R, T = 0.10, 4


@pytest.fixture
def cm():
    fc = m.fixed_costs(fc_ops=1_200_000, i_rd=6_000_000, n_life=5)
    sc = m.switching_cost(c_mig=25_000, n_emp=50, h_train=20, w_emp=35,
                          c_int=15_000, rho=0.05, c_down=300_000)
    npv = m.npv(p_inc=55_000, p_new=40_000, d_e=10_000, r=R, t=T, sc=sc)
    return {"fc": fc, "sc": sc, "npv": npv}


def test_fixed_costs(cm):
    assert cm["fc"] == 2_400_000


def test_breakeven_quantity(cm):
    assert m.breakeven_quantity(cm["fc"], P, VC) == 75


def test_switching_cost(cm):
    assert cm["sc"] == pytest.approx(90_000)


def test_annual_benefit():
    assert m.annual_benefit(55_000, 40_000, 10_000) == 25_000


def test_pvaf():
    assert round(m.pvaf(R, T), 6) == 3.169865


def test_npv(cm):
    assert round(cm["npv"], 1) == -10_753.4
    assert round(cm["npv"]) == -10_753


def test_roi(cm):
    assert round(m.roi(cm["npv"], cm["sc"]), 2) == -11.95


def test_breakeven_price_cut(cm):
    # CLAUDE.md target 3,392.3 is reproduced from the rounded NPV (-10,753) / PVAF.
    assert round(m.breakeven_price_cut(round(cm["npv"]), R, T), 1) == 3_392.3
    # Unrounded NPV gives 3,392.37 -> 3,392.4 (logged in outputs/issues.log).
    assert round(m.breakeven_price_cut(cm["npv"], R, T), 1) == 3_392.4
