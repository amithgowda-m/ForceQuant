"""Tests for Stage 7 scoring and Stage 7b sensitivity."""
import numpy as np
import pandas as pd
import pytest

from src import scoring as sc
from src import models


def test_linear_scale_boundaries():
    # Value at anchor 1 -> 1.0
    assert sc.linear_scale(0.0, 0.0, 0.33) == pytest.approx(1.0)
    # Value at anchor 10 -> 10.0
    assert sc.linear_scale(0.33, 0.0, 0.33) == pytest.approx(10.0)
    # Clamping below anchor 1
    assert sc.linear_scale(-0.1, 0.0, 0.33) == 1.0
    # Clamping above anchor 10
    assert sc.linear_scale(0.50, 0.0, 0.33) == 10.0
    # Inverted scale (e.g. gross margin where high margin = low threat)
    assert sc.linear_scale(0.72, 0.72, 0.38) == pytest.approx(1.0)
    assert sc.linear_scale(0.38, 0.72, 0.38) == pytest.approx(10.0)


def test_gross_margin_score():
    anchors = {"value_at_1": 0.72, "value_at_10": 0.38}
    assert sc.gross_margin_score(0.72, anchors) == pytest.approx(1.0)
    assert sc.gross_margin_score(0.38, anchors) == pytest.approx(10.0)
    assert sc.gross_margin_score("n/a", anchors) is None
    assert sc.gross_margin_score(None, anchors) is None


def test_dol_score():
    anchors = {"value_at_1": 5.0, "value_at_10": 1.0}
    assert sc.dol_score(5.0, anchors) == pytest.approx(1.0)
    assert sc.dol_score(1.0, anchors) == pytest.approx(10.0)
    assert sc.dol_score("n/m", anchors) is None
    assert sc.dol_score("n/a", anchors) is None


def test_lerner_trend_score():
    anchors = {"value_at_1": 0.01, "value_at_10": -0.01, "n_years": 3}
    # Rising margin (+0.01/yr) -> low threat (score 1)
    fm = pd.DataFrame({
        "Ticker": ["ZM", "ZM", "ZM"],
        "FiscalYear": ["FY2024", "FY2025", "FY2026"],
        "Lerner_proxy": [0.70, 0.71, 0.72],
    })
    s = sc.lerner_trend_score("ZM", fm, anchors)
    assert s == pytest.approx(1.0, abs=1e-2)

    # Missing ticker
    assert sc.lerner_trend_score("UNKNOWN", fm, anchors) is None


def test_compute_si_nlp_only():
    cfg = {
        "forces": ["Supplier_Power"],
        "scoring": {
            "nlp_only": ["Supplier_Power"],
            "anchors": {
                "mention_share": {"value_at_1": 0.0, "value_at_10": 0.33},
            },
        },
    }
    shares = pd.DataFrame([{"Ticker": "ZM", "Force": "Supplier_Power", "MentionShare": 0.033}])
    fm = pd.DataFrame()
    si, parts = sc.compute_si("ZM", "Supplier_Power", fm, shares, cfg)
    assert 1.0 <= si <= 10.0
    assert len(parts) == 1
    assert "NLP_share=" in parts[0]
