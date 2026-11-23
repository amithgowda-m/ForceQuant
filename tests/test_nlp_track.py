"""Tests for Stage 6: nlp_track (offline parts only — no network, no model download)."""
import numpy as np
import pandas as pd
import pytest

from src import nlp_track as nt

FORCES = [
    "Competitive_Rivalry", "Threat_of_New_Entry", "Threat_of_Substitutes",
    "Buyer_Power", "Supplier_Power", "Complementors",
]


def _make_labels(llm, human):
    return pd.DataFrame({
        "para_id": [f"p{i}" for i in range(len(llm))],
        "llm_label": llm,
        "human_label": human,
        "sample_type": ["random"] * len(llm),
        "ticker": ["ZM"] * len(llm),
    })


def test_labels_to_binary_roundtrip():
    labels = ["Competitive_Rivalry", "Buyer_Power;Supplier_Power", "None"]
    mat = nt.labels_to_binary(pd.Series(labels), FORCES)
    assert mat.shape == (3, 6)
    assert mat[0, FORCES.index("Competitive_Rivalry")] == 1
    assert mat[0, FORCES.index("Buyer_Power")] == 0
    assert mat[1, FORCES.index("Buyer_Power")] == 1
    assert mat[1, FORCES.index("Supplier_Power")] == 1
    assert mat[2].sum() == 0


def test_agreement_perfect():
    llm = ["Competitive_Rivalry", "None", "Buyer_Power"]
    human = ["Competitive_Rivalry", "None", "Buyer_Power"]
    df = _make_labels(llm, human)
    result = nt.compute_agreement(df, ["Competitive_Rivalry", "Buyer_Power"])
    for _, row in result.iterrows():
        assert row["PercentAgreement"] == pytest.approx(1.0)
        assert row["Kappa"] == pytest.approx(1.0)


def test_agreement_zero():
    llm = ["Competitive_Rivalry", "Competitive_Rivalry"]
    human = ["None", "None"]
    df = _make_labels(llm, human)
    result = nt.compute_agreement(df, ["Competitive_Rivalry"])
    row = result.iloc[0]
    assert row["PercentAgreement"] == pytest.approx(0.0)


def test_baseline_predict_above_threshold():
    # Perfect match: prototype vec IS the query
    proto = {"Competitive_Rivalry": np.array([[1.0, 0.0, 0.0]])}
    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    preds = nt.baseline_predict(emb, proto, threshold=0.9, forces=["Competitive_Rivalry"])
    assert preds[0] == ["Competitive_Rivalry"]
    assert preds[1] == []


def test_llm_label_shares_sums():
    labels = pd.DataFrame({
        "para_id": [f"p{i}" for i in range(10)],
        "ticker": ["ZM"] * 10,
        "llm_label": ["Competitive_Rivalry"] * 5 + ["None"] * 5,
        "sample_type": ["random"] * 10,
    })
    result = nt.llm_label_shares(labels, FORCES)
    zm_rivalry = result[(result["Ticker"] == "ZM") & (result["Force"] == "Competitive_Rivalry")]
    assert zm_rivalry["LLM_MentionShare_random"].iloc[0] == pytest.approx(0.5)
