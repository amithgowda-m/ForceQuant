"""Offline tests for src/labelling.py."""
import pandas as pd

from src import labelling as lb

FORCES = ["Competitive_Rivalry", "Threat_of_New_Entry", "Threat_of_Substitutes",
          "Buyer_Power", "Supplier_Power", "Complementors"]


def test_unique_keeps_latest_year_one_per_hash():
    p = pd.DataFrame({"para_hash": ["h", "h", "g"], "fiscal_year": ["FY2024", "FY2026", "FY2025"],
                      "item": ["1A", "1A", "7"], "para_idx": [1, 1, 2], "para_id": ["a", "b", "c"]})
    u = lb.unique_paragraphs(p)
    assert sorted(u.para_id) == ["b", "c"]


def test_stratified_sample_is_reproducible_and_capped():
    p = pd.DataFrame({"para_hash": [f"h{i}" for i in range(60)], "ticker": ["A"] * 30 + ["B"] * 30,
                      "item": (["1A", "7"] * 30), "fiscal_year": "FY2026", "para_idx": range(60),
                      "para_id": [f"p{i}" for i in range(60)]})
    s1, s2 = lb.stratified_sample(p, 5, 42), lb.stratified_sample(p, 5, 42)
    assert s1.para_id.tolist() == s2.para_id.tolist()
    assert s1.groupby(["ticker", "item"]).size().tolist() == [5, 5, 5, 5]


def test_check_sheet_columns_and_force_coverage():
    labels = pd.DataFrame({"para_id": [f"p{i}" for i in range(80)], "ticker": "A", "text": "t",
                           "llm_label": ["Buyer_Power"] * 10 + ["Complementors;Buyer_Power"] * 3 + ["None"] * 67})
    sheet = lb.check_sheet(labels, FORCES, 40, 6, 42)
    assert list(sheet.columns) == ["para_id", "text", "human_label"]
    assert len(sheet) == 40 and sheet.para_id.is_unique and (sheet.human_label == "").all()
    lab = labels.set_index("para_id").loc[sheet.para_id, "llm_label"]
    assert lab.str.contains("Complementors").sum() == 3          # all available (< 6)
    assert lab.str.contains("Buyer_Power").sum() >= 6
