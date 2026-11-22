"""Rules for src/kpi_extract.py (offline)."""
import pandas as pd

from src import kpi_extract as k


def test_customers_word_alone_is_not_a_match():
    assert k.parse_sentence("We serve many customers as of January 31, 2026.", "Customers") is None
    assert k.parse_sentence("The USG allowed us to ship 200 H200 products to China-based customers as of "
                            "January 31, 2026.", "Customers") is None
    # count without an "as of" date (a flow, not a stock) is not a customer KPI
    assert k.parse_sentence("We transitioned approximately 26,800 Enterprise customers away from direct sales.",
                            "Customers") is None


def test_value_must_be_in_same_sentence():
    text = "We track our Net Expansion Rate closely. It was approximately 112% as of April 30, 2026."
    hits = [k.parse_sentence(s, "NDR") for s in k.sentences(text)]
    assert hits == [None, None]


def test_illustrative_sentence_skipped():
    s = "For instance, if each customer renewed for the same amount, the Net Expansion Rate would be 100%."
    assert k.parse_sentence(s, "NDR") is None


def test_respectively_maps_values_to_years():
    s = ("Our trailing 12 month net dollar expansion rate for Enterprise customers was 98%, 98%, and 101% "
         "as of January 31, 2026, 2025, and 2024, respectively.")
    h = k.parse_sentence(s, "NDR")
    assert h["by_year"] == {2026: "98%", 2025: "98%", 2024: "101%"}
    assert "trailing 12 month" in h["period_as_written"]


def test_unmappable_values_not_assigned_a_period():
    h = k.parse_sentence("Our net ARR expansion rate has consistently been over 120%.", "NDR")
    assert h["values"] == ["over 120%"] and h["by_year"] == {}


def test_arr_definition_with_number_is_not_an_arr_value():
    assert k.parse_sentence("We calculate ARR by taking the MRR and multiplying it by 12.", "ARR") is None
    assert k.parse_sentence("The number of customers with $100,000 or greater in ARR was 2,799 as of "
                            "January 31, 2026.", "ARR") is None


def test_customer_threshold_and_segment():
    h = k.parse_sentence("The number of customers with $100,000 or greater in ARR was 2,799, 2,396 and 2,052 "
                         "as of January 31, 2026, 2025 and 2024, respectively.", "Customers")
    assert k.kpi_label(h) == "Customers_over_threshold"
    assert h["threshold"] == "$100,000 or greater in ARR"
    assert h["by_year"][2026] == "2,799"
    h = k.parse_sentence("As of January 31, 2025, we had over 7,500 customers that were sold through our direct "
                         "sales force, as compared to over 7,000 and over 6,400 such customers as of January 31, "
                         "2024 and 2023, respectively.", "Customers")
    assert k.kpi_label(h) == "Customers_count"
    assert h["values"] == ["over 7,500", "over 7,000", "over 6,400"]
    assert "direct sales force" in k.kpi_name_as_written(h)


def test_definition_prefers_calculate_sentences():
    sents = ["Our net dollar expansion rate compares revenue across periods.",
             "Our net dollar expansion rate was 98% as of January 31, 2026.",
             "We calculate net dollar expansion rate by starting with ARR from 12 months prior."]
    d = k.find_definition(sents, "net dollar expansion rate")
    assert d.startswith("We calculate") and "98%" not in d


def test_ndr_fill_skips_qualitative_bounds_and_prefers_own_year():
    base = {"family": "NDR", "phrase": "net ARR expansion rate", "Ticker": "X", "para_id": "p",
            "period_as_written": "", "accession": "a"}
    hits = [{**base, "by_year": {2024: "over 120%"}, "report_year": 2024, "FiscalYear": "FY2024"},
            {**base, "by_year": {2025: "118%", 2024: "119%"}, "report_year": 2026, "FiscalYear": "FY2026"},
            {**base, "by_year": {2025: "approximately 117%"}, "report_year": 2025, "FiscalYear": "FY2025"}]
    best = k.ndr_by_period(hits)
    assert best[("X", 2025)]["val"] == "approximately 117%"   # own-year filing wins
    assert best[("X", 2024)]["val"] == "119%"                  # comparative; "over 120%" ignored
