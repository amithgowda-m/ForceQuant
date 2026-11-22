"""Offline tests for the extraction logic in src/fetch_financials.py (no network)."""
import pytest

from src import fetch_financials as ff


def fact(val, start, end, fy, accn, filed, form="10-K", fp="FY"):
    return {"val": val, "start": start, "end": end, "fy": fy, "fp": fp,
            "form": form, "accn": accn, "filed": filed}


@pytest.fixture
def facts():
    # FY2024 10-K (current period ends 2024-01-31) carries FY2023 comparative;
    # FY2023 value is restated in the later filing (latest filed wins).
    rev = [
        fact(100, "2022-02-01", "2023-01-31", 2023, "A-23", "2023-03-15"),
        fact(105, "2022-02-01", "2023-01-31", 2024, "A-24", "2024-03-15"),  # restated
        fact(130, "2023-02-01", "2024-01-31", 2024, "A-24", "2024-03-15"),
        fact(30, "2023-11-01", "2024-01-31", 2024, "A-24", "2024-03-15"),   # quarter -> excluded
        fact(999, "2023-02-01", "2024-01-31", 2024, "Q-24", "2023-12-01", form="10-Q", fp="Q3"),
    ]
    cogs = [
        fact(40, "2022-02-01", "2023-01-31", 2023, "A-23", "2023-03-15"),
        fact(52, "2023-02-01", "2024-01-31", 2024, "A-24", "2024-03-15"),
    ]
    opinc = [
        fact(-5, "2022-02-01", "2023-01-31", 2023, "A-23", "2023-03-15"),
        fact(26, "2023-02-01", "2024-01-31", 2024, "A-24", "2024-03-15"),
    ]
    sga = [fact(20, "2023-02-01", "2024-01-31", 2024, "A-24", "2024-03-15")]
    usd = lambda rows: {"units": {"USD": rows}}
    return {"facts": {"us-gaap": {
        # FY2023 only under the old tag, FY2024 only under the ASC 606 tag -> fallback
        "Revenues": usd([r for r in rev if r["end"] == "2023-01-31"]),
        "RevenueFromContractWithCustomerExcludingAssessedTax": usd([r for r in rev if r["end"] != "2023-01-31"]),
        "CostOfRevenue": usd(cogs),
        "OperatingIncomeLoss": usd(opinc),
        "SellingGeneralAndAdministrativeExpense": usd(sga),
    }}}


@pytest.fixture
def cfg():
    from src.utils import load_config
    return load_config()


def test_resolve_ciks_pads_to_10_digits():
    tj = {"0": {"cik_str": 1585521, "ticker": "ZM", "title": "Zoom"}}
    assert ff.resolve_ciks(tj, ["ZM"]) == {"ZM": "0001585521"}
    with pytest.raises(KeyError):
        ff.resolve_ciks(tj, ["XXX"])


def test_fiscal_year_label_from_current_period_filing(facts, cfg):
    labels = ff.fiscal_year_labels(facts, cfg)
    assert labels == {"2023-01-31": 2023, "2024-01-31": 2024}


def test_latest_filed_wins_and_quarters_excluded(facts, cfg):
    hit = ff.pick_metric(facts, cfg["financials"]["tags"]["Revenue"], "2023-01-31", cfg)
    assert hit["val"] == 105 and hit["accn"] == "A-24"
    hit = ff.pick_metric(facts, cfg["financials"]["tags"]["Revenue"], "2024-01-31", cfg)
    assert hit["val"] == 130
    assert hit["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_build_metrics_rules(facts, cfg):
    rows, issues = ff.build_metrics("TST", facts, cfg)
    by_fy = {r["FiscalYear"]: r for r in rows}
    assert set(by_fy) == {"FY2023", "FY2024"}
    r24 = by_fy["FY2024"]
    assert r24["GrossMargin"] == pytest.approx((130 - 52) / 130)
    assert r24["Lerner_proxy"] == r24["GrossMargin"]
    assert r24["DOL"] == pytest.approx((130 - 52) / 26)
    assert by_fy["FY2023"]["DOL"] == "n/m"
    for col in ("R&D", "S&M", "G&A", "LTV_CAC", "NDR"):
        assert r24[col] == "n/a"
    assert "Revenue=RevenueFromContractWithCustomerExcludingAssessedTax@A-24" in r24["source_notes"]
    assert any("FY2024] S&M = n/a" in i and "SG&A" in i for i in issues)
    assert any("FY2023] DOL = n/m" in i for i in issues)


def test_placeholder_user_agent_rejected():
    for ua in ("ForceQuant-RVCE [YOUR NAME] [YOUR EMAIL]", "no-email-here", ""):
        with pytest.raises(SystemExit):
            ff.check_user_agent(ua)
    ff.check_user_agent("ForceQuant-RVCE someone someone@example.com")


def test_extraction_needs_no_network(facts, cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call during offline extraction")
    monkeypatch.setattr(ff, "_get", boom)
    ff.build_metrics("TST", facts, cfg)
