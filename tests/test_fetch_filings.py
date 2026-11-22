"""Offline tests for 10-K section extraction in src/fetch_filings.py."""
from src import fetch_filings as fl

WORDS = " ".join(["competition"] * 30) + "."


def body(n):
    return "".join(f"<p>Paragraph {i} {WORDS}</p>" for i in range(n))


def doc(with_1a_end=True):
    toc = ("<table><tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>"
           "<tr><td>Item 1B. Unresolved Staff Comments</td></tr>"
           "<tr><td>Item 7. Management's Discussion and Analysis</td></tr>"
           "<tr><td>Item 7A. Quantitative and Qualitative Disclosures</td></tr></table>")
    s = ('<html><body><div style="display:none"><ix:header>hidden</ix:header></div>'
         '<ix:nonNumeric name="dei:DocumentFiscalYearFocus">2026</ix:nonNumeric>' + toc +
         "<p>Item 1A. Risk Factors</p>" + body(20) +
         "<p>Some fragment that breaks across a page</p><p>12</p><p>Table of Contents</p>"
         "<p>and continues here on the next page with enough extra words to pass the minimum length filter for sure ok.</p>")
    if with_1a_end:
        s += "<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>"
    s += ("<p>Item 7. Management\u2019s Discussion and Analysis of Financial Condition</p>" + body(25) +
          "<p>\uf0b7 bullet item " + WORDS + "</p>"
          "<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p></body></html>")
    return s


def test_extracts_body_not_toc_and_fy():
    res = fl.extract_filing(doc(), min_words=25)
    assert res["fy_tag"] == "2026"
    s1a, s7 = res["sections"]["1A"], res["sections"]["7"]
    assert s1a["status"] == "ok" and s7["status"] == "ok"
    assert len(s1a["paragraphs"]) == 21          # 20 + one re-joined page-break paragraph
    assert len(s7["paragraphs"]) == 26           # curly apostrophe heading matched
    assert s1a["paragraphs"][-1].startswith("Some fragment that breaks across a page and continues")
    assert s7["paragraphs"][-1].startswith("bullet item")  # symbol-font bullet stripped
    assert all("Item 7" not in p for p in s1a["paragraphs"])


def test_missing_end_heading_is_failure_not_guess():
    sec = fl.extract_filing(doc(with_1a_end=False), min_words=25)["sections"]["1A"]
    # Without Item 1B/1C/2 the only end candidate is gone -> no section is invented.
    assert sec["status"] == "end_not_found"
    assert "paragraphs" not in sec


def test_missing_start_and_too_short():
    lines = ["Item 7. Management's Discussion and Analysis", "short text",
             "Item 7A. Quantitative and Qualitative Disclosures"]
    assert fl.find_section(lines, "1A")["status"] == "start_not_found"
    assert fl.find_section(lines, "7")["status"] == "too_short"


def test_select_10ks_latest_originals_only():
    recent = {"form": ["10-K", "10-Q", "10-K/A", "10-K", "10-K", "10-K"],
              "accessionNumber": ["a-5", "q", "amend", "a-4", "a-3", "a-2"],
              "filingDate": ["2026-03-01", "x", "x", "2025-03-01", "2024-03-01", "2023-03-01"],
              "reportDate": ["2026-01-31", "x", "2025-01-31", "2025-01-31", "2024-01-31", "2023-01-31"],
              "primaryDocument": ["d5.htm", "q.htm", "am.htm", "d4.htm", "d3.htm", "d2.htm"]}
    got = fl.select_10ks({"filings": {"recent": recent}}, "0000000123", 3)
    assert [g["accession"] for g in got] == ["a-5", "a-4", "a-3"]
    assert got[0]["filing_url"] == "https://www.sec.gov/Archives/edgar/data/123/a5/d5.htm"


def test_para_hash_normalises_case_and_space():
    assert fl.para_hash("Intense  Competition\n now.") == fl.para_hash("intense competition now.")
    assert fl.para_hash("a b") != fl.para_hash("a c")
