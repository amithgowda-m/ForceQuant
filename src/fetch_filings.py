"""Stage 3: latest 10-Ks -> Item 1A / Item 7 -> data/processed/paragraphs.csv.

Network: only `fetch_submissions` / `fetch_filing_html` (via fetch_financials._cached).
Only the 10-Ks the plan needs are downloaded (latest + up to n_years-1 previous, form
"10-K" exactly; amendments are not used). Everything else is offline and testable.

Section rule: a section starts at a line that BEGINS with its Item heading and ends at
the next line that begins with the following Item heading. Table-of-contents hits are
discarded by taking the longest start->end span. No end heading, or a span below
`min_section_words`, is logged as a failure and the section is skipped (never guessed).

Usage:  python -m src.fetch_filings [--refresh]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.fetch_financials import _cached, check_user_agent, fiscal_year_labels, resolve_ciks
from src.utils import load_config, root_path, sec_user_agent, write_issues

STAGE = "stage3"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{doc}"
MIN_SECTION_WORDS = 500

# Heading patterns (applied to normalised lines; must match at line start).
_ITEM = r"^\s*item\s*"
SECTION_PATTERNS = {
    "1A": {
        "start": re.compile(_ITEM + r"1a\s*[\.:\-–—]?\s*risk\s+factors", re.I),
        "end": re.compile(_ITEM + r"(1b|1c|2)\s*[\.:\-–—]?\s*(unresolved|cybersecurity|properties)", re.I),
    },
    "7": {
        "start": re.compile(_ITEM + r"7\s*[\.:\-–—]?\s*management'?s\s+discussion", re.I),
        "end": re.compile(_ITEM + r"(7a\s*[\.:\-–—]?\s*quantitative|8\s*[\.:\-–—]?\s*financial\s+statements)", re.I),
    },
}
BLOCK_TAGS = ["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "br"]
_PAGE_NOISE = re.compile(r"^(table of contents|\d{1,3}|page \d+|[ivxlc]{1,6})$", re.I)
_TERMINAL = re.compile(r"[\.\?!:;\"”\)]$")


# ---------------------------------------------------------------------------
# Network (only here)
# ---------------------------------------------------------------------------
def fetch_submissions(cik: str, cfg: dict, refresh: bool = False) -> Path:
    return _cached(SUBMISSIONS_URL.format(cik=cik),
                   root_path(cfg["paths"]["raw"]) / "submissions" / f"CIK{cik}.json", cfg, refresh)


def fetch_filing_html(ticker: str, filing: dict, cfg: dict, refresh: bool = False) -> Path:
    path = root_path(cfg["paths"]["raw"]) / "filings" / ticker / f"{filing['accession']}.htm"
    return _cached(filing["filing_url"], path, cfg, refresh)


# ---------------------------------------------------------------------------
# Offline: filing selection
# ---------------------------------------------------------------------------
def select_10ks(submissions: dict, cik: str, n: int) -> list[dict]:
    """Latest n original 10-K filings from submissions['filings']['recent']."""
    r = submissions["filings"]["recent"]
    rows = []
    for i, form in enumerate(r["form"]):
        if form != "10-K":
            continue
        accn = r["accessionNumber"][i]
        rows.append({
            "accession": accn,
            "filing_date": r["filingDate"][i],
            "report_date": r["reportDate"][i],
            "primary_doc": r["primaryDocument"][i],
            "filing_url": ARCHIVE_URL.format(cik_int=int(cik), accn_nodash=accn.replace("-", ""),
                                             doc=r["primaryDocument"][i]),
        })
    rows.sort(key=lambda x: x["report_date"], reverse=True)
    return rows[:n]


# ---------------------------------------------------------------------------
# Offline: HTML -> lines
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = "".join(c for c in text if unicodedata.category(c) != "Co")  # symbol-font bullets
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^[•▪●◦‣·]+\s*", "", text)  # leading bullets


def html_to_lines(html: str | bytes) -> tuple[list[str], str | None]:
    """Block-level text lines (normalised) and dei:DocumentFiscalYearFocus if tagged."""
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # iXBRL is XHTML
    soup = BeautifulSoup(html, "lxml")
    fy_tag = soup.find(attrs={"name": re.compile(r"^dei:DocumentFiscalYearFocus$", re.I)})
    fy = normalise(fy_tag.get_text()) if fy_tag else None
    for t in soup.find_all(["ix:header", "script", "style"]):
        t.decompose()
    for t in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        t.decompose()
    for t in soup.find_all(BLOCK_TAGS):
        t.insert_after("\n")
        if t.name == "br":
            t.replace_with("\n")
    raw = soup.get_text("")
    lines = [normalise(l) for l in raw.split("\n")]
    return [l for l in lines if l], (fy if fy and re.fullmatch(r"\d{4}", fy) else None)


# ---------------------------------------------------------------------------
# Offline: sections and paragraphs
# ---------------------------------------------------------------------------
def find_section(lines: list[str], item: str) -> dict:
    """Longest span from a start heading to the next end heading.

    Returns {"status": "ok", "start", "end", "words", "heading"} or
    {"status": "<failure>", "detail": ...}.
    """
    pat = SECTION_PATTERNS[item]
    starts = [i for i, l in enumerate(lines) if len(l.split()) <= 25 and pat["start"].search(l)]
    ends = [i for i, l in enumerate(lines) if len(l.split()) <= 25 and pat["end"].search(l)]
    if not starts:
        return {"status": "start_not_found", "detail": f"no line starting with Item {item} heading"}
    best = None
    for s in starts:
        nxt = [e for e in ends if e > s]
        if not nxt:
            continue
        e = nxt[0]
        words = sum(len(l.split()) for l in lines[s + 1:e])
        if best is None or words > best["words"]:
            best = {"status": "ok", "start": s, "end": e, "words": words, "heading": lines[s]}
    if best is None:
        return {"status": "end_not_found",
                "detail": f"{len(starts)} start heading(s) for Item {item}, no following end heading"}
    if best["words"] < MIN_SECTION_WORDS:
        return {"status": "too_short",
                "detail": f"longest Item {item} span has {best['words']} words (< {MIN_SECTION_WORDS})"}
    return best


def para_hash(text: str) -> str:
    """sha256 (first 16 hex) of lowercased, whitespace-normalised text."""
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()[:16]


def split_paragraphs(lines: list[str], min_words: int) -> list[str]:
    """Drop page noise, re-join lines broken by page breaks, keep paragraphs >= min_words."""
    body = [l for l in lines if not _PAGE_NOISE.match(l)]
    merged: list[str] = []
    for l in body:
        if merged and not _TERMINAL.search(merged[-1]) and l[:1].islower():
            merged[-1] = f"{merged[-1]} {l}"
        else:
            merged.append(l)
    return [p for p in merged if len(p.split()) >= min_words]


def extract_filing(html: str | bytes, min_words: int, items=("1A", "7")) -> dict:
    lines, fy = html_to_lines(html)
    out = {"fy_tag": fy, "sections": {}}
    for item in items:
        sec = find_section(lines, item)
        if sec["status"] == "ok":
            sec["paragraphs"] = split_paragraphs(lines[sec["start"] + 1:sec["end"]], min_words)
        out["sections"][item] = sec
    return out


def main(refresh: bool = False) -> pd.DataFrame:
    cfg = load_config()
    cfg["sec_user_agent"] = sec_user_agent(cfg)
    check_user_agent(cfg["sec_user_agent"])
    raw = root_path(cfg["paths"]["raw"])
    tickers = [c["ticker"] for c in cfg["companies"]]
    n_years = cfg["filings"]["n_years"]
    min_words = cfg["filings"]["min_paragraph_words"]
    items = cfg["filings"]["items"]

    with open(raw / "company_tickers.json", encoding="utf-8") as f:  # cached in Stage 2
        ciks = resolve_ciks(json.load(f), tickers)

    paras, index, issues = [], [], []
    for t in tickers:
        print(f"{t}:")
        with open(fetch_submissions(ciks[t], cfg, refresh), encoding="utf-8") as f:
            filings = select_10ks(json.load(f), ciks[t], n_years)
        if len(filings) < n_years:
            issues.append(f"[{t}] only {len(filings)} original 10-K filings found (wanted {n_years})")
        cf_path = raw / "companyfacts" / f"CIK{ciks[t]}.json"
        cf_labels = {}
        if cf_path.exists():
            with open(cf_path, encoding="utf-8") as f:
                cf_labels = fiscal_year_labels(json.load(f), cfg)

        for fl in filings:
            html = fetch_filing_html(t, fl, cfg, refresh).read_bytes()
            res = extract_filing(html, min_words, items)
            fy_tag, fy_cf = res["fy_tag"], cf_labels.get(fl["report_date"])
            if fy_tag:
                fy = int(fy_tag)
                if fy_cf is not None and fy_cf != fy:
                    issues.append(f"[{t} {fl['accession']}] fiscal-year mismatch: dei tag FY{fy} vs companyfacts FY{fy_cf}; using dei tag")
            elif fy_cf is not None:
                fy = fy_cf
                issues.append(f"[{t} {fl['accession']}] no dei:DocumentFiscalYearFocus tag; FY{fy} from companyfacts for period {fl['report_date']}")
            else:
                fy = None
                issues.append(f"[{t} {fl['accession']}] fiscal year unknown (no dei tag, no companyfacts match); label n/a")
            fy_label = f"FY{fy}" if fy else "n/a"

            for item in items:
                sec = res["sections"][item]
                rec = {"ticker": t, "fiscal_year": fy_label, "item": item, "accession": fl["accession"],
                       "report_date": fl["report_date"], "filing_date": fl["filing_date"],
                       "filing_url": fl["filing_url"], "status": sec["status"],
                       "heading": sec.get("heading", ""), "section_words": sec.get("words", 0),
                       "n_paragraphs": len(sec.get("paragraphs", []))}
                index.append(rec)
                if sec["status"] != "ok":
                    issues.append(f"[{t} {fy_label} Item {item}] extraction failed ({sec['status']}): {sec['detail']}; section skipped")
                    continue
                for k, text in enumerate(sec["paragraphs"], 1):
                    paras.append({"para_id": f"{t}_{fy_label}_{item}_{k:04d}", "ticker": t,
                                  "fiscal_year": fy_label, "item": item, "para_idx": k,
                                  "n_words": len(text.split()), "para_hash": para_hash(text), "text": text,
                                  "accession": fl["accession"], "filing_url": fl["filing_url"]})
            print(f"  {fy_label} {fl['accession']}: " + ", ".join(
                f"Item {i}={res['sections'][i]['status']}/{len(res['sections'][i].get('paragraphs', []))}p" for i in items))

    proc = root_path(cfg["paths"]["processed"])
    proc.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(paras)
    df.to_csv(proc / "paragraphs.csv", index=False)
    pd.DataFrame(index).to_csv(proc / "filings_index.csv", index=False)
    write_issues(cfg, STAGE, issues)
    print(f"Wrote data/processed/paragraphs.csv ({len(df)} paragraphs) and filings_index.csv")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    main(ap.parse_args().refresh)
