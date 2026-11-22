"""Stage 4: KPIs from Item 7 paragraphs (offline).

Rules
- A KPI is reported only from a single sentence that contains BOTH the KPI phrase and
  a value of the right type (% for retention/churn, $ for ARR, a count next to
  "customers" for customer counts). "customers" alone never matches.
- Illustrative sentences ("for instance", "if each customer ...", "would be") are skipped.
- Values are mapped to periods only when the sentence's dates line up one-to-one with
  its values (or one value with one date). Otherwise the value is kept as written and
  the period is left unresolved (logged), never guessed.
- Definitions differ by company (ZM: trailing-12-month net dollar expansion rate for
  Enterprise customers; MDB: net ARR expansion rate; ESTC: Net Expansion Rate), so
  each value keeps its own kpi_name_as_written / kpi_definition; nothing is merged.

Outputs: outputs/kpis.csv, and NDR / LTV_CAC (+ source_notes) in
outputs/financial_metrics.csv. Run after src.fetch_financials and src.fetch_filings.

Usage:  python -m src.kpi_extract
"""
from __future__ import annotations

import re

import pandas as pd

from src.utils import load_config, root_path, write_issues

STAGE = "stage4"
ND = "Not Disclosed"
N_A = "n/a"

MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_RE = re.compile(MONTHS + r"\s+\d{1,2},\s*\d{4}(?:,?\s*(?:and\s+)?\d{4}\b)*")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PCT_RE = re.compile(r"(?:(?:approximately|over|about|more than)\s+)?\d+(?:\.\d+)?\s*%")
MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?", re.I)
ILLUSTRATIVE_RE = re.compile(r"\bfor (?:instance|example)\b|\bif each\b|\bwould be\b|\bhypothetical", re.I)
SENT_SPLIT = re.compile(r"(?<=[a-z0-9%)\]\"][.!?])\s+(?=[A-Z\"(])")
DEFINITION_STRONG = re.compile(r"\b(?:we calculate|is calculated|we define|is defined|defined as|we compute|calculated by)\b", re.I)
DEFINITION_WEAK = re.compile(r"\b(?:is the weighted average|is based (?:up)?on|includes|compares?|consists? of|divid)", re.I)

FAMILIES = {
    "NDR": {"phrase": re.compile(r"net\s+(?:dollar\s+|ARR\s+)?(?:retention|expansion)(?:\s+rate)?", re.I),
            "value": "pct"},
    "Churn": {"phrase": re.compile(r"(?:[\w-]+\s+){0,3}\bchurn(?:\s+rate)?\b", re.I), "value": "pct"},
    "ARR": {"phrase": re.compile(r"\b(?:total\s+)?(?:ARR|annual(?:ized)?\s+recurring\s+revenue)\b", re.I),
            "value": "money"},
    "Customers": {"phrase": re.compile(r"\bcustomers\b", re.I), "value": "count"},
}
# count list directly before "customers", e.g. "over 7,500 customers", "4,468, 4,088, and 3,810 customers",
# "over 7,000 and over 6,400 such customers", "approximately 192,600 ... Enterprise customers"
_NUM = r"(?:(?:over|approximately|about|more than)\s+)?\d{1,3}(?:,\d{3})+|(?:(?:over|approximately|about|more than)\s+)?\d+"
COUNT_BEFORE_RE = re.compile(
    rf"((?:{_NUM})(?:,?\s+(?:and\s+)?(?:{_NUM}))*)\s+((?:(?:[A-Z][\w-]*|such)\s+){{0,3}})customers")
COUNT_AFTER_RE = re.compile(r"number of ((?:[\w-]+\s+){0,2})customers\b(.*?)\b(?:was|were)\s+(.*)", re.I)
THRESHOLD_RE = re.compile(r"(?:(?:more|greater) than\s+)?\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion))?(?:\s+or greater)?"
                          r"(?:\s+(?:of|in)\s+.*?(?=\s+(?:was|were|as of)\b|,\s|;|\.\s|\.$|$))?", re.I)
SEGMENT_TAIL_RE = re.compile(r"\bcustomers\s+((?:that|who|across|sold)\b.*?)(?=,|;|\s+as of\b|\s+(?:as\s+)?compared\b|\.\s|\.$|$)", re.I)
NUM_IN_LIST = re.compile(r"((?:over|approximately|about|more than)\s+)?(\d{1,3}(?:,\d{3})+|\d+)")
BASIS_RE = re.compile(r"trailing[\s-]+(?:12|twelve)[\s-]+months?|per month|monthly|annual(?:ized)?|as of", re.I)


# ---------------------------------------------------------------------------
# Sentence-level parsing
# ---------------------------------------------------------------------------
def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def date_years(sentence: str) -> tuple[list[int], list[str]]:
    """Years in order of appearance inside date phrases, and the date phrases themselves."""
    phrases = DATE_RE.findall(sentence)
    years = [int(y) for ph in phrases for y in YEAR_RE.findall(ph)]
    return years, phrases


def _strip_noise(sentence: str) -> str:
    s = DATE_RE.sub(" ", sentence)
    return re.sub(r"\b\d+[\s-]+months?\b", " ", s)


def pct_values(sentence: str) -> list[str]:
    return [v.strip() for v in PCT_RE.findall(_strip_noise(sentence))]


def money_values(sentence: str) -> list[str]:
    return [v.strip() for v in MONEY_RE.findall(_strip_noise(sentence))]


def count_values(sentence: str) -> tuple[list[str], str]:
    """Customer counts as written and the segment words ('Enterprise', 'Atlas', ...)."""
    s = MONEY_RE.sub(" $ ", _strip_noise(sentence))
    m = COUNT_AFTER_RE.search(s)
    if m:
        vals = [a + b for a, b in NUM_IN_LIST.findall(m.group(3).split(" as of")[0])]
        return [v.strip() for v in vals], (m.group(1) + "customers").strip()
    found, segment = [], ""
    for m in COUNT_BEFORE_RE.finditer(s):
        found += [(a + b).strip() for a, b in NUM_IN_LIST.findall(m.group(1))]
        segment = segment or (m.group(2) + "customers").strip()
    return found, segment


def to_number(v: str) -> float:
    return float(re.sub(r"[^\d.]", "", v.split("%")[0]))


def parse_sentence(sentence: str, family: str) -> dict | None:
    """KPI hit for one sentence, or None. Requires phrase + value in the same sentence."""
    spec = FAMILIES[family]
    pm = spec["phrase"].search(sentence)
    if not pm or ILLUSTRATIVE_RE.search(sentence):
        return None
    if spec["value"] == "pct":
        values, extra = pct_values(sentence), {}
    elif spec["value"] == "money":
        values, extra = money_values(sentence), {}
        if family == "ARR" and not re.search(r"\b(?:ARR|recurring revenue)\b[^.]{0,40}\b(?:was|of|totaled|reached|grew to)\b\s*(?:approximately\s+|over\s+)?\$", sentence, re.I):
            return None  # ARR mentioned as a threshold/definition, not as a total amount
    else:
        values, segment = count_values(sentence)
        if not values:
            return None
        if "as of" not in sentence.lower():
            return None  # customer counts are stocks; require an "as of" date
        thr = THRESHOLD_RE.search(sentence)
        tail = SEGMENT_TAIL_RE.search(sentence)
        if tail and not thr:
            segment = f"{segment} {tail.group(1).strip()}"
        extra = {"threshold": thr.group(0).strip() if thr else "", "segment": segment}
    if not values:
        return None
    years, phrases = date_years(sentence)
    if len(values) == len(years):
        by_year = dict(zip(years, values))
    elif len(values) == 1 and len(years) == 1:
        by_year = {years[0]: values[0]}
    else:
        by_year = {}
    basis = sorted({b.lower() for b in BASIS_RE.findall(sentence)})
    return {"family": family, "phrase": pm.group(0).strip(), "values": values, "by_year": by_year,
            "period_as_written": "; ".join(phrases + [b for b in basis if b != "as of"]) or "not stated",
            "sentence": sentence, **extra}


def kpi_label(hit: dict) -> str:
    if hit["family"] != "Customers":
        return hit["family"]
    return "Customers_over_threshold" if hit.get("threshold") else "Customers_count"


def kpi_name_as_written(hit: dict) -> str:
    if hit["family"] == "Customers":
        if hit.get("threshold"):
            return f"{hit['segment']} ({hit['threshold']})"
        return hit["segment"]
    return hit["phrase"]


def find_definition(all_sentences: list[str], core: str, max_sentences: int = 2) -> str:
    """Up to `max_sentences` definitional sentences (as written) mentioning `core`.

    "We calculate / define ..." sentences rank first; sentences carrying a % value
    (i.e. the evidence itself) and illustrative examples are excluded.
    """
    key = re.compile(re.escape(core), re.I)
    cands = [s for s in all_sentences if key.search(s) and not ILLUSTRATIVE_RE.search(s)
             and not PCT_RE.search(s)]
    strong = [s for s in cands if DEFINITION_STRONG.search(s)]
    weak = [s for s in cands if DEFINITION_WEAK.search(s) and s not in strong]
    return " ".join(list(dict.fromkeys(strong + weak))[:max_sentences])


def definition_core(hit: dict) -> str:
    """Phrase used to locate definitions: the KPI phrase without leading qualifiers."""
    if hit["family"] == "Churn":
        return "churn"
    m = FAMILIES["NDR"]["phrase"].search(hit["phrase"])
    return m.group(0) if m else hit["phrase"]


# ---------------------------------------------------------------------------
# Filing-level extraction
# ---------------------------------------------------------------------------
def extract_filing_kpis(paras: pd.DataFrame, report_year: int) -> tuple[list[dict], dict]:
    """Hits for one filing's Item 7 paragraphs; plus phrase-mention counts for reasons."""
    hits, mentions = [], {f: 0 for f in FAMILIES}
    sents_all = []
    for _, r in paras.iterrows():
        for s in sentences(r.text):
            sents_all.append(s)
            for fam in FAMILIES:
                if FAMILIES[fam]["phrase"].search(s):
                    mentions[fam] += 1
                h = parse_sentence(s, fam)
                if h:
                    h["para_id"] = r.para_id
                    hits.append(h)
    for h in hits:
        h["current"] = h["by_year"].get(report_year)
        if h["family"] in ("NDR", "Churn", "ARR"):
            h["definition"] = find_definition(sents_all, definition_core(h)) or "Definition not stated in Item 7"
        else:
            h["definition"] = h["sentence"] if h.get("threshold") else f"{h['segment']} as of period end"
    return hits, mentions


def _value_kind(fam):
    return {"pct": "a percentage", "money": "a $ amount", "count": "a count next to 'customers' with an 'as of' date"}[FAMILIES[fam]["value"]]


def build_kpis(paragraphs: pd.DataFrame, index: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[str]]:
    rows, all_hits, issues = [], [], []
    filings = index[index.item == "7"]
    for _, f in filings.iterrows():
        t, fy = f.ticker, f.fiscal_year
        report_year = int(str(f.report_date)[:4])
        base = {"Ticker": t, "FiscalYear": fy, "PeriodEnd": f.report_date,
                "accession": f.accession, "filing_url": f.filing_url}
        if f.status != "ok":
            for fam in FAMILIES:
                rows.append({**base, "KPI": fam, "status": ND, "value_current_period": ND,
                             "reason": f"Item 7 not extracted ({f.status})"})
            issues.append(f"[{t} {fy}] all KPIs Not Disclosed: Item 7 not extracted ({f.status})")
            continue
        p = paragraphs[(paragraphs.accession == f.accession) & (paragraphs["item"].astype(str) == "7")]
        hits, mentions = extract_filing_kpis(p, report_year)
        for h in hits:
            h.update(Ticker=t, FiscalYear=fy, accession=f.accession, report_year=report_year)
        all_hits += hits
        seen = set()
        for fam in FAMILIES:
            fam_hits = [h for h in hits if h["family"] == fam]
            if not fam_hits:
                reason = (f"phrase appears in {mentions[fam]} Item 7 sentence(s) but never with {_value_kind(fam)} in the same sentence"
                          if mentions[fam] else f"no Item 7 sentence mentions the KPI phrase")
                rows.append({**base, "KPI": fam, "status": ND, "value_current_period": ND, "reason": reason})
                issues.append(f"[{t} {fy}] {fam} = Not Disclosed: {reason}")
                continue
            for h in fam_hits:
                vals = h["current"] or "|".join(h["values"])
                key = (kpi_label(h), kpi_name_as_written(h).lower(), re.sub(r"[^\d.|%]", "", vals))
                if key in seen:
                    continue  # same KPI + value repeated in another sentence
                seen.add(key)
                if h["current"] is None:
                    issues.append(f"[{t} {fy}] {kpi_label(h)} '{kpi_name_as_written(h)}': value(s) {h['values']} "
                                  f"not mapped to period {report_year} (dates in sentence: {h['period_as_written']}); kept as written")
                rows.append({**base, "KPI": kpi_label(h), "status": "Disclosed",
                             "kpi_name_as_written": kpi_name_as_written(h),
                             "value_current_period": h["current"] if h["current"] else N_A,
                             "values_as_written": ", ".join(h["values"]),
                             "period_as_written": h["period_as_written"],
                             "kpi_definition": h["definition"], "evidence_sentence": h["sentence"],
                             "para_id": h["para_id"], "reason": ""})
    cols = ["Ticker", "FiscalYear", "PeriodEnd", "KPI", "status", "kpi_name_as_written",
            "value_current_period", "values_as_written", "period_as_written", "kpi_definition",
            "evidence_sentence", "reason", "para_id", "accession", "filing_url"]
    return pd.DataFrame(rows).reindex(columns=cols).fillna(""), all_hits, issues


# ---------------------------------------------------------------------------
# financial_metrics.csv: NDR and LTV_CAC
# ---------------------------------------------------------------------------
_NOTE_TAG = "; [stage4] "


def ndr_by_period(hits: list[dict]) -> dict[tuple[str, int], dict]:
    """(ticker, period year) -> best NDR hit: own-year filing first, else latest filing."""
    best: dict[tuple[str, int], dict] = {}
    for h in hits:
        if h["family"] != "NDR":
            continue
        for year, val in h["by_year"].items():
            if "%" not in val or re.search(r"over|more than", val):
                continue  # qualitative bound ("over 120%"), not a point value
            rank = (year == h["report_year"], h["report_year"])
            k = (h["Ticker"], year)
            if k not in best or rank > best[k]["rank"]:
                best[k] = {"rank": rank, "val": val, "hit": h}
    return best


def update_financial_metrics(fm: pd.DataFrame, hits: list[dict], index: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    issues = []
    fm = fm.copy()
    fm["source_notes"] = fm["source_notes"].astype(str).str.split(re.escape(_NOTE_TAG)).str[0]
    best = ndr_by_period(hits)
    examined = {(r.ticker, r.fiscal_year) for r in index[(index.item == "7") & (index.status == "ok")].itertuples()}
    tick_has_ndr = {h["Ticker"] for h in hits if h["family"] == "NDR"}
    churn = {(h["Ticker"], h["FiscalYear"]) for h in hits if h["family"] == "Churn"}
    for i, r in fm.iterrows():
        year = int(str(r.PeriodEnd)[:4])
        notes = []
        b = best.get((r.Ticker, year))
        if b:
            h = b["hit"]
            fm.at[i, "NDR"] = round(to_number(b["val"]) / 100, 4)
            src = "own-year 10-K" if b["rank"][0] else f"comparative in FY{h['report_year']} 10-K"
            notes.append(f"NDR = '{h['phrase']}' as defined by {r.Ticker} ({b['val']}; {h['period_as_written']}); "
                         f"not comparable across companies; {src} {h['accession']} {h['para_id']}")
        elif (r.Ticker, r.FiscalYear) in examined:
            fm.at[i, "NDR"] = ND
            why = ("no point value for this period in its own 10-K Item 7" if r.Ticker in tick_has_ndr
                   else "no net retention/expansion rate in Item 7")
            issues.append(f"[{r.Ticker} {r.FiscalYear}] NDR = Not Disclosed: {why}")
        else:
            fm.at[i, "NDR"] = N_A
            issues.append(f"[{r.Ticker} {r.FiscalYear}] NDR = n/a: 10-K for this year not in fetched scope and no comparative value in later 10-Ks")
        # LTV/CAC needs churn, ARPU (LTV proxy) and new-customer count (CAC); no derivation.
        missing = ["ARPU", "new-customer count"] + ([] if (r.Ticker, r.FiscalYear) in churn else ["churn"])
        fm.at[i, "LTV_CAC"] = N_A
        issues.append(f"[{r.Ticker} {r.FiscalYear}] LTV_CAC = n/a: inputs not disclosed in Item 7: {', '.join(missing)}")
        if notes:
            fm.at[i, "source_notes"] = r.source_notes + _NOTE_TAG + " | ".join(notes)
    return fm, issues


def main() -> pd.DataFrame:
    cfg = load_config()
    proc, out = root_path(cfg["paths"]["processed"]), root_path(cfg["paths"]["outputs"])
    paragraphs = pd.read_csv(proc / "paragraphs.csv", keep_default_na=False, dtype={"item": str})
    index = pd.read_csv(proc / "filings_index.csv", keep_default_na=False, dtype={"item": str})
    kpis, hits, issues = build_kpis(paragraphs, index)
    kpis.to_csv(out / "kpis.csv", index=False)
    fm = pd.read_csv(out / "financial_metrics.csv", keep_default_na=False)
    fm, fm_issues = update_financial_metrics(fm, hits, index)
    fm.to_csv(out / "financial_metrics.csv", index=False)
    write_issues(cfg, STAGE, issues + fm_issues)
    print(f"Wrote outputs/kpis.csv ({len(kpis)} rows); updated NDR/LTV_CAC in outputs/financial_metrics.csv")
    return kpis


if __name__ == "__main__":
    main()
