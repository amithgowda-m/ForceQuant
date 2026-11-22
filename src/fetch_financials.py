"""Stage 2: SEC companyfacts -> outputs/financial_metrics.csv.

Network calls happen only in the `fetch_*` / `_get` functions below; everything is
cached in data/raw/. Extraction (`extract_*`, `build_metrics`) is offline and testable.

Usage:  python -m src.fetch_financials [--refresh]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src import models
from src.utils import load_config, root_path, sec_user_agent, write_issues

STAGE = "stage2"
N_A = models.N_A
METRICS = ["Revenue", "COGS", "RD", "SM", "GA", "OpInc"]
OUT_COLS = ["Ticker", "FiscalYear", "Revenue", "COGS", "R&D", "S&M", "G&A", "OpInc",
            "GrossMargin", "Lerner_proxy", "DOL", "LTV_CAC", "NDR", "source_notes",
            "PeriodEnd"]
COL_NAME = {"Revenue": "Revenue", "COGS": "COGS", "RD": "R&D", "SM": "S&M",
            "GA": "G&A", "OpInc": "OpInc"}


# ---------------------------------------------------------------------------
# Network (only here)
# ---------------------------------------------------------------------------
def check_user_agent(ua: str) -> None:
    if not ua or "[" in ua or "]" in ua or "@" not in ua or "CHANGE_ME" in ua:
        raise SystemExit("SEC_USER_AGENT is not set (or is a placeholder). Put a real name and email in .env; see .env.example.")


def _get(url: str, cfg: dict) -> bytes:
    import requests  # imported here so offline modules never need it

    headers = {"User-Agent": cfg["sec_user_agent"], "Accept-Encoding": "gzip, deflate"}
    for attempt in range(4):
        resp = requests.get(url, headers=headers, timeout=60)
        time.sleep(cfg["sec_request_sleep_s"])
        if resp.status_code == 200:
            return resp.content
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.content


def _cached(url: str, path: Path, cfg: dict, refresh: bool) -> Path:
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(url, cfg))
        print(f"  downloaded {url}")
    else:
        print(f"  cached     {path.relative_to(root_path('.'))}")
    return path


def fetch_company_tickers(cfg: dict, refresh: bool = False) -> Path:
    return _cached(cfg["sec_company_tickers_url"],
                   root_path(cfg["paths"]["raw"]) / "company_tickers.json", cfg, refresh)


def fetch_companyfacts(cik: str, cfg: dict, refresh: bool = False) -> Path:
    url = cfg["financials"]["companyfacts_url"].format(cik=cik)
    return _cached(url, root_path(cfg["paths"]["raw"]) / "companyfacts" / f"CIK{cik}.json", cfg, refresh)


# ---------------------------------------------------------------------------
# Offline extraction
# ---------------------------------------------------------------------------
def resolve_ciks(tickers_json: dict, tickers: list[str]) -> dict[str, str]:
    """Map ticker -> 10-digit CIK from company_tickers.json."""
    by_ticker = {row["ticker"].upper(): row["cik_str"] for row in tickers_json.values()}
    out = {}
    for t in tickers:
        if t.upper() not in by_ticker:
            raise KeyError(f"ticker {t} not in company_tickers.json")
        out[t] = f"{int(by_ticker[t.upper()]):010d}"
    return out


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def annual_facts(facts: dict, tag: str, days_window: tuple[int, int]) -> list[dict]:
    """All USD duration facts for `tag` from 10-K filings with fp == FY and ~1-year span."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return []
    lo, hi = days_window
    rows = []
    for e in node.get("units", {}).get("USD", []):
        if e.get("form") != "10-K" or e.get("fp") != "FY" or "start" not in e:
            continue
        if lo <= _days(e["start"], e["end"]) <= hi:
            rows.append({**e, "tag": tag})
    return rows


def fiscal_year_labels(facts: dict, cfg: dict) -> dict[str, int]:
    """period end -> fiscal-year label (`fy`) of the 10-K whose CURRENT period ends there.

    companyfacts `fy` is the fiscal year of the filing, which also carries prior-year
    comparatives; so a period's label is taken from the filing where it is the latest period.
    """
    window = tuple(cfg["financials"]["annual_days"])
    by_accn: dict[str, dict] = {}
    for metric in METRICS:
        for tag in cfg["financials"]["tags"][metric]:
            for e in annual_facts(facts, tag, window):
                cur = by_accn.setdefault(e["accn"], {"end": e["end"], "fy": e["fy"]})
                if e["end"] > cur["end"]:
                    cur["end"] = e["end"]
    labels: dict[str, int] = {}
    for accn, v in sorted(by_accn.items(), key=lambda kv: kv[0]):
        if v["fy"] is not None:
            labels[v["end"]] = v["fy"]  # later accession (same period) overwrites
    return labels


def pick_metric(facts: dict, tags: list[str], period_end: str, cfg: dict) -> dict | None:
    """First tag in the fallback list with a value for period_end; latest filed wins."""
    window = tuple(cfg["financials"]["annual_days"])
    for tag in tags:
        cands = [e for e in annual_facts(facts, tag, window) if e["end"] == period_end]
        if cands:
            best = max(cands, key=lambda e: (e["filed"], e["accn"]))
            return {"val": best["val"], "tag": tag, "accn": best["accn"], "filed": best["filed"]}
    return None


def build_metrics(ticker: str, facts: dict, cfg: dict) -> tuple[list[dict], list[str]]:
    """Rows for the latest n_years fiscal years, plus issue lines."""
    issues: list[str] = []
    labels = fiscal_year_labels(facts, cfg)
    ends = sorted(labels, reverse=True)[: cfg["financials"]["n_years"]]
    if not ends:
        issues.append(f"[{ticker}] no annual 10-K facts found for any configured tag")
    rows = []
    for end in sorted(ends):
        fy = labels[end]
        row = {"Ticker": ticker, "FiscalYear": f"FY{fy}", "PeriodEnd": end}
        notes = []
        vals = {}
        for metric in METRICS:
            hit = pick_metric(facts, cfg["financials"]["tags"][metric], end, cfg)
            if hit is None:
                vals[metric] = N_A
                tried = ", ".join(cfg["financials"]["tags"][metric])
                reason = f"no 10-K FY value for tags [{tried}]"
                if metric == "SM" and pick_metric(facts, ["SellingGeneralAndAdministrativeExpense"], end, cfg):
                    reason += " (only SG&A reported; not split, left n/a)"
                if metric == "GA" and pick_metric(facts, ["SellingGeneralAndAdministrativeExpense"], end, cfg):
                    reason += " (only SG&A reported; not split, left n/a)"
                issues.append(f"[{ticker} FY{fy}] {COL_NAME[metric]} = n/a: {reason}")
            else:
                vals[metric] = hit["val"]
                notes.append(f"{COL_NAME[metric]}={hit['tag']}@{hit['accn']}")
            row[COL_NAME[metric]] = vals[metric]

        gm = models.lerner_proxy(vals["Revenue"], vals["COGS"])
        row["GrossMargin"] = gm
        row["Lerner_proxy"] = gm  # same filings proxy, (Revenue - COGS) / Revenue (eq. 12)
        if gm == N_A:
            issues.append(f"[{ticker} FY{fy}] GrossMargin/Lerner_proxy = n/a: Revenue or COGS missing")
        d = models.dol_empirical(vals["Revenue"], vals["COGS"], vals["OpInc"])
        row["DOL"] = d
        if d == models.N_M:
            issues.append(f"[{ticker} FY{fy}] DOL = n/m: OperatingIncome <= 0 ({vals['OpInc']})")
        elif d == N_A:
            issues.append(f"[{ticker} FY{fy}] DOL = n/a: Revenue, COGS or OpInc missing")
        row["LTV_CAC"] = N_A
        issues.append(f"[{ticker} FY{fy}] LTV_CAC = n/a: churn and new-customer counts are not XBRL facts in companyfacts")
        row["NDR"] = N_A
        issues.append(f"[{ticker} FY{fy}] NDR = n/a: not an XBRL fact; text KPI from Item 7 is extracted in Stage 4")
        row["source_notes"] = "; ".join(notes) + f"; FiscalYear label = filing fy, period end {end}"
        rows.append(row)
    return rows, issues


def main(refresh: bool = False) -> pd.DataFrame:
    cfg = load_config()
    cfg["sec_user_agent"] = sec_user_agent(cfg)  # env / .env overrides config placeholder
    check_user_agent(cfg["sec_user_agent"])
    tickers = [c["ticker"] for c in cfg["companies"]]

    print("Fetching (cached in data/raw/):")
    with open(fetch_company_tickers(cfg, refresh), encoding="utf-8") as f:
        ciks = resolve_ciks(json.load(f), tickers)
    paths = {t: fetch_companyfacts(ciks[t], cfg, refresh) for t in tickers}

    all_rows, issues = [], []
    for t in tickers:
        with open(paths[t], encoding="utf-8") as f:
            facts = json.load(f)
        rows, iss = build_metrics(t, facts, cfg)
        all_rows += rows
        issues += [f"[{t}] CIK {ciks[t]} resolved from company_tickers.json"] + iss

    df = pd.DataFrame(all_rows)[OUT_COLS]
    out = root_path(cfg["paths"]["outputs"]) / "financial_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    write_issues(cfg, STAGE, issues)
    print(f"Wrote {out.relative_to(root_path('.'))} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    main(ap.parse_args().refresh)
