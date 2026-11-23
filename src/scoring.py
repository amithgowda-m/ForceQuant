"""Stage 7: S_i scores, MAS, forces_summary.csv.

Usage:  python -m src.scoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_config, root_path, write_issues
from src import models

STAGE = "stage7"
N_A = models.N_A
N_M = models.N_M
FORCES = [
    "Competitive_Rivalry", "Threat_of_New_Entry", "Threat_of_Substitutes",
    "Buyer_Power", "Supplier_Power", "Complementors",
]


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------
def linear_scale(value: float, v_at_1: float, v_at_10: float) -> float:
    """Map `value` linearly from [v_at_1, v_at_10] -> [1, 10], clamped to [1, 10]."""
    if v_at_1 == v_at_10:
        return 5.0
    slope = 9.0 / (v_at_10 - v_at_1)
    s = 1.0 + slope * (value - v_at_1)
    return float(np.clip(s, 1.0, 10.0))


def mention_score(share: float, anchors: dict) -> float:
    return linear_scale(share, anchors["value_at_1"], anchors["value_at_10"])


def gross_margin_score(gm, anchors: dict):
    if isinstance(gm, str) or gm is None:
        return None
    return linear_scale(float(gm), anchors["value_at_1"], anchors["value_at_10"])


def dol_score(dol_val, anchors: dict):
    if isinstance(dol_val, str):  # "n/m" or "n/a"
        return None
    if dol_val is None:
        return None
    return linear_scale(float(dol_val), anchors["value_at_1"], anchors["value_at_10"])


def lerner_trend_score(ticker: str, fm: pd.DataFrame, anchors: dict):
    """OLS slope of Lerner_proxy over last 3 fiscal years available."""
    if fm.empty or "Ticker" not in fm.columns or "Lerner_proxy" not in fm.columns:
        return None
    sub = fm[fm["Ticker"] == ticker].copy()
    if sub.empty:
        return None
    # Keep only rows where Lerner_proxy is numeric
    sub = sub[sub["Lerner_proxy"].apply(lambda v: isinstance(v, float) or (
        isinstance(v, str) and v not in (N_A, N_M, "Not Disclosed", "")
    ))]
    if sub.empty:
        return None
    sub["lp"] = pd.to_numeric(sub["Lerner_proxy"], errors="coerce")
    sub = sub.dropna(subset=["lp"]).sort_values("FiscalYear")
    n_years = anchors.get("n_years", 3)
    sub = sub.tail(n_years)
    if len(sub) < 2:
        return None
    x = np.arange(len(sub), dtype=float)
    y = sub["lp"].values
    slope = float(np.polyfit(x, y, 1)[0])
    return linear_scale(slope, anchors["value_at_1"], anchors["value_at_10"])


# ---------------------------------------------------------------------------
# S_i computation per company per force
# ---------------------------------------------------------------------------
def compute_si(ticker: str, force: str, fm: pd.DataFrame,
               shares: pd.DataFrame, cfg: dict) -> tuple[float, list[str]]:
    """Return (S_i, parts_used_list)."""
    scoring = cfg["scoring"]
    anchors = scoring["anchors"]
    nlp_only = scoring.get("nlp_only", [])

    parts, parts_used = [], []

    # NLP salience part (all forces)
    row = shares[(shares["Ticker"] == ticker) & (shares["Force"] == force)]
    if not row.empty:
        share = float(row["MentionShare"].iloc[0])
        ms = mention_score(share, anchors["mention_share"])
        parts.append(ms)
        parts_used.append(f"NLP_share={share:.4f}->S={ms:.2f}")

    # Financial part
    if force not in nlp_only:
        # Latest fiscal year row for this ticker
        fm_t = fm[fm["Ticker"] == ticker].sort_values("FiscalYear")

        if force == "Buyer_Power":
            gm_row = fm_t.dropna(subset=["GrossMargin"])
            # also exclude string values
            gm_row = gm_row[gm_row["GrossMargin"].apply(lambda v: isinstance(v, float) or (
                isinstance(v, str) and v not in (N_A, N_M, "Not Disclosed", "")))]
            if not gm_row.empty:
                latest_gm = pd.to_numeric(gm_row.iloc[-1]["GrossMargin"], errors="coerce")
                if pd.notna(latest_gm):
                    s_fin = gross_margin_score(latest_gm, anchors["gross_margin"])
                    if s_fin is not None:
                        parts.append(s_fin)
                        parts_used.append(f"GrossMargin={latest_gm:.4f}->S={s_fin:.2f}")

        elif force == "Threat_of_New_Entry":
            dol_row = fm_t.copy()
            dol_row["dol_num"] = pd.to_numeric(dol_row["DOL"], errors="coerce")
            dol_row = dol_row.dropna(subset=["dol_num"]).tail(1)
            if not dol_row.empty:
                s_fin = dol_score(float(dol_row.iloc[0]["dol_num"]), anchors["dol"])
                if s_fin is not None:
                    parts.append(s_fin)
                    parts_used.append(f"DOL={dol_row.iloc[0]['dol_num']:.4f}->S={s_fin:.2f}")

        elif force in ("Competitive_Rivalry", "Threat_of_Substitutes"):
            s_fin = lerner_trend_score(ticker, fm_t, anchors["lerner_trend"])
            if s_fin is not None:
                parts.append(s_fin)
                parts_used.append(f"LernerTrend->S={s_fin:.2f}")

    si = float(np.mean(parts)) if parts else 5.0  # default mid if no data
    # Clamp to [1, 10]
    si = float(np.clip(si, 1.0, 10.0))
    if not parts:
        parts_used.append("default=5.0 (no data)")
    return si, parts_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict[str, float]:
    cfg = load_config()
    forces = cfg["forces"]
    weights = cfg["scoring"]["weights"]
    out = root_path(cfg["paths"]["outputs"])
    issues: list[str] = []

    # Load inputs
    fm = pd.read_csv(out / "financial_metrics.csv", keep_default_na=False)
    # Convert numeric columns
    for col in ["GrossMargin", "Lerner_proxy", "DOL"]:
        fm[col] = pd.to_numeric(fm[col], errors="coerce")

    shares = pd.read_csv(out / "force_shares.csv", keep_default_na=False)
    evidence = pd.read_csv(out / "top_evidence.csv", keep_default_na=False)

    tickers = sorted(fm["Ticker"].unique())

    # Print anchor assumptions
    print("\n=== SCORING ANCHORS (assumptions) ===")
    for name, a in cfg["scoring"]["anchors"].items():
        if isinstance(a, dict) and "basis_at_1" in a:
            print(f"  {name}: [{a['value_at_1']}->{a['value_at_10']}] "
                  f"basis_at_1={a['basis_at_1']}, basis_at_10={a['basis_at_10']}")
            print(f"    note: {a.get('note','')}")
    print()

    rows = []
    mas_by_ticker: dict[str, float] = {}
    for t in tickers:
        scores: dict[str, float] = {}
        for f in forces:
            si, parts_used = compute_si(t, f, fm, shares, cfg)
            # Grab evidence
            ev_row = evidence[(evidence["Ticker"] == t) & (evidence["Force"] == f)]
            top_quote = ev_row["TopEvidenceQuote"].iloc[0] if not ev_row.empty else ""
            filing_url = ev_row["FilingURL"].iloc[0] if not ev_row.empty else ""
            top_prob = ev_row["TopProbability"].iloc[0] if not ev_row.empty else ""
            shares_row = shares[(shares["Ticker"] == t) & (shares["Force"] == f)]
            mention_share = float(shares_row["MentionShare"].iloc[0]) if not shares_row.empty else 0.0
            scores[f] = si
            rows.append({
                "Ticker": t, "Force": f, "MentionShare": mention_share,
                "S_i": round(si, 4), "parts_used": " | ".join(parts_used),
                "TopEvidenceQuote": top_quote, "FilingURL": filing_url,
                "TopProbability": top_prob,
            })

        # MAS
        mas_val = models.mas(scores, weights)
        mas_by_ticker[t] = round(mas_val, 4)
        for row in rows:
            if row["Ticker"] == t:
                row["MAS"] = round(mas_val, 4)

        print(f"{t}: MAS={mas_val:.2f}  forces={scores}")

    df = pd.DataFrame(rows, columns=[
        "Ticker", "Force", "MentionShare", "S_i", "MAS",
        "parts_used", "TopEvidenceQuote", "TopProbability", "FilingURL",
    ])
    df.to_csv(out / "forces_summary.csv", index=False)

    write_issues(cfg, STAGE, issues)
    print(f"\nWrote outputs/forces_summary.csv ({len(df)} rows)")
    return mas_by_ticker


if __name__ == "__main__":
    main()
