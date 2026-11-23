"""Stage 7b: sensitivity analysis of MAS scores.

Reruns MAS with:
  1. Each anchor moved +/-20%
  2. Two alternative weight sets
  3. Financial parts removed (NLP only)

Outputs: outputs/sensitivity.csv

Usage:  python -m src.sensitivity
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_config, root_path, write_issues
from src import models, scoring as sc_mod

STAGE = "stage7b"
FORCES = [
    "Competitive_Rivalry", "Threat_of_New_Entry", "Threat_of_Substitutes",
    "Buyer_Power", "Supplier_Power", "Complementors",
]


def _run_scenario(cfg: dict, fm: pd.DataFrame, shares: pd.DataFrame,
                  evidence: pd.DataFrame, label: str) -> list[dict]:
    """Compute S_i and MAS for all tickers under modified cfg, return rows."""
    tickers = sorted(fm["Ticker"].unique())
    rows = []
    for t in tickers:
        s_scores = {}
        for f in FORCES:
            si, parts_used = sc_mod.compute_si(t, f, fm.copy(), shares.copy(), cfg)
            s_scores[f] = si
        mas_val = models.mas(s_scores, cfg["scoring"]["weights"])
        rows.append({"Scenario": label, "Ticker": t, "MAS": round(mas_val, 4),
                     **{f: round(s_scores[f], 4) for f in FORCES}})
    return rows


def _perturb_anchor(cfg: dict, anchor_name: str, field: str, factor: float) -> dict:
    """Return a copy of cfg with one anchor field multiplied by factor."""
    c = copy.deepcopy(cfg)
    c["scoring"]["anchors"][anchor_name][field] *= factor
    return c


def main() -> None:
    cfg = load_config()
    out = root_path(cfg["paths"]["outputs"])
    issues: list[str] = []

    fm = pd.read_csv(out / "financial_metrics.csv", keep_default_na=False)
    for col in ["GrossMargin", "Lerner_proxy", "DOL"]:
        fm[col] = pd.to_numeric(fm[col], errors="coerce")
    shares = pd.read_csv(out / "force_shares.csv", keep_default_na=False)
    evidence = pd.read_csv(out / "top_evidence.csv", keep_default_na=False)

    all_rows: list[dict] = []

    # --- Baseline ---
    all_rows += _run_scenario(cfg, fm, shares, evidence, "Baseline")

    # --- Anchors +/-20% ---
    anchor_perturbations = [
        ("mention_share", "value_at_10", 1.2, "mention_anchor_at10_+20%"),
        ("mention_share", "value_at_10", 0.8, "mention_anchor_at10_-20%"),
        ("gross_margin",  "value_at_1",  1.2, "gm_anchor_at1_+20%"),
        ("gross_margin",  "value_at_1",  0.8, "gm_anchor_at1_-20%"),
        ("dol",           "value_at_1",  1.2, "dol_anchor_at1_+20%"),
        ("dol",           "value_at_1",  0.8, "dol_anchor_at1_-20%"),
        ("lerner_trend",  "value_at_1",  1.2, "lerner_anchor_at1_+20%"),
        ("lerner_trend",  "value_at_1",  0.8, "lerner_anchor_at1_-20%"),
    ]
    for anchor_name, field, factor, label in anchor_perturbations:
        c2 = _perturb_anchor(cfg, anchor_name, field, factor)
        all_rows += _run_scenario(c2, fm, shares, evidence, label)

    # --- Alternative weight sets ---
    # 1. Favour Rivalry + Substitutes (w=0.25 each, rest split)
    w1 = {f: 0.10 for f in FORCES}
    w1["Competitive_Rivalry"] = 0.25
    w1["Threat_of_Substitutes"] = 0.25
    # Normalise
    total = sum(w1.values())
    w1 = {k: v / total for k, v in w1.items()}
    c3 = copy.deepcopy(cfg)
    c3["scoring"]["weights"] = w1
    all_rows += _run_scenario(c3, fm, shares, evidence, "AltWeights_RivalrySubstitutes")

    # 2. Favour Buyer + Supplier (w=0.25 each, rest split)
    w2 = {f: 0.10 for f in FORCES}
    w2["Buyer_Power"] = 0.25
    w2["Supplier_Power"] = 0.25
    total = sum(w2.values())
    w2 = {k: v / total for k, v in w2.items()}
    c4 = copy.deepcopy(cfg)
    c4["scoring"]["weights"] = w2
    all_rows += _run_scenario(c4, fm, shares, evidence, "AltWeights_BuyerSupplier")

    # --- NLP-only (remove financial part by zeroing out financial anchors' influence) ---
    c5 = copy.deepcopy(cfg)
    # Put all forces in nlp_only list
    c5["scoring"]["nlp_only"] = list(FORCES)
    all_rows += _run_scenario(c5, fm, shares, evidence, "NLP_only")

    # --- Build output ---
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "sensitivity.csv", index=False)

    # --- Print rank changes ---
    print("\n=== Sensitivity: Company ranks by MAS ===")
    scenarios = df["Scenario"].unique()
    for s in scenarios:
        sub = df[df["Scenario"] == s].sort_values("MAS", ascending=False)
        rank_str = " > ".join(f"{row['Ticker']}({row['MAS']:.1f})" for _, row in sub.iterrows())
        print(f"  {s:40s}: {rank_str}")

    # Flag if ranking changes vs baseline
    baseline_rank = df[df["Scenario"] == "Baseline"].sort_values("MAS", ascending=False)["Ticker"].tolist()
    for s in scenarios:
        if s == "Baseline":
            continue
        s_rank = df[df["Scenario"] == s].sort_values("MAS", ascending=False)["Ticker"].tolist()
        if s_rank != baseline_rank:
            issues.append(f"[{s}] ranking changes vs baseline: {s_rank} vs {baseline_rank}")
            print(f"  ** RANKING CHANGE in '{s}': {s_rank}")

    write_issues(cfg, STAGE, issues)
    print(f"\nWrote outputs/sensitivity.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
