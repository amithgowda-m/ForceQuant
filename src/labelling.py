"""Stage 5: stratified labelling sample and human check sheet (offline).

The llm_label values themselves are written by Claude Code in-session (batches of ~25,
following LABEL_GUIDE.md) into data/processed/llm_label_batches/batch_XX.csv. No external
API is called. This module only samples, merges the batches and exports the check sheet.

Usage:
  python -m src.labelling sample      # -> data/processed/label_sample.csv
  python -m src.labelling merge       # batches -> data/processed/llm_labels.csv + counts
  python -m src.labelling checksheet  # -> data/processed/human_check_sheet.csv
"""
from __future__ import annotations

import sys

import pandas as pd

from src.utils import load_config, root_path

PER_CELL = 20
CHECK_N = 40
PER_FORCE_IN_CHECK = 6


def _proc(cfg):
    return root_path(cfg["paths"]["processed"])


def unique_paragraphs(paragraphs: pd.DataFrame) -> pd.DataFrame:
    """One row per para_hash: the latest fiscal year's copy (Item 1A before Item 7 on ties)."""
    p = paragraphs.sort_values(["fiscal_year", "item", "para_idx"], ascending=[False, True, True])
    return p.drop_duplicates("para_hash", keep="first")


def stratified_sample(paragraphs: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    u = unique_paragraphs(paragraphs)
    parts = []
    for (t, item), g in u.groupby(["ticker", "item"], sort=True):
        parts.append(g.sample(n=min(per_cell, len(g)), random_state=seed))
    return pd.concat(parts).sort_values(["ticker", "item", "fiscal_year", "para_idx"]).reset_index(drop=True)


def split_labels(s: str) -> set[str]:
    s = str(s).strip()
    return set() if s in ("", "None") else {x.strip() for x in s.split(";") if x.strip()}


def label_counts(labels: pd.DataFrame, forces: list[str]) -> pd.DataFrame:
    rows = []
    for f in forces + ["None"]:
        if f == "None":
            mask = labels.llm_label.map(lambda s: not split_labels(s))
        else:
            mask = labels.llm_label.map(lambda s, f=f: f in split_labels(s))
        r = {"force": f, "total": int(mask.sum())}
        for item in sorted(labels["item"].unique()):
            r[f"Item {item}"] = int((mask & (labels["item"] == item)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def check_sheet(labels: pd.DataFrame, forces: list[str], n: int, per_force: int, seed: int) -> pd.DataFrame:
    """Up to `per_force` llm-positives per force (where available), then random fill to n."""
    chosen: list[str] = []
    for i, f in enumerate(forces):
        pos = labels[labels.llm_label.map(lambda s, f=f: f in split_labels(s))]
        already = pos.para_id.isin(chosen).sum()
        need = max(0, per_force - already)
        pool = pos[~pos.para_id.isin(chosen)]
        chosen += pool.sample(n=min(need, len(pool)), random_state=seed + i).para_id.tolist()
    rest = labels[~labels.para_id.isin(chosen)]
    chosen += rest.sample(n=max(0, n - len(chosen)), random_state=seed).para_id.tolist()
    sheet = labels[labels.para_id.isin(chosen[:n])][["para_id", "text"]]
    sheet = sheet.sample(frac=1, random_state=seed).reset_index(drop=True)  # no force grouping
    sheet["human_label"] = ""
    return sheet


def main(cmd: str) -> None:
    cfg = load_config()
    seed, forces, proc = cfg["random_seed"], cfg["forces"], _proc(cfg)
    if cmd == "sample":
        paragraphs = pd.read_csv(proc / "paragraphs.csv", keep_default_na=False, dtype={"item": str})
        s = stratified_sample(paragraphs, PER_CELL, seed)
        s.to_csv(proc / "label_sample.csv", index=False)
        print(s.groupby(["ticker", "item"]).size().unstack())
        print(f"Wrote data/processed/label_sample.csv ({len(s)} rows)")
    elif cmd == "merge":
        sample = pd.read_csv(proc / "label_sample.csv", keep_default_na=False, dtype={"item": str})
        batches = sorted((proc / "llm_label_batches").glob("batch_*.csv"))
        lab = pd.concat([pd.read_csv(b, keep_default_na=False) for b in batches])
        assert not lab.para_id.duplicated().any(), "para_id labelled twice"
        bad = {x for s in lab.llm_label for x in split_labels(s)} - set(forces)
        assert not bad, f"unknown labels: {bad}"
        m = sample.merge(lab[["para_id", "llm_label"]], on="para_id", how="left")
        missing = m.llm_label.isna().sum()
        assert missing == 0, f"{missing} sampled paragraphs have no llm_label"
        m[["para_id", "ticker", "text", "llm_label"]].to_csv(proc / "llm_labels.csv", index=False)
        print(label_counts(m, forces).to_string(index=False))
        print(f"Wrote data/processed/llm_labels.csv ({len(m)} rows)")
    elif cmd == "checksheet":
        lab = pd.read_csv(proc / "llm_labels.csv", keep_default_na=False)
        sheet = check_sheet(lab, forces, CHECK_N, PER_FORCE_IN_CHECK, seed)
        sheet.to_csv(proc / "human_check_sheet.csv", index=False)
        print(f"Wrote data/processed/human_check_sheet.csv ({len(sheet)} rows)")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
