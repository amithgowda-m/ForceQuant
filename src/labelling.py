"""Stage 5: labelling samples and human check sheet (offline).

The llm_label values themselves are written by Claude Code in-session (batches of ~25,
following LABEL_GUIDE.md) into data/processed/llm_label_batches/batch_XX.csv. No external
API is called. Corrections after a review are kept separately in llm_relabels.csv
(old -> new + reason) so the first-pass labels stay auditable.

Two sample types (column sample_type in llm_labels.csv):
  random   - 20 unique paragraphs per company x item cell (label_sample.csv)
  enriched - Item 1A paragraphs selected by per-force keyword rules in config.yaml
             (enrich_sample.csv); different base rate, see README.

Usage:
  python -m src.labelling sample      # -> data/processed/label_sample.csv
  python -m src.labelling enrich      # -> data/processed/enrich_sample.csv
  python -m src.labelling merge       # batches + relabels -> data/processed/llm_labels.csv + counts
  python -m src.labelling checksheet  # -> data/processed/human_check_sheet.csv
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from src.utils import load_config, root_path, write_issues

STAGE = "stage5"
PER_CELL = 20
CHECK_N = 40
PER_FORCE_IN_CHECK = 6


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


def enriched_sample(paragraphs: pd.DataFrame, exclude_hashes: set, enrich_cfg: dict,
                    forces: list[str], seed: int) -> pd.DataFrame:
    """Per force: up to per_ticker keyword-matching Item 1A paragraphs per company, topped up to per_force."""
    u = unique_paragraphs(paragraphs)
    pool = u[(u["item"] == str(enrich_cfg["item"])) & ~u.para_hash.isin(exclude_hashes)].copy()
    chosen, parts = set(), []
    for i, force in enumerate(forces):
        rx = re.compile(enrich_cfg["keywords"][force], re.I)
        hits = pool[pool.text.map(lambda s: bool(rx.search(s))) & ~pool.para_hash.isin(chosen)]
        picked = []
        for t, g in hits.groupby("ticker", sort=True):
            picked.append(g.sample(n=min(enrich_cfg["per_ticker"], len(g)), random_state=seed + i))
        picked = pd.concat(picked) if picked else hits.iloc[0:0]
        short = enrich_cfg["per_force"] - len(picked)
        if short > 0:
            rest = hits[~hits.para_hash.isin(picked.para_hash)]
            picked = pd.concat([picked, rest.sample(n=min(short, len(rest)), random_state=seed + i)])
        picked = picked.assign(enrich_force=force,
                               enrich_terms=picked.text.map(lambda s: "; ".join(sorted({m.group(0).lower() for m in rx.finditer(s)}))))
        chosen |= set(picked.para_hash)
        parts.append(picked)
    return pd.concat(parts).reset_index(drop=True)


def split_labels(s: str) -> set[str]:
    s = str(s).strip()
    return set() if s in ("", "None") else {x.strip() for x in s.split(";") if x.strip()}


def has_force(series: pd.Series, force: str) -> pd.Series:
    return series.map(lambda s: force in split_labels(s))


def label_counts(labels: pd.DataFrame, forces: list[str], by: str) -> pd.DataFrame:
    rows = []
    for f in forces + ["None"]:
        mask = labels.llm_label.map(lambda s: not split_labels(s)) if f == "None" else has_force(labels.llm_label, f)
        r = {"force": f, "total": int(mask.sum())}
        for v in sorted(labels[by].unique()):
            r[f"{by}={v}"] = int((mask & (labels[by] == v)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def check_sheet(labels: pd.DataFrame, forces: list[str], n: int, per_force: int, seed: int) -> pd.DataFrame:
    """Up to `per_force` llm-positives per force (where available), then random fill to n.

    Draws from all rows (both sample types); output has no llm_label / sample_type columns.
    """
    chosen: list[str] = []
    for i, f in enumerate(forces):
        pos = labels[has_force(labels.llm_label, f)]
        need = max(0, per_force - pos.para_id.isin(chosen).sum())
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
    seed, forces = cfg["random_seed"], cfg["forces"]
    proc = root_path(cfg["paths"]["processed"])
    read = lambda name: pd.read_csv(proc / name, keep_default_na=False, dtype={"item": str})
    if cmd == "sample":
        s = stratified_sample(read("paragraphs.csv"), PER_CELL, seed)
        s.to_csv(proc / "label_sample.csv", index=False)
        print(s.groupby(["ticker", "item"]).size().unstack())
        print(f"Wrote data/processed/label_sample.csv ({len(s)} rows)")
    elif cmd == "enrich":
        rnd = read("label_sample.csv")
        e = enriched_sample(read("paragraphs.csv"), set(rnd.para_hash), cfg["labelling"]["enrichment"], forces, seed)
        e.to_csv(proc / "enrich_sample.csv", index=False)
        print(e.groupby(["enrich_force", "ticker"]).size().unstack(fill_value=0))
        print(f"Wrote data/processed/enrich_sample.csv ({len(e)} rows)")
    elif cmd == "merge":
        samples = [read("label_sample.csv").assign(sample_type="random")]
        if (proc / "enrich_sample.csv").exists():
            samples.append(read("enrich_sample.csv").assign(sample_type="enriched"))
        sample = pd.concat(samples, ignore_index=True)
        assert not sample.para_hash.duplicated().any(), "same text in both samples"
        batches = sorted((proc / "llm_label_batches").glob("batch_*.csv"))
        lab = pd.concat([pd.read_csv(b, keep_default_na=False) for b in batches])
        assert not lab.para_id.duplicated().any(), "para_id labelled twice in batches"
        issues = []
        rel_path = proc / "llm_relabels.csv"
        if rel_path.exists():
            rel = pd.read_csv(rel_path, keep_default_na=False)
            cur = lab.set_index("para_id").llm_label
            for r in rel.itertuples():
                assert cur[r.para_id] == r.old_llm_label, f"relabel old value mismatch for {r.para_id}"
                lab.loc[lab.para_id == r.para_id, "llm_label"] = r.new_llm_label
                issues.append(f"[relabel] {r.para_id}: {r.old_llm_label} -> {r.new_llm_label} ({r.reason})")
        bad = {x for s in lab.llm_label for x in split_labels(s)} - set(forces)
        assert not bad, f"unknown labels: {bad}"
        m = sample.merge(lab[["para_id", "llm_label"]], on="para_id", how="left")
        missing = int(m.llm_label.isna().sum())
        assert missing == 0, f"{missing} sampled paragraphs have no llm_label"
        m[["para_id", "ticker", "text", "llm_label", "sample_type"]].to_csv(proc / "llm_labels.csv", index=False)
        write_issues(cfg, STAGE, issues)
        print("By item:\n" + label_counts(m, forces, "item").to_string(index=False))
        print("\nBy sample_type:\n" + label_counts(m, forces, "sample_type").to_string(index=False))
        print(f"\nWrote data/processed/llm_labels.csv ({len(m)} rows; {len(issues)} relabel(s) logged)")
    elif cmd == "checksheet":
        lab = read("llm_labels.csv")
        sheet = check_sheet(lab, forces, CHECK_N, PER_FORCE_IN_CHECK, seed)
        sheet.to_csv(proc / "human_check_sheet.csv", index=False)
        print(f"Wrote data/processed/human_check_sheet.csv ({len(sheet)} rows)")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
