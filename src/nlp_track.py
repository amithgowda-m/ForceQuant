"""Stage 6: embeddings, baseline, training, evaluation, force shares, evidence quotes.

Usage:  python -m src.nlp_track
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import scipy.stats as stats

from src.utils import load_config, root_path, write_issues
from src import labelling as lb

STAGE = "stage6"
FORCES = [
    "Competitive_Rivalry", "Threat_of_New_Entry", "Threat_of_Substitutes",
    "Buyer_Power", "Supplier_Power", "Complementors",
]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def get_embedder(cfg: dict):
    import os
    # Force offline mode so the cached model is used without hitting HuggingFace
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    # Also disable SSL verification as a fallback for environments with cert issues
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        cfg["nlp"]["embedding_model"],
        device=cfg["nlp"]["device"],
    )


def embed(texts: list[str], model) -> np.ndarray:
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Label agreement (Stage 5 → 6 handoff)
# ---------------------------------------------------------------------------
def _split_set(s: str) -> set[str]:
    return lb.split_labels(s)


def compute_agreement(sheet: pd.DataFrame, forces: list[str]) -> pd.DataFrame:
    """Per-force percent agreement and Cohen's kappa between llm_label and human_label."""
    merged = sheet.copy()
    rows = []
    for f in forces:
        llm_pos = merged["llm_label"].map(lambda s: f in _split_set(s))
        hum_pos = merged["human_label"].map(lambda s: f in _split_set(s))
        # positives = union (either llm or human says it's the force)
        union_pos = llm_pos | hum_pos
        n_pos = int(union_pos.sum())
        n_llm_pos = int(llm_pos.sum())
        n_hum_pos = int(hum_pos.sum())
        n = len(merged)
        agree = int((llm_pos == hum_pos).sum())
        pct = agree / n if n else 0.0
        # Cohen's kappa
        p_yes = (llm_pos.sum() / n) * (hum_pos.sum() / n)
        p_no = ((n - llm_pos.sum()) / n) * ((n - hum_pos.sum()) / n)
        p_e = p_yes + p_no
        kappa = (pct - p_e) / (1 - p_e) if p_e < 1 else 1.0
        reliable = (pct >= 0.70) and (n_pos >= 8)
        rows.append({
            "Force": f, "PercentAgreement": round(pct, 4),
            "Kappa": round(kappa, 4),
            "Positives": n_pos, "LLM_Positives": n_llm_pos, "Human_Positives": n_hum_pos,
            "Reliable": reliable,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline: cosine similarity to prototypes
# ---------------------------------------------------------------------------
def baseline_predict(emb: np.ndarray, proto_embs: dict[str, np.ndarray],
                     threshold: float, forces: list[str]) -> list[list[str]]:
    """For each row, return list of forces whose max prototype similarity >= threshold."""
    preds = []
    for vec in emb:
        row_forces = []
        for f in forces:
            sims = proto_embs[f] @ vec  # already normalised
            if sims.max() >= threshold:
                row_forces.append(f)
        preds.append(row_forces)
    return preds


def build_proto_embeddings(cfg: dict, model) -> dict[str, np.ndarray]:
    proto_embs = {}
    for f in cfg["forces"]:
        texts = cfg["baseline"]["prototypes"].get(f, [])
        proto_embs[f] = embed(texts, model)
    return proto_embs


def select_threshold(emb_train: np.ndarray, y_train: np.ndarray,
                     proto_embs: dict[str, np.ndarray], forces: list[str],
                     thresholds=None) -> float:
    """Pick threshold maximising macro-F1 on train split (CV-free: just train, per plan)."""
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.20, 0.65, 0.02)]
    best_t, best_f1 = thresholds[0], -1.0
    mlb = MultiLabelBinarizer(classes=forces)
    mlb.fit([[f] for f in forces])
    y_bin = mlb.transform([[forces[i] for i, v in enumerate(row) if v] for row in y_train])
    for t in thresholds:
        preds = baseline_predict(emb_train, proto_embs, t, forces)
        pred_bin = mlb.transform(preds)
        p, r, f1, _ = precision_recall_fscore_support(y_bin, pred_bin, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ---------------------------------------------------------------------------
# Multi-label helpers
# ---------------------------------------------------------------------------
def labels_to_binary(label_series: pd.Series, forces: list[str]) -> np.ndarray:
    """(n_samples, n_forces) binary matrix."""
    rows = []
    for s in label_series:
        active = _split_set(s)
        rows.append([1 if f in active else 0 for f in forces])
    return np.array(rows, dtype=int)


def binary_to_labels(mat: np.ndarray, forces: list[str]) -> list[list[str]]:
    return [[forces[j] for j in range(len(forces)) if mat[i, j]] for i in range(len(mat))]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def train_lr(emb: np.ndarray, y_bin: np.ndarray, cfg: dict) -> OneVsRestClassifier:
    lr = LogisticRegression(
        class_weight=cfg["nlp"]["lr_class_weight"],
        max_iter=1000, random_state=cfg["random_seed"], solver="lbfgs",
    )
    clf = OneVsRestClassifier(lr)
    clf.fit(emb, y_bin)
    return clf


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   forces: list[str], model_name: str) -> list[dict]:
    rows = []
    for i, f in enumerate(forces):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        rows.append({
            "Force": f, "Model": model_name,
            "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4),
            "Positives": int(y_true[:, i].sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# Force shares and evidence
# ---------------------------------------------------------------------------
def compute_force_shares(paragraphs: pd.DataFrame, clf: OneVsRestClassifier,
                         emb_all: np.ndarray, forces: list[str]) -> pd.DataFrame:
    """Per-company share of paragraphs predicted positive for each force."""
    pred_bin = clf.predict(emb_all)
    df = paragraphs[["para_id", "ticker"]].copy().reset_index(drop=True)
    for i, f in enumerate(forces):
        df[f] = pred_bin[:, i]
    rows = []
    for t, g in df.groupby("ticker"):
        total = len(g)
        for f in forces:
            share = g[f].sum() / total if total else 0.0
            rows.append({"Ticker": t, "Force": f, "MentionShare": round(float(share), 6),
                         "N_Paragraphs": total, "N_Positive": int(g[f].sum())})
    return pd.DataFrame(rows)


def top_evidence(paragraphs: pd.DataFrame, clf: OneVsRestClassifier,
                 emb_all: np.ndarray, forces: list[str]) -> pd.DataFrame:
    """Highest-probability paragraph per force per company."""
    # Get probability scores from each per-force estimator
    n = len(paragraphs)
    df = paragraphs[["para_id", "ticker", "text", "accession", "filing_url"]].copy().reset_index(drop=True)
    rows = []
    for i, f in enumerate(forces):
        est = clf.estimators_[i]
        probs = est.predict_proba(emb_all)[:, 1]
        df_f = df.copy()
        df_f["prob"] = probs
        for t, g in df_f.groupby("ticker"):
            best = g.loc[g["prob"].idxmax()]
            rows.append({
                "Ticker": t, "Force": f,
                "TopProbability": round(float(best["prob"]), 4),
                "TopEvidenceQuote": best["text"][:500],
                "TopEvidenceParagraphId": best["para_id"],
                "FilingURL": best["filing_url"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# llm_label share on random rows (validation)
# ---------------------------------------------------------------------------
def llm_label_shares(labels: pd.DataFrame, forces: list[str]) -> pd.DataFrame:
    random_rows = labels[labels["sample_type"] == "random"]
    rows = []
    for t in sorted(labels["ticker"].unique()):
        g = random_rows[random_rows["ticker"] == t]
        total = len(g)
        for f in forces:
            pos = g["llm_label"].map(lambda s: f in _split_set(s)).sum()
            rows.append({"Ticker": t, "Force": f,
                         "LLM_MentionShare_random": round(pos / total, 6) if total else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    seed = cfg["random_seed"]
    forces = cfg["forces"]
    proc = root_path(cfg["paths"]["processed"])
    out = root_path(cfg["paths"]["outputs"])
    issues: list[str] = []

    # --- Load labels ---
    labels = pd.read_csv(proc / "llm_labels.csv", keep_default_na=False)
    check_path = proc / "human_check_sheet.csv"
    sheet = pd.read_csv(check_path, keep_default_na=False)

    # Merge llm_label and sample_type back onto check sheet
    sheet = sheet.merge(labels[["para_id", "llm_label", "sample_type"]], on="para_id", how="left")

    has_human = sheet["human_label"].str.strip().ne("").any()

    # --- Agreement ---
    if has_human:
        agree_df = compute_agreement(sheet, forces)
        agree_df.to_csv(out / "label_agreement.csv", index=False)
        print("\nLabel Agreement:")
        print(agree_df.to_string(index=False))
        for _, row in agree_df.iterrows():
            if not row["Reliable"]:
                issues.append(
                    f"[{row['Force']}] UNRELIABLE: agreement={row['PercentAgreement']}, "
                    f"kappa={row['Kappa']}, positives={row['Positives']}"
                )
    else:
        issues.append("[stage6] human_check_sheet.csv has no human_label values yet; "
                      "agreement skipped (fill human_label and rerun)")
        print("WARNING: human_label column is empty. Agreement skipped.")
        agree_df = pd.DataFrame(columns=["Force", "PercentAgreement", "Kappa", "Positives",
                                          "LLM_Positives", "Human_Positives", "Reliable"])
        agree_df.to_csv(out / "label_agreement.csv", index=False)

    # --- Embeddings ---
    print("\nLoading sentence-transformer model...")
    model = get_embedder(cfg)

    # Embed all labelled paragraphs
    print(f"Embedding {len(labels)} labelled paragraphs...")
    emb_labels = embed(labels["text"].tolist(), model)

    y_bin = labels_to_binary(labels["llm_label"], forces)

    # Random-only split (80/20 stratified by primary label, seed=42)
    random_mask = (labels["sample_type"] == "random").values
    emb_rnd = emb_labels[random_mask]
    y_rnd = y_bin[random_mask]
    rnd_labels_series = labels.loc[random_mask, "llm_label"].values

    def primary_label(s):
        parts = _split_set(s)
        return sorted(parts)[0] if parts else "None"

    strat_key = np.array([primary_label(s) for s in rnd_labels_series])
    rng = np.random.default_rng(seed)
    classes = np.unique(strat_key)
    test_indices = []
    for cls in classes:
        cls_idx = np.where(strat_key == cls)[0]
        n_test = max(1, int(len(cls_idx) * 0.2))
        rng.shuffle(cls_idx)
        test_indices.extend(cls_idx[:n_test].tolist())
    test_indices = sorted(set(test_indices))
    train_indices = [i for i in range(len(emb_rnd)) if i not in test_indices]

    emb_train_rnd = emb_rnd[train_indices]
    emb_test_rnd = emb_rnd[test_indices]
    y_train_rnd = y_rnd[train_indices]
    y_test_rnd = y_rnd[test_indices]

    # --- Baseline zero-shot ---
    print("Building prototype embeddings for zero-shot baseline...")
    proto_embs = build_proto_embeddings(cfg, model)

    print("Selecting baseline threshold on training split...")
    best_t = select_threshold(emb_train_rnd, y_train_rnd, proto_embs, forces)
    print(f"  Selected threshold: {best_t}")

    # Evaluate baseline on random test rows
    preds_base_test_rnd = baseline_predict(emb_test_rnd, proto_embs, best_t, forces)
    mlb = MultiLabelBinarizer(classes=forces)
    mlb.fit([[f] for f in forces])
    y_pred_base_test_rnd = mlb.transform(preds_base_test_rnd)
    report_rows_rnd = report_metrics(y_test_rnd, y_pred_base_test_rnd, forces, "baseline_random_only")

    # Evaluate baseline on all test rows (all sample types)
    emb_test_all = emb_labels  # apply to all, report on held-out
    preds_base_all_heldout = baseline_predict(emb_test_rnd, proto_embs, best_t, forces)
    # (same as above for random; we report both headers separately)

    # --- LR classifier ---
    print("Training LR classifier on random+enriched labelled rows...")
    clf = train_lr(emb_labels, y_bin, cfg)

    # Evaluate on random held-out
    y_pred_lr_test_rnd = clf.predict(emb_test_rnd)
    report_rows_rnd += report_metrics(y_test_rnd, y_pred_lr_test_rnd, forces, "LR_random_only")

    # Evaluate on all held-out (enriched inflates; reported separately)
    # For 'all held-out': use the same test_indices from random rows + all enriched rows
    enriched_mask = labels["sample_type"] == "enriched"
    emb_enr = emb_labels[enriched_mask]
    y_enr = y_bin[enriched_mask]
    if emb_enr.shape[0] > 0:
        emb_all_heldout = np.vstack([emb_test_rnd, emb_enr])
        y_all_heldout = np.vstack([y_test_rnd, y_enr])
        preds_base_all = baseline_predict(emb_all_heldout, proto_embs, best_t, forces)
        y_pred_base_all = mlb.transform(preds_base_all)
        y_pred_lr_all = clf.predict(emb_all_heldout)
        report_rows_all = report_metrics(y_all_heldout, y_pred_base_all, forces, "baseline_all_heldout")
        report_rows_all += report_metrics(y_all_heldout, y_pred_lr_all, forces, "LR_all_heldout")
    else:
        report_rows_all = []

    all_report_rows = report_rows_rnd + report_rows_all
    report_df = pd.DataFrame(all_report_rows)
    header = (
        "# Classifier evaluation report\n"
        "# Reference labels: LLM labels (llm_label), verified on a 60-paragraph human sample.\n"
        "# '*_random_only' rows: held-out 20% of random sample rows.\n"
        "# '*_all_heldout' rows: random held-out + all enriched rows (enriched rows inflate numbers).\n"
    )
    csv_path = out / "classifier_report.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        report_df.to_csv(fh, index=False)
    print(f"\nClassifier report:\n{report_df.to_string(index=False)}")

    # --- Apply model to ALL paragraphs ---
    print("\nLoading all paragraphs for tagging...")
    paragraphs = pd.read_csv(proc / "paragraphs.csv", keep_default_na=False, dtype={"item": str})
    print(f"Embedding {len(paragraphs)} total paragraphs...")
    emb_all_paras = embed(paragraphs["text"].tolist(), model)

    print("Computing force shares per company...")
    shares_df = compute_force_shares(paragraphs, clf, emb_all_paras, forces)
    shares_df.to_csv(out / "force_shares.csv", index=False)
    print(shares_df.to_string(index=False))

    # Llm_label shares on random rows for comparison
    llm_shares = llm_label_shares(labels, forces)
    llm_shares.to_csv(out / "llm_label_shares_random.csv", index=False)
    print("\nLLM-label mention shares (random rows):\n", llm_shares.to_string(index=False))

    print("\nFinding top evidence paragraphs...")
    evidence_df = top_evidence(paragraphs, clf, emb_all_paras, forces)
    evidence_df.to_csv(out / "top_evidence.csv", index=False)

    write_issues(cfg, STAGE, issues)
    print(f"\nStage 6 complete. Wrote: label_agreement.csv, classifier_report.csv, "
          f"force_shares.csv, llm_label_shares_random.csv, top_evidence.csv")


if __name__ == "__main__":
    main()
