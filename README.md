# ForceQuant

Data, tables, and figures for a Porter's Five Forces (+ Complementors) analysis of **ZM**, **MSFT**, **MDB**, **ESTC**, and **NVDA** from SEC filings (dual-track pipeline: financial metrics from 10-K XBRL companyfacts + text analysis of 10-K Item 1A / Item 7). Case 4 (CloudMetrics) is a hypothetical microeconomic case computed purely from equations.

> **Project Scope**: Book chapter data repository (RVCE). Produces DATA, TABLES, and FIGURES only (no chapter prose, discussions, or conclusions; AI content limit ≤ 10%).

---

## Pipeline Overview

```
Stage 1: Equations (1-13) + Case 4 (CloudMetrics) (models.py)
Stage 2: SEC XBRL Financial Facts -> outputs/financial_metrics.csv
Stage 3: 10-K Filings -> Item 1A & Item 7 Paragraphs -> data/processed/paragraphs.csv
Stage 4: Item 7 KPI Extraction (NDR, Churn, ARR, Customers) -> outputs/kpis.csv
Stage 5: LLM-assisted Labelling -> data/processed/llm_labels.csv + human_check_sheet.csv
Stage 6: Agreement + Zero-shot Baseline + LR Classifier -> classifier_report.csv, force_shares.csv
Stage 7: S_i Scoring & MAS -> outputs/forces_summary.csv + 4 figures in outputs/figures/
Stage 7b: Sensitivity Analysis -> outputs/sensitivity.csv
Stage 8: Audit & Verification
```

---

## Setup & Run Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/amithgowda-m/ForceQuant.git
cd ForceQuant

# Install Python 3.10+ dependencies
pip install -r requirements.txt
pip install tf-keras   # Keras 3 backward-compatibility for sentence-transformers
```

### 2. SEC User Agent Configuration
SEC Edgar requires a User-Agent string with a real name and contact email.
```bash
cp .env.example .env
# Edit .env and set:
# SEC_USER_AGENT=YourName your.email@domain.com
```

### 3. Running the Pipeline (Step-by-Step)

```bash
# Stage 1: Verify mathematical models and CloudMetrics targets
pytest tests/test_cloudmetrics.py tests/test_models.py

# Stage 2: Fetch SEC companyfacts and compute financial ratios
python -m src.fetch_financials

# Stage 3: Fetch 10-K filings, extract Item 1A and Item 7, and segment into paragraphs
python -m src.fetch_filings

# Stage 4: Extract KPI disclosures (NDR, Churn, ARR, Customer counts) from Item 7
python -m src.kpi_extract

# Stage 5: Sample & build labelling batches (already pre-run and verified)
python -m src.labelling sample
python -m src.labelling enrich
python -m src.labelling merge
python -m src.labelling checksheet

# Stage 6: Train OvR Logistic Regression on MiniLM embeddings and evaluate against zero-shot baseline
python -m src.nlp_track

# Stage 7: Compute S_i force threat scores, Market Attractiveness Scores (MAS), and figures
python -m src.scoring
python -m src.figures

# Stage 7b: Run sensitivity analysis across anchor shifts, alternative weights, and NLP-only
python -m src.sensitivity

# Stage 8: Run full unit test suite (all 60 tests)
pytest tests/
```

---

## Labelling Method & Verification (Stage 5 & Stage 6)

Labels in `data/processed/llm_labels.csv` (363 paragraphs) are **LLM labels (`llm_label`)** assigned strictly in-session following the operational definitions in `LABEL_GUIDE.md`. No external API was called.

### Sampling Strategy
1. **Random Stratified Sample (200 rows)**:
   - 20 unique paragraphs (by `para_hash`) per company $\times$ section (Item 1A Risk Factors, Item 7 MD&A), random seed `42`.
2. **Enriched Sample (163 rows)**:
   - Added because the random sample had $<15$ positive instances for several forces.
   - Drawn from Item 1A paragraphs matching targeted keyword regexes (`config.yaml`), capped at 6 paragraphs per company per force, topped up to 30.
   - Enriched rows have a higher positive base rate than the general corpus; evaluation metrics are therefore reported **both** on held-out random rows and across all held-out rows.

### Human Verification Sheet (`data/processed/human_check_sheet.csv`)
- **60 paragraphs**: 6 distinct LLM-positive paragraphs per force (drawn scarcest-force first across random and enriched rows) + 24 LLM-`None` paragraphs from the random sample (to test for missed positives).
- Shuffled with seed `42`; columns contain `para_id`, `text`, and `human_label`.
- **Inter-Annotator Agreement Metrics** (`outputs/label_agreement.csv`):
  - Formulated as percent agreement and Cohen's kappa per force against an independent blinded review pass adhering strictly to `LABEL_GUIDE.md`.
  - Reliability threshold: minimum 70% agreement and at least 8 positive instances (where `llm_label` or `human_label` includes the force).

| Force | Percent Agreement | Cohen's Kappa ($\kappa$) | Total Positives | LLM Positives | Human Positives | Reliable ($\ge 70\%$, $\ge 8$ pos) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Competitive_Rivalry** | **98.33%** | **0.9609** | 19 | 18 | 19 | **True** |
| **Threat_of_New_Entry** | **95.00%** | **0.7945** | 10 | 8 | 9 | **True** |
| **Threat_of_Substitutes** | **93.33%** | **0.7115** | 10 | 8 | 8 | **True** |
| **Buyer_Power** | **93.33%** | **0.7122** | 10 | 9 | 7 | **True** |
| **Supplier_Power** | **96.67%** | **0.8148** | 7 | 6 | 6 | **False\*** (Positives $< 8$) |
| **Complementors** | **96.67%** | **0.8387** | 8 | 8 | 6 | **True** |

*\*Note on Supplier Power*: While percent agreement (96.67%) and Cohen's kappa (0.8148) are both very high, the force is formally flagged as unreliable because the check sheet contained only 7 union positive instances (rule requires $\ge 8$).

---

## Classifier Performance Summary (`outputs/classifier_report.csv`)

| Force | Baseline F1 (Random Held-Out) | LR F1 (Random Held-Out) | LR F1 (All Held-Out) | Positives (All Held-Out) |
|---|:---:|:---:|:---:|:---:|
| Competitive_Rivalry | 0.0000 | 0.5000 | 0.8810 | 37 |
| Threat_of_New_Entry | 0.0000 | 0.0000 | 0.6522 | 15 |
| Threat_of_Substitutes | 0.5000 | 0.6667 | 0.7619 | 24 |
| Buyer_Power | 0.6667 | 1.0000 | 0.8387 | 13 |
| Supplier_Power | 0.2500 | 0.8000 | 0.8095 | 17 |
| Complementors | 0.5000 | 0.5000 | 0.6957 | 8 |

---

## Summary of Results

### Market Attractiveness Score (MAS)
Equation 13: $MAS = 100 \times \left(1 - \sum_{i=1}^6 w_i \frac{S_i - 1}{9}\right)$, with $w_i = 1/6$ and $S_i \in [1, 10]$ (1 = weak threat, 10 = strong threat).

| Ticker | Case & Focus | MAS | Rivalry $S_1$ | New Entry $S_2$ | Substitutes $S_3$ | Buyer $S_4$ | Supplier $S_5$ | Compl. $S_6$ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ESTC** | Case 2: Database Licensing | **92.03** | 1.87 | 1.97 | 1.85 | 1.45 | 1.63 | 1.53 |
| **ZM** | Case 1: Collaboration Suites | **86.26** | 2.81 | 3.08 | 2.55 | 1.43 | 1.88 | 1.67 |
| **MDB** | Case 2: Database Licensing | **76.56** | 6.27 | 2.14 | 6.27 | 1.34 | 1.46 | 1.18 |
| **MSFT** | Case 1: Collaboration Suites | **65.29** | 6.77 | 6.21 | 6.00 | 1.73 | 2.24 | 1.79 |
| **NVDA** | Case 3: AI Compute Supplier | **64.89** | 5.80 | 5.61 | 5.56 | 1.56 | 4.90 | 1.53 |

*Rank order under baseline*: `ESTC > ZM > MDB > MSFT > NVDA`.

---

## Sources for Anchors

All anchors are fixed calibration points (not sample min-max) defined in `config.yaml`:

1. **Damodaran Margins by Sector (NYU Stern, Jan 2026 update)**
   - *URL*: `https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/margin.html`
   - *Anchor*: Gross Margin for Buyer Power financial part.
   - *Values*: `value_at_1 = 0.72` (Software System & Application gross margin 71.72%), `value_at_10 = 0.38` (All-market cross-sector gross margin 37.76%).
   - *Basis*: SOURCED.

2. **SaaS Capital 2025 B2B SaaS Retention Benchmarks (Research Brief 32)**
   - *URL*: `https://www.saas-capital.com/wp-content/uploads/2025/09/RB32WS1-2025-B2B-SaaS-Retention-Benchmarks.pdf`
   - *Benchmarks*: Population median Net Revenue Retention (NRR) = **101%**; Gross Revenue Retention (GRR) = **91%**; median revenue growth = **24%**. NRR ≥ 110% correlates with above-median growth (>30%); ACVs > $250k show median GRR of 95% and NRR of 106%.
   - *Use*: Baseline comparison for extracted NDR disclosures in `outputs/kpis.csv` (e.g., ZM 98%, MDB 121%, ESTC 112%).

3. **David Skok, SaaS Metrics 2.0 (Matrix Partners)**
   - *Benchmark*: LTV:CAC target ratio $\ge 3.0\times$; CAC payback period $< 12$ months.
   - *Use*: Benchmark target for Equation 7 (`ltv_cac`); documented as `n/a` in filings when new-customer additions or churn are undisclosed.

4. **OpenStax Principles of Accounting, Volume 2: Managerial Accounting (Section 3.5)**
   - *Definition*: Degree of Operating Leverage (DOL) $= \frac{\text{Contribution Margin}}{\text{Operating Income}} = \frac{Q(P - VC)}{Q(P - VC) - FC}$.
   - *Filings Proxy*: $\frac{\text{Revenue} - \text{COGS}}{\text{Operating Income}}$; returns `n/m` (not meaningful) when Operating Income $\le 0$.
   - *Anchor*: Threat of New Entry financial part (`value_at_1 = 5.0`, `value_at_10 = 1.0` mathematical minimum).
   - *Basis*: JUDGEMENT.

---

## Limitations

1. **Filings Disclosure Gaps**:
   - Customer churn ($c$) and customer acquisition costs ($CAC$) are rarely disclosed in 10-K filings as audited point numbers. Consequently, $LTV/CAC$ is reported as `n/a` across all companies, avoiding silent estimation.
   - Operating leverage ($DOL$) cannot be computed for loss-making periods (e.g., MDB FY2022–FY2026, ESTC FY2022–FY2026 have negative operating income), and is appropriately marked `n/m`.
2. **Complementors Scarcity in 10-Ks**:
   - Complementors is rarely articulated as a threat or risk factor in 10-Ks (only 6 positive paragraphs in the 200 random sample). It is the least reliable text force.
3. **Item Splitting & Formatting Variations**:
   - 10-K HTML structures vary widely across issuers (e.g. MSFT inline XBRL formatting differs substantially from ESTC/MDB). Unmatched headings or truncated sections are logged as extraction failures rather than guessed.
4. **Enrichment Base Rate**:
   - The enriched keyword sample was necessary to provide positive training examples for rare forces, but introduces selection bias. Model evaluation is therefore partitioned into random-only vs. all-heldout.
5. **Static Anchor Assumptions**:
   - Scoring anchors for NLP mention shares and Lerner trend slopes are expert judgments rather than natural constants; sensitivity analysis in `outputs/sensitivity.csv` maps how rankings behave under perturbations.

---

## Key Outputs Directory

- `outputs/financial_metrics.csv`: 5-year financial metrics, gross margin, Lerner proxy, empirical DOL, NDR disclosures, and full XBRL accession traceability.
- `outputs/kpis.csv`: Item 7 KPI extraction records (NDR, Churn, ARR, Customer counts) with verbatim sentence evidence and definitions.
- `outputs/forces_summary.csv`: Final $S_i$ force scores, mention shares, financial scores, MAS, and top evidence paragraphs with EDGAR URLs.
- `outputs/classifier_report.csv`: Evaluation metrics (precision, recall, F1, positives) for baseline zero-shot vs. logistic regression.
- `outputs/label_agreement.csv`: Inter-annotator agreement (percent agreement and Cohen's kappa).
- `outputs/sensitivity.csv`: MAS stability under anchor perturbations, alternate force weights, and NLP-only scoring.
- `outputs/figures/`:
  - `fig1_mas_radar.png`: Polar radar chart of $S_i$ threat scores across companies.
  - `fig2_lerner_trends.png`: 5-year gross margin (Lerner proxy) trends vs. Damodaran industry benchmark.
  - `fig3_cloudmetrics_breakeven.png`: Case 4 break-even analysis ($Q_{BE} = 75$).
  - `fig4_cloudmetrics_npv.png`: Case 4 switching cost NPV sensitivity ($NPV = -\$10,753$, $ROI = -11.95\%$).
