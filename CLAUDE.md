# Project: Porter's Five Forces for software companies (dual-track pipeline)

Book chapter project (RVCE). This repo produces DATA, TABLES and FIGURES only.
Do NOT write chapter prose, abstracts, discussions or conclusions (AI-content limit is 10%).
Short README and code comments are fine.

## Hard rules
1. Never invent data. If a value is missing, write `Not Disclosed` (text KPIs) or `n/a` (numeric). Never estimate silently.
2. Network calls happen ONLY in `src/fetch_*.py`. Everything is cached in `data/raw/`. All other code runs offline.
3. SEC requires a User-Agent with a real contact. Read it from the `SEC_USER_AGENT` environment variable or the git-ignored `.env` file (template: `.env.example`). The `config.yaml` `sec_user_agent` value is only a placeholder fallback that makes fetch scripts refuse to run. Never print the user-agent value in logs, issues.log or outputs. Stay under 10 requests/second (sleep 0.2s).
4. Fixed random seed 42 everywhere. Results must be reproducible.
5. Every number in the outputs must be traceable: keep source tag / filing URL / accession number columns.
6. After each stage: run `pytest`, commit with a clear message, and print a short summary of what changed.
7. If something is ambiguous or fails (e.g. Item 7 not found), log it in `outputs/issues.log` and continue. Do not fake a result.
8. Read JSON caches with `json.load`, never `pd.read_json`.
9. Be honest about labels: LLM labels are called `llm_label`, human checks are called `human_label`. Never call LLM labels "manual" or "human".

## Companies and cases
ZM, MSFT (Case 1: collaboration suites), MDB, ESTC (Case 2: database licensing), NVDA (Case 3: AI compute supplier).
Look up CIKs from `https://www.sec.gov/files/company_tickers.json`, do not hard-code them.
Fiscal years differ (ZM, MDB, NVDA end Jan; MSFT Jun; ESTC Apr). Use the fiscal-year label from the filing, and say so in outputs.
Case 4 (CloudMetrics) is hypothetical and computed from equations only.

## Layout
```
config.yaml            # user agent, tickers, anchors for scoring, prototypes, paths
LABEL_GUIDE.md         # written definitions of the 6 forces (used for labelling)
src/models.py          # equations 1-13 (pure functions, no I/O)
src/fetch_financials.py
src/fetch_filings.py   # 10-K download + Item 1A / Item 7 extraction + paragraphs
src/kpi_extract.py     # regex KPIs from Item 7
src/nlp_track.py       # embeddings, baseline, training, evaluation, tagging
src/scoring.py         # S_i, MAS
src/figures.py
tests/
data/raw/  data/processed/  outputs/  outputs/figures/
```

## Financial extraction (companyfacts JSON)
- Use a fallback tag list per metric (e.g. Revenue: Revenues, RevenueFromContractWithCustomerExcludingAssessedTax, SalesRevenueNet).
- COGS: CostOfRevenue, CostOfGoodsAndServicesSold. R&D: ResearchAndDevelopmentExpense. S&M: SellingAndMarketingExpense (NVDA reports only SG&A, so S&M = n/a). OpInc: OperatingIncomeLoss.
- Use only annual (form 10-K, fp FY) values, one per fiscal year, latest filed wins. Record the tag used in `source_notes`.

## Equations (src/models.py) with exact test targets
1. FC = FC_ops + A_RD + D_hw, A_RD = I_RD / N_life
2. TC = FC + VC*Q, TR = P*Q
3. Q_BE = FC / (P - VC)
4. DOL = Q(P-VC) / (Q(P-VC) - FC). Empirical version: (Revenue - COGS) / OperatingIncome; return `n/m` if OperatingIncome <= 0.
5. NRR = 1 - c + g
6. LTV = (P-VC)/(r+c-g), only valid if r+c>g (raise ValueError otherwise). Filings proxy: LTV = ARPU*GM / c, only if churn c is disclosed.
7. LTV/CAC = LTV / CAC. CAC = S&M / new customers, only if both are disclosed, else `n/a`.
8. SC = C_mig + N_emp*H_train*W_emp + C_int + rho*C_down
9. NPV = sum_{t=1..T} ((P_inc - P_new) + dE)/(1+r)^t - SC
10. ROI = NPV / SC * 100
11. E_d = |%dQ / %dP|
12. L = (P-MC)/P = 1/E_d. Filings proxy: (Revenue - COGS)/Revenue.
13. MAS = 100 * (1 - sum_i w_i * (S_i - 1)/9), i = 1..6, sum w_i = 1, S_i in [1,10]

CloudMetrics must reproduce exactly (tests/test_cloudmetrics.py):
- P=40000, VC=8000, R&D 6.0M over 5 yrs + 1.2M ops => FC = 2,400,000, Q_BE = 75
- SC = 25000 + 50*20*35 + 15000 + 0.05*300000 = 90,000
- annual benefit = (55000-40000)+10000 = 25,000; r=10%, T=4; PVAF = 3.169865
- NPV = -10,753.4 (round to -10,753), ROI = -11.95%, break-even price cut = 3,392.3 per year

## Text track design
- Sections: Item 1A (Risk Factors) and Item 7 (MD&A) of each company's latest 10-K, plus the previous two years if available. Item splitting is brittle (MSFT layout is unusual): log failures, never guess.
- Split into paragraphs (min 25 words). Keep metadata: ticker, fiscal year, item, paragraph id, accession number.
- Embeddings: sentence-transformers `all-MiniLM-L6-v2`, normalized, CPU.
- Forces (6): Competitive_Rivalry, Threat_of_New_Entry, Threat_of_Substitutes, Buyer_Power, Supplier_Power, Complementors. Definitions are in LABEL_GUIDE.md. A paragraph can have several forces or none.

### Labelling (LLM-assisted, human-verified)
- Claude Code labels a stratified sample of ~200 paragraphs (spread evenly across companies and items) following LABEL_GUIDE.md. Do this yourself in-session by reading the paragraphs in batches of ~25 and writing the labels to `data/processed/llm_labels.csv` (columns: para_id, ticker, text, llm_label). Do NOT call any external API for labelling.
- Then export `data/processed/human_check_sheet.csv`: 60 paragraphs = 6 distinct llm-positive paragraphs per force (fewer if a force has fewer positives; both sample types) + 24 llm-None paragraphs from the random sample (to test for missed positives), shuffled. Columns `para_id`, `text`, empty `human_label` only: NO llm_label or sample_type column (so the human is not biased). Then STOP and wait for the human.
- After the human returns it, compute per-force agreement between `human_label` and `llm_label` (percent agreement and Cohen's kappa) into `outputs/label_agreement.csv`. Flag any force with agreement under 70% or fewer than 8 positives as unreliable. For this rule, Positives = check-sheet paragraphs where llm_label OR human_label includes the force; also report LLM-positive and human-positive counts separately.

### Tagger and evaluation
- Baseline (zero-shot): cosine similarity to hand-written prototype sentences in config.yaml, with a threshold.
- Main model: One-vs-Rest LogisticRegression (class_weight balanced) on MiniLM embeddings, trained on `llm_label` paragraphs with a stratified 80/20 split (seed 42).
- Report precision/recall/F1 per force on the held-out 20% against `llm_label`, for both baseline and main model, in `outputs/classifier_report.csv`. State in the file header that the reference labels are LLM labels, verified on a 60-paragraph human sample.
- Apply the main model to ALL paragraphs. Report force share per company, and keep the top evidence paragraph (highest probability) per force per company.
- KPI regex (Item 7): match variants such as "net dollar retention", "net retention", "net dollar expansion rate", "net expansion rate", "ARR", "customers". Not found => `Not Disclosed`. Keep the matched sentence as evidence.

## Scoring (src/scoring.py), all anchors live in config.yaml
S_i (threat of force i, 1 = weak threat, 10 = strong threat) = average of the parts that exist:
- NLP salience part: share of a company's paragraphs tagged with force i, rescaled to 1-10 using fixed anchors (not min-max over 5 firms).
- Financial part, only where a metric exists:
  - Buyer_Power: gross margin (high margin = low threat)
  - Threat_of_New_Entry: DOL (high operating leverage = scale barrier = low threat), n/m => skip
  - Competitive_Rivalry and Threat_of_Substitutes: 3-year trend of Lerner proxy (falling = high threat)
  - Supplier_Power and Complementors: no financial metric, NLP part only
- Weights: equal 1/6. Record which parts were used per cell in the CSV (`parts_used`).
- Default anchors are assumptions. Print them in outputs so the team can state them in the chapter.

## Outputs (data contract)
- `outputs/financial_metrics.csv`: Ticker, FiscalYear, Revenue, COGS, R&D, S&M, G&A, OpInc, GrossMargin, Lerner_proxy, DOL, LTV_CAC, NDR, source_notes
- `outputs/forces_summary.csv`: Ticker, Force, MentionShare, FinancialScore, S_i, parts_used, MAS (per company), TopEvidenceQuote, FilingURL
- `outputs/classifier_report.csv`: Force, Model (baseline/LR), Precision, Recall, F1, Positives
- `outputs/label_agreement.csv`: Force, PercentAgreement, Kappa, Positives, Reliable
- `outputs/figures/` (300 DPI PNG): fig1_mas_radar, fig2_lerner_trends, fig3_cloudmetrics_breakeven, fig4_cloudmetrics_npv
- `outputs/issues.log`