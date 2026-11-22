# Prompts for Claude Code (paste one at a time, in order)

Before starting: put CLAUDE.md, LABEL_GUIDE.md and PROMPTS.md in an empty project folder, open Claude Code there.
Put a real email in `sec_user_agent` when it asks (SEC blocks anonymous requests).
Be honest in the chapter: labels are LLM-assisted and human-verified on a 60-paragraph sample.

## Stage 0: scaffold
Read CLAUDE.md and LABEL_GUIDE.md fully. Create the folder layout, config.yaml (sec_user_agent placeholder, tickers, scoring anchors as clearly marked assumptions, prototype sentences placeholder), requirements.txt, and a pytest setup. Do not fetch anything yet. Show me config.yaml and wait.

## Stage 1: equations + CloudMetrics (no internet needed)
Implement src/models.py (equations 1-13, with the n/m, n/a and ValueError rules in CLAUDE.md) and tests/test_cloudmetrics.py with the exact targets. Run pytest. Stage is done only when every target matches. Commit.

## Stage 2: financials
Implement src/fetch_financials.py: resolve CIKs from company_tickers.json, download and cache companyfacts JSON for ZM, MSFT, MDB, ESTC, NVDA, extract annual metrics with fallback tags, and write outputs/financial_metrics.csv including Lerner_proxy and DOL (n/m when OpInc <= 0). List every n/a with the reason in outputs/issues.log. Show me the CSV and the issues. Commit.

## Stage 3: filings text
Implement src/fetch_filings.py: get the latest 10-K (and up to two previous) per company via the submissions API, cache the HTML, extract Item 1A and Item 7, split into paragraphs (>= 25 words), save data/processed/paragraphs.csv. Report paragraph counts per company and any extraction failure. Do not guess when a section is missing. Commit.

## Stage 4: KPIs
Implement src/kpi_extract.py for NDR/net retention, ARR, customer counts, churn from Item 7. Output the matched sentence as evidence; otherwise Not Disclosed. Show me a table of what was found per company. Commit.

## Stage 5: LLM labelling, then stop for my check
Sample ~200 paragraphs from paragraphs.csv, stratified across companies and items (seed 42). Label them yourself, in this session, following LABEL_GUIDE.md exactly, in batches of ~25. Save data/processed/llm_labels.csv (para_id, ticker, text, llm_label). Show me the label counts per force. Then export data/processed/human_check_sheet.csv with 60 paragraphs (6 distinct llm-positives per force, fewer if short, plus 24 llm-None from the random sample; shuffled), columns para_id, text and an empty human_label only (no llm_label, no sample_type). STOP and tell me to fill human_label by hand using LABEL_GUIDE.md.

## Stage 6: agreement, train, evaluate
After I return the filled sheet: compute per-force agreement between human_label and llm_label (percent and Cohen's kappa) into outputs/label_agreement.csv and flag unreliable forces. Then run the zero-shot prototype baseline and the MiniLM + OvR logistic regression model (80/20 stratified split, seed 42) on llm_label, and write outputs/classifier_report.csv for both. Apply the main model to all paragraphs, save force shares per company and the top evidence paragraph per force per company. Commit.

Stage 6 additions (agreed after Stage 5): report precision/recall/F1 twice, on held-out random rows only and on all held-out rows (enriched rows inflate the all-rows number); group the split by para_hash; set the decision threshold on random-sample rows, not enriched ones; show predicted share next to llm_label share on the random rows before using model shares in Stage 7. Reliability rule: Positives = check-sheet paragraphs where llm_label OR human_label includes the force; report LLM-positive and human-positive counts separately as well.

## Stage 7: scoring and figures
Implement src/scoring.py (S_i and MAS exactly as in CLAUDE.md, with parts_used) and src/figures.py (4 figures at 300 DPI). Write outputs/forces_summary.csv with the top evidence quote and filing URL per force. Print the assumptions (anchors) used. Commit.

## Stage 7b: sensitivity analysis
Re-run MAS with each anchor moved +/-20%, with two alternative weight sets (one favouring Rivalry and Substitutes, one favouring Buyer and Supplier power), and with the financial parts removed. Save outputs/sensitivity.csv with company ranks for each run and print whether the ranking changes. Commit.

## Stage 8: audit
Re-check every number in the outputs against the raw cache, confirm no network calls outside src/fetch_*.py, run pytest, and write a README with run instructions, the labelling method (LLM-assisted, human-verified on 60 paragraphs, with the agreement numbers) and a limitations list. In the README, include a "Sources for anchors" list: Damodaran margins page (Jan 2026), SaaS Capital 2025 B2B SaaS Retention Benchmarks (median NRR 101%), Skok SaaS Metrics 2.0 (LTV:CAC about 3x guideline), OpenStax Principles of Accounting Vol. 2 sec. 3.5 (DOL definition). Do not write any chapter text.