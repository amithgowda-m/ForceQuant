# ForceQuant

Data, tables and figures for a Porter's Five Forces (+ Complementors) analysis of ZM, MSFT, MDB, ESTC and NVDA
from SEC filings (dual track: financial ratios + 10-K text). Hypothetical Case 4 (CloudMetrics) is computed from
equations only. Run instructions, full method and limitations are added in Stage 8.

## Setup
- `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and set `SEC_USER_AGENT` to a real name and contact email (required by SEC).

## Labelling sample (Stage 5)
Labels in `data/processed/llm_labels.csv` are **LLM labels (`llm_label`)** written by Claude Code in-session
following `LABEL_GUIDE.md`; a 40-paragraph sheet is checked by a human (`human_label`). Each row has `sample_type`:

- **random** (200 rows): 20 unique paragraphs (by `para_hash`) per company x item (Item 1A, Item 7), seed 42.
- **enriched** (163 rows): added because the random sample had fewer than 15 positives for every force.
  Item 1A only; unique texts not already sampled; for each force (in the order below), up to
  6 paragraphs per company whose text matches the force's keyword regex (case-insensitive),
  topped up from other companies to 30; a paragraph is used for at most one force; seed 42.
  Keywords only select candidates; every label is assigned by reading the paragraph against `LABEL_GUIDE.md`,
  in a shuffled order that hides which force selected it. Enriched rows have a much higher positive rate than
  the corpus, so metrics and shares must be reported on the random rows separately (see Stage 6/7 notes).

| Force | Keyword regex (config.yaml `labelling.enrichment.keywords`) | Rows selected per company | Rows | Rows labelled with that force |
|---|---|---|---|---|
| Competitive_Rivalry | `intense(ly)? compet|highly competitive|competitors (offer|may|could|have|with)|price competition|pricing pressure|bundl|market share` | ESTC 8 / MDB 6 / MSFT 4 / NVDA 6 / ZM 6 | 30 | 16 |
| Threat_of_New_Entry | `new entrants?|barriers? to entry|emerging (companies|competitors)|start-?ups?|new competitors|enter (our|the|these) markets?` | ESTC 3 / MDB 3 / MSFT 2 / NVDA 3 / ZM 2 | 13 | 10 |
| Threat_of_Substitutes | `open[- ]source|in-house|build their own|develop their own|managed services?|alternative technolog|substitut` | ESTC 6 / MDB 6 / MSFT 6 / NVDA 6 / ZM 6 | 30 | 10 |
| Buyer_Power | `price concessions|discounts?|negotiat|switching costs?|switch to|large customers|concentration of|limited number of customers|delay(ed|ing)? (their )?purchas` | ESTC 10 / MDB 4 / MSFT 3 / NVDA 5 / ZM 8 | 30 | 7 |
| Supplier_Power | `third-party data cent|rely on third part|limited number of (suppliers|vendors)|sole[- ]source|single[- ]source|suppliers?|foundr|cloud (hosting|infrastructure) providers?|Amazon Web Services` | ESTC 6 / MDB 0 / MSFT 7 / NVDA 11 / ZM 6 | 30 | 11 |
| Complementors | `ecosystem|partner network|integrators?|integrations?|marketplace|third-party (developers|applications|apps)|resellers|channel partners` | ESTC 6 / MDB 6 / MSFT 6 / NVDA 6 / ZM 6 | 30 | 4 |

Only 13 Item 1A paragraphs outside the random sample matched the Threat_of_New_Entry keywords; MDB had no
Supplier_Power keyword matches (other companies topped it up).

Relabelling after review: `data/processed/llm_relabels.csv` (old -> new label with reason; also in `outputs/issues.log`).
