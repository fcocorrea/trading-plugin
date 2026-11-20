# Chapter 4: Fundamental and Alternative Data

## Core Idea
Turning fundamental and alternative data into signals depends on disciplined pipelines, not clever algorithms: two failures dominate, time consistency (point-in-time) and entity consistency (entity resolution). The chapter provides the engineering playbook (bitemporal, EDGAR, entity resolution, alt data evaluation, text pipeline).

## Frameworks Introduced
- **Bitemporal storage + as-of query**: each fact carries the period it describes (valid time) and the timestamp at which that specific version became observable (knowledge time); a period can have several versions (original, amendment, restatement).
  - **When to use**: all revisable data (SEC, macro, commodities, crypto).
  - **How**: (1) choose the latest reported period eligible at t; (2) within that period, the latest version with availability timestamp <= t. Using `accepted_at` from the 10-Q/10-K is conservative (the market often reacts earlier to the 8-K Item 2.02). XBRL Frames returns "last filed": useful for snapshots and prototypes, not PIT-safe; reconstruct from per-filing records.
  - **Why it works / failure mode**: eliminates restatement lookahead (Q3 revenue $95M original -> $98M -> $100M: a backtest using $100M in October applies the later accounting view). Fails if the vendor does not explain revisions, amendments, and identifier history: presume non-PIT until proven otherwise.
- **Timestamp authority by modality** (Table 4.1): SEC = `accepted_at`; Macro = normalized official release timestamp + ALFRED; Commodities = agency release + mapping to contract roll; Crypto = block time + confirmation/finality policy + ingestion time (available only after N confirmations, avoids reorg lookahead).
- **Hierarchical entity resolution**: Stage 1 deterministic (standard identifiers), Stage 2 probabilistic (fuzzy), Stage 3 manual review with audit trail; embeddings/knowledge graphs when these fail (Ch 10, Ch 23).
  - **When to use**: before combining different sources.
  - **How**: resolve to the correct layer (legal entity / security / derivatives contract) with three master tables linked by time-valid mappings; fuzzy with rapidfuzz (Levenshtein, Jaro-Winkler, token Jaccard); tiered thresholds: auto-accept ~>=95%, manual review 85-95%, <85% escalate to richer features (addresses, websites, descriptions); calibrate against a labeled sample.
  - **Why it works / failure mode**: prioritizes precision; one false positive contaminates every join (ZM vs ZOOM Technologies). Fails by mapping to the wrong layer or with a static mapping.
- **Temporal entity mapping**: table `identifier_type, identifier_value, effective_date, end_date`; every join constrained to `effective_date <= decision_time < end_date` (e.g., FB 2012-05-18..2022-06-08, META from 2022-06-09).
- **Mapping QA**: confidence tiers per batch; monthly sampling audit of 1-5% of auto-accepted matches; weekly drift monitoring (match rate, confidence distribution). Metrics: match rate per tier, score distribution (watch for bimodality), false positive estimate, temporal drift.
- **Alternative data evaluation framework (4 dimensions)**: Signal content (uniqueness, decay/durability, failure modes), Data quality (stability, coverage, latency), Legal (hard-fail gate), Commercial/engineering (TCO, operationalization test).
  - **Decision rule**: favor hard constraints (legal, PIT integrity) over marginal differences; only compare scores between datasets that pass the hard gates.
- **Build vs buy for alt data**: hybrid: buy clean data from mainstream sources; build only truly proprietary signals (alpha crowding, asymmetric legal risk, maintenance burden).
- **Text pipeline (4 steps)**: document selection (primary document, not exhibits), section extraction (with quality checks), cleaning/normalization (preserve paragraphs), PIT-correct storage (cik + accession + section as PK; `filing_date`, `accepted_at`, `period_end`; raw and clean; extraction metadata).

## Key Concepts
- **Knowledge date vs valid period**: the filing date comes from submission metadata, not from the financial data.
- **XBRL**: each value carries concept, period, unit, and dimensions; companies can extend the taxonomy with custom tags.
- **FSN datasets**: SUB, NUM, TAG, DIM tables (+ TXT, PRE, CAL, REN) with 10-K/10-Q/8-K since 2009.
- **ALFRED**: FRED with real-time validity windows per vintage (which value was current between two dates).
- **Revision ladder**: explicit revision history stored when the strategy is sensitive to revisions.
- **COT**: positions as of Tuesday, published Friday 3:30 pm ET; PIT only after the release.
- **CIK / LEI / FIGI / CUSIP-ISIN-SEDOL**: SEC filing entity / legal entity (20 chars, ISO 17442) / instrument / security identifiers (US / global / UK).
- **TVL**: Total Value Locked, proxy for capital committed to a DeFi protocol; may reflect double counting, leverage, or token price.
- **McLean-Pontiff**: published predictors lose ~58% of performance post-publication.
- **MNPI by narrowness**: small panels or narrow geographies can imply material non-public information.

## Mental Models
- Use **"which version was eligible at t"** when querying fundamentals; not "what is the value today."
- Use **precision first, recall second** in entity resolution: a missed match costs coverage, a false one corrupts everything.
- Use **hard gates before scoring** when evaluating alt data: legal and PIT first.
- Use **the naive-vs-as-known test** when you suspect leakage: run the same strategy with an as-of view and with "latest today"; if the second looks better, it is later information, not signal.
- Use **timestamped events** (not interpolation) for macro: carry forward only after the release.

## Anti-patterns
- **Using the final restated value in past decisions**: lookahead (one of the "seven sins" of Luo et al. 2014).
- **XBRL Frames as a PIT source**: returns the last filed.
- **Interpolating macro** backward: mixes future values into the past.
- **Hardcoding release times (8:30 ET)**: calendars change; store the published timestamp.
- **Joins by static ticker**: ticker reuse and rebranding create datasets that never existed.
- **Resolving to the wrong layer** (Class A vs B, ADR vs ordinary share, issuer vs security).
- **Name-only fuzzy matching** for subsidiaries (Amazon Web Services vs Amazon.com Inc).
- **Treating TVL or on-chain activity as net usage** without checking bots, double counting, definitions.
- **Adopting alt data without an incremental test** over the feature baseline.

## Code Examples
No inline snippets. Described query pattern (reconstructed):
```sql
SELECT *
FROM facts
WHERE entity_id = :id
  AND available_at <= :t
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY period_end ORDER BY available_at DESC) = 1
ORDER BY period_end DESC
LIMIT 1;
```
What it demonstrates: a two-step as-of query (latest available version per period, then latest eligible period); it is an illustrative reconstruction, not code from the book.

## Reference Tables
| Modality | Authoritative timestamp | Note |
|---|---|---|
| SEC | `accepted_at` | Optional: earlier earnings release |
| Macro | Normalized official release | ALFRED for vintages |
| Commodities | Agency release + roll mapping | EIA, USDA, CFTC |
| Crypto | Block time + finality + ingestion | N confirmations |

| Release | Schedule | Note |
|---|---|---|
| EIA Weekly Petroleum | Wednesday 10:30 ET | Holiday exceptions |
| USDA WASDE | 12:00 ET on scheduled dates | Risk for grains/oilseeds |
| CFTC COT | Friday 3:30 pm ET (data as of Tuesday) | Holiday delays |

| Confidence band (entity resolution) | Action |
|---|---|
| >= ~95% | Auto-accept |
| 85-95% | Manual review |
| < 85% | Escalate to rich features / embeddings |

| QA check | Method | Frequency |
|---|---|---|
| Confidence tiers | Calibrated cutoffs | Per batch |
| Sampling audit | Verify 1-5% of auto-accepted | Monthly |
| Drift monitoring | Match rate and distribution | Weekly |

| Alt data dimension | Key question |
|---|---|
| Signal | Incremental over baseline? Half-life? |
| Quality | Versioning, coverage, accurate latency? |
| Legal | Provenance, MNPI, GDPR/CCPA, usage rights? |
| Commercial | TCO and conservative value after costs/capacity? |

## Worked Example
Restatement and leakage: a company reports Q3 revenue of $95M in October, restates it to $98M in the next 10-Q and to $100M in the 10-K. A naive "latest value" dataset assigns $100M to dates before the release: the P/E strategy ends up with lookahead. A PIT dataset keeps $95M until the restatement date. Decision: run the strategy twice (as-known vs latest-today) and distrust a vendor that does not explain revisions.
Text pipeline with Apple's 10-K: 2025 Risk Factors of 9,792 words with 92.9% word overlap vs the prior year (Jaccard 0.31), 28 new risk terms; lesson: measure boilerplate and decide whether to remove it (novelty features) or keep it (level sentiment). Another data point: DeFi TVL (4 chains, 2017-2026) as a predictor of forward returns with peak IC 0.12 = weak signal. 10y-2y curve inverted on 15.5% of days since 2000 and 36% since 2020.

## Key Takeaways
- Store valid time and `accepted_at`/release timestamp for every source; always query as-of.
- Resolve entities with an entity/security/contract master and mappings with `effective_date`/`end_date`.
- Prioritize precision in matching; audit 1-5% of auto-accepted matches every month and monitor drift weekly.
- Align macro by release (ALFRED/vintages), commodities by active contract, and crypto by finality.
- Evaluate alt data with an incremental test, decay (~58% post-publication as a baseline), legal as a hard gate, and TCO.
- For text, key = cik + accession + section; store raw/clean and extraction metadata.

## Connects To
Ch 2 (PIT vocabulary, checklist, identifiers), Ch 3 (futures roll logic), Ch 8 (feature engineering on PIT artifacts), Ch 10 (NLP features, news surprise, FinBERT, embeddings), Ch 23 (knowledge graphs, FinKG), Ch 5 (synthetic data).

## Code Reference

Official repository: [04_fundamental_alternative_data/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/04_fundamental_alternative_data). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_academic_characteristics.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/01_academic_characteristics.ipynb) + `.py` | 4.1 | Reads the Chen, Pelger and Zhu (2021) panel: fifty years of monthly US equity observations, each carrying 46 rank-normalized firm characteristics and the following month… |
| [`02_sec_filing_explorer.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/02_sec_filing_explorer.ipynb) + `.py` | 4.2 | Works through the EDGAR surface once, through the EdgarTools library: finding a filer by its Central Index Key, filtering its submissions by form, pulling the three fina… |
| [`03_sec_form4_insider_transactions.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/03_sec_form4_insider_transactions.ipynb) + `.py` | 4.2 | Parses Form 4 insider trades from the raw XML, reconciles the extracted rows against the transaction blocks in the files, and separates the trades that reflect a decisio… |
| [`04_sec_xbrl_fundamentals.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/04_sec_xbrl_fundamentals.ipynb) + `.py` | 4.2 | Reads a quarterly fundamentals panel assembled from the SEC's XBRL Frames API, measures how long after each period end its filing arrived, and builds the as-of query tha… |
| [`05_entity_resolution.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/05_entity_resolution.ipynb) + `.py` | 4.2 | Builds the three stages of a name-to-identifier mapping: a coalescing join on identifiers, a fuzzy string score, and a sentence embedding. |
| [`06_fred_macro_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/06_fred_macro_eda.ipynb) + `.py` | 4.3 | First contact with the FRED macro panel: what grid its rows sit on, and how to recover each series' release frequency from a file that has already carried every value fo… |
| [`07_macro_data_alignment.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/07_macro_data_alignment.ipynb) + `.py` | 4.3 | Re-dates each macro observation from the period it measures to the day it can be assumed public, derived from each agency's release schedule at the late end of its range… |
| [`08_futures_positioning.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/08_futures_positioning.ipynb) + `.py` | 4.3 | Turns the CFTC's weekly Commitment of Traders reports into a positioning signal: resolving the several contract markets that share one product code, standardizing a net… |
| [`09_onchain_fundamentals.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/09_onchain_fundamentals.ipynb) + `.py` | 4.4 | Takes total value locked, the closest thing decentralized finance has to a fundamental, and tests whether it predicts ether returns — mostly by establishing how little a… |
| [`10_institutional_holdings_13f.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/10_institutional_holdings_13f.ipynb) + `.py` | 4.4 | Reads a whole quarter of Form 13F filings — several million positions from around seven thousand managers — and does the screening an aggregate over that file needs firs… |
| [`11_defi_tvl_evaluation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/11_defi_tvl_evaluation.ipynb) + `.py` | 4.4 | Runs the four-question alternative-data evaluation on one real dataset — signal, data, legal, commercial — keeping the questions that can block on their own separate fro… |
| [`12_kalshi_prediction_markets.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/12_kalshi_prediction_markets.ipynb) + `.py` | 4.4 | Reads the first CFTC-designated prediction market: what a binary contract's price means and which price the feed actually carries, how a ladder of thresholds on one even… |
| [`13_polymarket_prediction_markets.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/13_polymarket_prediction_markets.ipynb) + `.py` | 4.4 | The unregulated counterpart, settled in a stablecoin on the Polygon blockchain and closed to US persons. |
| [`14_text_data_extraction.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/04_fundamental_alternative_data/14_text_data_extraction.ipynb) + `.py` | 4.5 | Gets from a filing to a section reliably: locating an item heading that appears in the contents, in cross-references and once as the section itself, checking the extract… |
