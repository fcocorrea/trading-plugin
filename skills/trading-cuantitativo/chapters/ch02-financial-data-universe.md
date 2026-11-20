# Chapter 2: The Financial Data Universe

## Core Idea
Every dataset embeds definitions (timestamps, adjustments, identifiers, revisions), and most "alpha" failures in production stem from a handful of data defects: point-in-time, survivorship, corporate actions, and identifier joins. Sourcing, validation, and storage are risk management that determines research velocity.

## Frameworks Introduced
- **Market / Fundamental / Alternative data taxonomy**: Market = what the market did (quotes, trades, aggregates); Fundamental = economic drivers of value with release lags; Alternative = signals outside standard feeds, often proxies.
  - **When to use**: to reason about which real-world process a dataset measures and what it omits.
  - **How**: before modeling, pin down four items: timestamp semantics, corporate-action adjustment methodology, identifier stability, handling of revisions/restatements; store them as pipeline metadata.
- **Five-dimension data quality**: Timeliness, Completeness, Accuracy, Consistency, Validity. Prioritize checks that create lookahead (PIT, backfills, revisions) and universe leakage (survivorship).
- **Point-in-time (PIT) correctness / bitemporal storage**: each value reflects what was knowable at the decision.
  - **When to use**: all fundamental or alternative data.
  - **How**: store valid time + availability timestamp (`available_at`), `source_id`, `entity_id`, version id. Join rule: `available_at <= decision_time`; for identifier mappings `effective_date <= decision_time < end_date`. Three checklist dimensions: Values, Universe, Identifiers.
  - **Why it works / failure mode**: reconstructs the exact information set at t. Fails silently with "correct" joins that were impossible at the time; passing value PIT but failing universe PIT still biases results.
- **Survivorship bias remedy**: point-in-time universe construction (the 2010 backtest uses 2010 membership, with correct delisting returns and corporate actions).
  - **Why it works / failure mode**: the sign of the bias depends on the dominant cause of delisting (bankruptcies inflate, M&A can deflate). Using current S&P 500 constituents bakes in the success filter.
- **Vendor due diligence (3 questions)**: can I trust the numbers (quality)? can I use them as intended (legal)? will it behave as a reliable dependency (technical/commercial)?
- **Build vs Buy**: build what differentiates (ingestion, storage/versioning, validation, canonical IDs, sessionization, adjustment policies, roll rules, PIT features); buy what is commodity (long clean histories, corporate actions at scale, survivorship-aware universes, FIGI crosswalks).
- **Data governance (5 areas)**: lineage, versioning (never overwrite), cleaning logs, validation automation (Great Expectations, Pandera; fail loudly), parallel runs when switching vendors.
- **Default storage architecture**: partitioned Parquet + DuckDB (SQL) + Polars (feature engineering); servers only if concurrency/governance justify them.

## Key Concepts
- **Event time / Knowledge time / As-of time / Reference period**: when it happened, when it became public, the reconstructed historical moment, which period the value refers to.
- **NBBO**: consolidated best bid/offer in the US; direct feeds are faster and richer but cost more.
- **Continuous futures series**: perpetual ticker built with roll rules (calendar, volume, open interest) and adjustment (ratio or difference).
- **Back-adjusted futures ≈ excess return**: they do not include collateral return unless it is modeled.
- **Derived option surfaces**: the IV surface (ATM vol, skew, term structure) is modeled, not the raw chain.
- **AMM / DEX**: price set by reserves, not an order book; "depth" = slippage at a given size.
- **MEV**: validator profit from reordering/inserting/censoring transactions.
- **Matrix pricing**: mapping illiquid bonds to similar liquid ones; introduces model assumptions.
- **ETP**: listed wrapper (ETF with creation/redemption, ETN with issuer credit risk, CEF with persistent premium/discount).
- **ASOF join**: aligns unsynchronized events (a trade with the prevailing quote); requires sorted inputs.
- **Lakehouse formats** (Delta, Iceberg, Hudi): Parquet + transaction log (ACID, schema evolution, multi-writer).

## Mental Models
- Use **"what was knowable at t"** when working with fundamentals/alt data, not "what is the value."
- Use **observability by asset class** (high for exchange-traded, low for OTC) when deciding which microstructure tools apply.
- Use **"price is a convention"** in FX/OTC: choose a venue and aggregation rule and document it.
- Use **partitioned Parquet first** when starting a project; add a server DB only when concurrency/governance demand it.
- Use **sourcing as risk management** when choosing a vendor: PIT and survivorship prevent the most expensive mistakes.

## Anti-patterns
- **Querying "Q4 2019 revenue" today for a 2020 backtest**: uses restated values (lookahead).
- **Universe = current constituents**: bakes in success.
- **Mixing PIT adjustments with back-adjusted (end-of-sample factors)**: introduces lookahead.
- **Joins by ticker without effective dates**: ticker reuse produces fabricated correlation.
- **Unadjusted price returns**: a 7:1 split looks like a ~86% crash.
- **Treating OTC quotes as executable** or "assessed" spot as an exchange price.
- **Full option chain as a single dataset**: mixes liquid data with illiquid noise.
- **Trusting reported volume on crypto venues without screening**; treating AMM quotes as order book depth.
- **Looking at NAV as tradable return** or using current holdings for past exposures (look-through leakage).
- **Silently dropping invalid records**: fail loudly.

## Code Examples
There is no inline code in the chapter; implementations live in notebooks (13_data_quality_framework with ml4t-data and AnomalyManager; 14_point_in_time_validation; 06_futures_continuous; 18_data_management with DataManager/Universe/HiveStorage). Join rule the text prescribes:
```
available_at <= decision_time
effective_date <= decision_time < end_date
```
What it demonstrates: the two temporal constraints that every cross-source join and identifier mapping must respect.

## Reference Tables
| Asset Class | Observability | Key failure modes | Engineering priority |
|---|---|---|---|
| Equities | High (consolidated) | Corp actions, fragmentation, identifiers | Adjustment methodology |
| Futures | High | Roll rules, continuity | Continuous series construction |
| Options | High (sparse tails) | Illiquid noise, surface | Surface representation |
| Digital assets | Medium | Volume integrity, 24/7 | Venue screening |
| FX | Low (OTC) | No consolidated tape, close conventions | Aggregation rule |
| Fixed income | Low | Matrix pricing, indicative vs firm | Liquidity inference |
| Swaps/OTC | Low | Curve construction, conventions | Curve-based representation |
| Commodities | Medium | Spot ambiguity, delivery specs | Contract spec in metadata |

| Provider tier | PIT | Survivorship-aware | Best for |
|---|---|---|---|
| Free/public | Rare (ALFRED) | Rare | Learning, prototypes |
| Prosumer API-first | Partial | Sometimes | Individual research |
| Institutional | Often | Often | Production, regulated |

| Format (1M OHLCV rows) | Size | Write | Full scan |
|---|---|---|---|
| CSV | 136 MB | 0.13s | 0.05s |
| Parquet | 40 MB | 0.07s | 0.02s |
| HDF5 | 71 MB | 0.13s | 0.40s |

| Goal | Default | Alternative |
|---|---|---|
| Research velocity | Parquet + Polars | Parquet + DuckDB |
| SQL over files | DuckDB + Parquet | Polars lazy |
| Production reliability | PostgreSQL | TimescaleDB |
| Extreme time-series throughput | kdb+ | ClickHouse |
| High ingestion | QuestDB | ClickHouse |
| In-memory ASOF | Polars | pandas |
| ASOF with SQL | DuckDB | QuestDB |
| Lowest operational burden | Parquet + DuckDB | Managed Postgres |

Minimum viable validation: PIT with no zero-lag records; universe with no future filter; documented adjustments; joins on permanent IDs + dates; explicit timezone; stale quote/outlier detection; versioning with reproducible as-of queries.

## Worked Example
Survivorship in a panel of 3,199 US stocks (WIKI/Quandl): 777 (24.3%) delisted before the end of the dataset, all post-2014. Monte Carlo (1,000 draws per scenario) simulating terminal returns by exit reason (empirical scenario based on CRSP 2010-2020: 26% compliance failures at -60%, 65% acquisitions at +25%, 9% other at 0%). Result 2014-2018: the bias inflated perceived returns by 3.5-15.3 percentage points (+8.0 in the empirical scenario); exiting stocks returned a median +13.5% vs +30.8% for survivors. Caveat: the +25% for M&A is usually already priced in before delisting, so it is overstated in M&A-heavy panels. Decision: build a PIT universe. Other numbers: 7:1 split = ~-86% apparent; AAPL 5.9x raw vs 398x adjusted (~68x understatement); 473 OHLC violations (~0.1%) across 100 ETFs from adjusting the 4 prices independently.

## Key Takeaways
- Document and pin down timestamps, adjustments, identifiers, and revisions as metadata before modeling.
- Store bitemporally (valid time + `available_at`) and constrain every join temporally.
- Build point-in-time universes; never use current constituents.
- For futures, store raw contracts plus continuous variants; for options, model surfaces; for OTC, distinguish observed vs derived.
- Vet vendors on quality, legal (MNPI, GDPR/CCPA, usage rights, auditability), and technical grounds; run sources in parallel when migrating.
- Storage default: partitioned Parquet + DuckDB/Polars.

## Connects To
Ch 1 (workflow, PIT as an invariant), Ch 3 (order book, bars, corporate actions in detail, microstructure), Ch 4 (PIT fundamentals, FRED vintages, CFTC COT, prediction markets, on-chain, SEC EDGAR, Section 4.4 alt data evaluation), Ch 5 (synthetic data), Ch 6 (strategy specification), Ch 10 (NLP features), Ch 25 (operations).

## Code Reference

Official repository: [02_financial_data_universe/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/02_financial_data_universe). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_us_equities_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/01_us_equities_eda.ipynb) + `.py` | 2.2 | This notebook introduces the Wiki Prices dataset - a survivorship-bias-free collection of US equity prices. |
| [`02_corporate_actions.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/02_corporate_actions.ipynb) + `.py` | 2.2 | This notebook demonstrates how stock splits and dividends break historical price series, and shows the industry-standard backward adjustment methodology used by major da… |
| [`03_etfs_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/03_etfs_eda.ipynb) + `.py` | 2.2 | This notebook introduces the 50-ETF universe that serves as the foundation for the ETF Rotational Momentum case study throughout the book. |
| [`04_cme_futures_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/04_cme_futures_eda.ipynb) + `.py` | 2.2 | This notebook introduces the CME futures dataset shipped with the book. |
| [`05_futures_session_aggregation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/05_futures_session_aggregation.ipynb) + `.py` | 2.2 | This notebook converts hourly continuous futures data (stored in UTC) to session-aware daily bars. |
| [`06_futures_continuous.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/06_futures_continuous.ipynb) + `.py` | 2.2 | This notebook tackles one of the most critical challenges in futures analysis: creating continuous price series from individual expiring contracts. |
| [`07_sp500_options_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/07_sp500_options_eda.ipynb) + `.py` | 2.2 | This notebook provides a comprehensive exploration of the AlgoSeek S&P 500 Options Analytics dataset. |
| [`08_options_greeks_computation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/08_options_greeks_computation.ipynb) + `.py` | 2.2 | This notebook derives and implements the Black-Scholes option pricing framework from first principles. |
| [`09_options_continuous.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/09_options_continuous.ipynb) + `.py` | 2.2 | Options are time-decaying instruments. |
| [`10_crypto_perps_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/10_crypto_perps_eda.ipynb) + `.py` | 2.2 | This notebook introduces the cryptocurrency dataset from Binance Futures. |
| [`11_crypto_premium_analysis.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/11_crypto_premium_analysis.ipynb) + `.py` | 2.2 | This notebook demonstrates how to work with Binance perpetual futures premium index data - the foundation for funding rate arbitrage strategies. |
| [`12_fx_pairs_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/12_fx_pairs_eda.ipynb) + `.py` | 2.2 | This notebook introduces the FX dataset from OANDA. |
| [`13_data_quality_framework.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/13_data_quality_framework.ipynb) + `.py` | 2.3 | The ml4t-data library provides purpose-built tools for financial data quality. |
| [`14_point_in_time_validation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/14_point_in_time_validation.ipynb) + `.py` | 2.3 | Point-in-time correctness is essential for valid backtesting. |
| [`15_survivorship_bias_detection.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/15_survivorship_bias_detection.ipynb) + `.py` | 2.3 | Survivorship bias is arguably the most dangerous form of data contamination in quantitative finance. |
| [`16_provider_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/16_provider_comparison.ipynb) + `.py` | 2.3 | ML4T Third Edition - Chapter 2: The Financial Data... |
| [`17_complete_pipeline.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/17_complete_pipeline.ipynb) + `.py` | 2.4 | This notebook demonstrates end-to-end data pipelines, bringing together concepts from this chapter: Uses crypto_perps, wiki_provider data. |
| [`18_data_management.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/18_data_management.ipynb) + `.py` | 2.4 | Previous notebooks fetched and validated data. |
| [`19_incremental_updates.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/19_incremental_updates.ipynb) + `.py` | 2.4 | The previous notebook introduced DataManager and HiveStorage. |
| [`20_storage_benchmark_file.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/20_storage_benchmark_file.ipynb) + `.py` | 2.4 | Focus: Pure file format comparison (no query engines) Technologies: CSV, Parquet, Feather (Arrow IPC), HDF5 Operations: Write, Read (with forced materialization), Column… |
| [`21_storage_benchmark_database.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/21_storage_benchmark_database.ipynb) + `.py` | 2.4 | > Docker required: This notebook depends on the benchmark environment and > database services. |
| [`22_pandas_polars_benchmark.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/02_financial_data_universe/22_pandas_polars_benchmark.ipynb) + `.py` | 2.4 | DataFrame-engine comparison (pandas vs Polars) across read, filter, groupby, join, and lazy operations on synthetic financial data at S/M/L scales. |
