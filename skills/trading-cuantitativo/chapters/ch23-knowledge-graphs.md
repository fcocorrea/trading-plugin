# Chapter 23: Knowledge Graphs

## Core Idea
A knowledge graph (nodes = entities, typed edges = relationships, with identifiers, timestamps and provenance) is only justified when the question is genuinely relational (paths between entities). The chapter covers explicit financial KGs for deterministic traversal and auditable answers: contract-governed construction, Graph RAG, ML features, statistical networks, and temporal discipline to avoid leakage.

## Frameworks Introduced
- **Path pattern test**: rephrase the question as a path pattern ("entities connected by chains of relationships").
  - When to use: before investing in graph infrastructure.
  - How: if it decomposes into multi-hop paths, contagion, co-ownership or temporal evolution of relationships, use a graph; if it is an attribute lookup on one entity, narrative synthesis over a broad corpus, or a sparse graph (average degree < 2-3), do not.
  - Why it works / failure mode: recursive joins become unmanageable beyond 2 hops in SQL, and vector retrieval infers paths from fragments. Classic mistake: encoding attributes as relationships (Company -HAS_REVENUE-> $500B). Sparsity (e.g., <200 edges in the S&P 100) is usually an extraction failure.
- **Three governance contracts (identity, schema, provenance)**: identity (mentions are not identities; CIK/LEI/CUSIP/ISIN as primary keys, tickers as attributes with validity intervals, corporate actions as events); schema (finite, typed vocabulary with an allowlist and source-target constraints); provenance (source document ID, public date, span/section/page, extraction timestamp, extractor version).
- **Five-stage extraction workflow**: (1) targeted document slicing (10-K Items 1, 1A, 7); (2) schema-constrained generation (JSON Schema/typed output); (3) canonicalization and dedup; (4) rule validation; (5) human review queue. It must be idempotent (same filing + same extractor = same objects; merge-style by canonical identity).
  - Why it works / failure mode: the schema is defined first and the LLM emits ready-made objects; the classic pre-LLM failure was normalizing text and then forcing it into the graph (duplicates, drifting relationships).
- **Reflection-driven correction (FinReflectKG)**: extract candidate triples, evaluate them against rules, propose fixes, iterate until convergence; useful for entity normalization.
- **5-stage Graph RAG**: (1) query routing (graph/document/hybrid); (2) text-to-Cypher with a compact schema, or parameterized templates + LLM router; (3) query safety validation; (4) deterministic retrieval with provenance; (5) grounded synthesis with dual citation (graph row + disclosure text).
- **Three-timestamp model**: event time, disclosure time, extraction time; visibility is decided by disclosure time.
- **Leakage-safe evaluation protocol (5 rules)**: split by disclosure time; embargo near split boundaries; cutoff on every query path; log snapshot hash and extractor version; compare against tabular-only and static-graph baselines.
- **Recommended graph ML path**: (1) graph analytics and hand-crafted features; (2) portfolio methods with explicit structure (MST, community detection); (3) GNN/embeddings only if they improve out-of-sample net of costs and leakage; (4) learned features as supplements.

## Key Concepts
- **Bipartite 13F network**: institutions and securities; projecting onto securities yields co-ownership.
- **Crowding score**: count of institutional holders relative to the cross-sectional median.
- **Ownership Herfindahl Index**: holder concentration; 3 large funds are more fragile than 30 small ones.
- **Co-ownership Jaccard**: similarity of holder sets between two stocks; predicts correlation beyond fundamentals.
- **PageRank / Betweenness / Clustering coefficient / Degree**: weighted importance, bottlenecks, path redundancy, direct exposure.
- **MST (Mantegna 1999)**: minimum spanning tree over correlation distances to reveal hierarchical structure.
- **Message passing (GNN)**: node representation updated by aggregating neighbors (GCN, GAT, GraphSAGE).
- **FIBO**: financial ontology with more than 1,200 classes; an impractical starting point.
- **Position masking / confidential treatment**: hidden 13F positions that appear in later amendments and bias crowding features downward.
- **Relationship churn / centrality momentum / concentration drift / theme acceleration**: temporal features of the structure.

## Mental Models
- Use Graph RAG when the answer depends on relationship paths and vector RAG when it is open-ended narrative; in production, hybrid with query routing.
- Use hand-crafted graph features as the baseline first: more auditable, stable and explainable than GNNs.
- Use disclosure time as the visibility gate, never event time or extraction time.
- Use a small schema (5 object types; 8-15 typical relationships) that grows additively, instead of adopting all of FIBO.
- Use parameterized Cypher templates when you need auditability; they reduce the attack surface compared with free-form Cypher.

## Anti-patterns
- **Persisting "Apple", "Apple Inc.", "AAPL" as separate nodes**: duplicates that inflate path counts.
- **Measuring only extraction precision**: hides identity or provenance collapse.
- **Placeholder entities ("vendors", "leaf merchants")**: they pollute the graph; they require a stop list.
- **Writing without a read-only role or allowlist in text-to-Cypher**: injection risk and runaway queries.
- **Interpolating strings into queries**: instead of parameterizing.
- **Forgetting the cutoff filter in historical queries**: leaks post-cutoff information.
- **Using a 13F's quarter-end date**: the holding is tradable only after the filing date.
- **Treating the MST as stable**: it changes a lot between adjacent windows and generates turnover.
- **GNN by default for alpha**: complexity without proportional benefit.
- **Assuming 10-Ks are complete**: they are high-precision, low-recall.

## Code Examples
No literal code in the text. Visibility-safe query pattern as described (reconstructed, illustrative):

```
MATCH (a:Company {cik: $cik})-[r:HAS_SUPPLIER]->(s:Company)
WHERE r.public_date <= $cutoff
RETURN s.name, r.source_doc, r.evidence_span
```
What it demonstrates: filter every edge by disclosure time before any aggregation, with parameters and no interpolation.

## Reference Tables
**Topology features (Table 23.2)**

| Feature | Calculation | Interpretation |
|---|---|---|
| PageRank | iterative influence propagation | systemic importance in the supply chain |
| Betweenness | fraction of shortest paths | bottleneck |
| Clustering coef. | local density | redundancy of alternative paths |
| Degree | direct connections | immediate exposure |

**GNN maturity**

| Domain | Maturity | Evidence |
|---|---|---|
| Fraud detection | Production-ready | Elliptic Bitcoin (Weber 2019) |
| Systemic risk | Emerging | research and regulatory pilots |
| Alpha generation | Experimental | little reproducible evidence after costs |

**Graph DB (Table 23.5)**: Neo4j (teaching, research, mid-scale production; default); Neptune or another managed service (operations, IAM, more cloud coupling); TigerGraph (massive graph analytics, higher complexity); PostgreSQL with recursive CTEs for narrow graphs and 3-4 hop queries.

**8-K lag**: Items 1.01, 2.01, 5.02: 1-4 business days; 7.01: same day; 8.01: variable.

**Monitoring targets**: schema-valid rate > 90%, provenance coverage > 95%; plus duplicate-node rate and temporal-consistency rate.

## Worked Example
S&P 100 supply chain KG: 601 10-K filings, 101 companies, 1,057 unique relationships (276 supplier, 472 competitor, 309 customer) across 127 subject companies; TSMC, Foxconn and Samsung are the most connected suppliers (61, 33, 28 downstream companies). Combined features: 31-column matrix (topology 5, supply 7, holdings 9, temporal 6, cross-graph 4) with 13F (201,145 rows; 80 of 127 matched). Cross-graph interaction: supply_chain_crowding = supplier overlap x institutional crowding. Graph vs vector benchmark on 13F holdings (7 questions, 10,624 rows, 10 institutions, 250 stocks): support recall 1.00 vs 0.13; ~82 vs 267 tokens per query; FinReflectKG-MultiHop: +24% correctness and ~85% fewer tokens. Network portfolios (100 stocks, 2015-2018, 504-day lookback): Sharpe 1.58 / 1.51 / 1.54 (equal-weight / inverse-vol / network) and MaxDD -10.3% / -10.6% / -10.1%, no separation; contagion from BRK.B (high centrality) is -87.3% vs -23.0% from peripheral AAL (~3.8x). GAT+8 tabular factors IC 0.215 vs 0.233 tabular-only (-7.7%).

## Key Takeaways
- Apply the path pattern test before building a graph; discard it if the average degree is < 2-3.
- Define identity/schema/provenance before the prompt; store evidence on the edge; idempotent extraction.
- Enforce in text-to-Cypher: read-only role, allowlist, parameters, timeout/row cap/depth limit, blocking of write keywords, logging.
- Tag every edge with event/disclosure/extraction time and filter by disclosure time; for 13F use the filing date (45-day lag).
- Derive topology, supply risk, crowding and cross-graph features; add churn and centrality momentum only with multiple vintages.
- Keep GNNs as a supplementary experiment; validate with walk-forward and costs.
- Adopt additive schema versioning and keep a gold subset of filings for regression.

## Connects To
Ch 22 (RAG, 13F co-ownership graph, FinDER evaluation); Ch 10 (NLP); Ch 12 (gradient boosting); Ch 14 (factor models); Ch 17 (Herfindahl); Ch 24 (agents, memory, evidence); Ch 26 (governance); temporal leakage and point-in-time.

## Code Reference

Official repository: [23_knowledge_graphs/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/23_knowledge_graphs). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

This chapter has no notebooks; only `README.md` in the folder.

Other notebooks in the folder, not linked from the README: `01_sp100_sec_download.ipynb`, `02_supply_chain_kg_construction.ipynb`, `03_graph_rag_qa.ipynb`, `04_rag_comparison_benchmark.ipynb`, `05_institutional_holdings_kg.ipynb`, `06_gnn_feature_engineering.ipynb`, `07_dynamic_kg_temporal.ipynb`, `08_8k_event_extraction.ipynb`, `09_knowledge_graph_features.ipynb`, `10_network_portfolio_construction.ipynb`.
