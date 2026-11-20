# Chapter 22: RAG for Financial Research

## Core Idea
LLMs hallucinate with persuasive fluency, which is unacceptable in finance; the answer is architectural: Retrieval-Augmented Generation (RAG) forces the model to synthesize only over retrieved evidence, with metadata, timestamps and auditable citations. RAG is built as an engineering stack (ingestion, embeddings, hybrid retrieval, re-ranking, restrictive prompting, tool-verified computation), not as a prompt trick.

## Frameworks Introduced
- **Index-Retrieve-Generate pipeline**: (1) Index offline: parse, chunk, embed, store vectors + text + metadata in a vector DB; (2) Retrieve online: embed the query with the same model, top-k by cosine similarity; (3) Generate: prompt with query + chunks and the instruction "answer only from the context".
  - When to use: any Q&A over filings (10-K/10-Q), earnings calls, research reports, regulation.
  - How: orchestrate with LlamaIndex (RAG-centric) or LangChain (general agents/chains); Haystack as an alternative.
  - Why it works / failure mode: grounding is non-parametric and traceable to the source. It fails if the naive version (fixed-size chunks + generic embedding + simple vector search) is used on dense, jargon-heavy documents.
- **Parent Document Retrieval ("small-to-big")**: retrieve with small, precise chunks, pass the full parent section to the LLM.
  - When to use: when small chunks give precision but lose context ("Revenue grew 15%" without segment/period).
  - How: embed each paragraph, keep a chunk-to-parent mapping during ingestion, return the full subsection on a match.
- **Hybrid retrieval + Reciprocal Rank Fusion (RRF)**: score(d) = sum over methods of 1/(k + rank_r(d)), with k = 60 typical; a document absent from a ranking contributes 0.
  - When to use: queries with tickers, regulatory codes or exact figures plus concepts ("AAPL capital return program").
  - How: run BM25 and semantic search in parallel; fuse with RRF; no training.
  - Why it works / failure mode: it favors documents ranked well by several signals over those that dominate only one. Learned fusion gives a marginal improvement with more complexity. A generic web cross-encoder degraded MRR by 36% vs Hybrid: it may be poorly calibrated for finance.
- **Retrieve-extract-compute-narrate**: (1) retrieve evidence incl. tables; (2) extract numbers into a schema (cells, units, period, segment); (3) compute with code/tools, not with the LLM; (4) narrate with citations and a reproducible calculation trace.
  - When to use: every numerical question (YoY operating margin, current ratio vs median).
- **Reranking pipeline**: top-50 candidates from fast search, a cross-encoder re-scores them, top-5 go to the LLM (latency 100-200 ms per 50 chunks).
- **Constraint-based prompting ("prompt as contract")**: role assignment, context constraint ("ONLY"), uncertainty handling, citation instruction `[Page X, Section Y]`, structured output (direct answer / evidence with citations / caveats).
- **5-failure-mode diagnosis**: retrieval (low context recall), context (low context precision), synthesis (low faithfulness), computation (correct evidence, wrong arithmetic), abstention (fabricates when there is no answer).
  - How: (1) eval set of 50-100 queries with ground truth, stratified by single-hop/multi-hop/comparative and by document type, 10-20% unanswerable, validated by experts; (2) RAGAs baseline; (3) isolate the failure mode; (4) fix the weakest component; (5) continuous monitoring (TruLens).
- **CRAG / self-RAG**: the LLM evaluates retrieval quality and re-queries if the score falls below a threshold. Cost: one LLM call per cycle; limit rounds in interactive applications.
- **ReAct (Thought, Action, Observation, repeat)**: agent loop; RAG becomes one more tool (alongside web search, code interpreter, DB, APIs).

## Key Concepts
- **Hallucination**: plausible but false output, an intrinsic property of the probabilistic model, not a patchable bug.
- **Parametric vs non-parametric knowledge**: model weights vs a curated, verifiable external base.
- **Context confusion**: the LLM mixes fiscal years or sections because of fragmented chunks.
- **Structure-aware parsing**: parsers (LlamaParse, Docling, Marker) that preserve sections, tables and hierarchies.
- **Point-in-time integrity**: filter by the filing's publication date, not by the fiscal period label.
- **Matryoshka embeddings**: trained so that truncating to 256/512/1024 dims remains usable; truncating a normal embedding does not work.
- **Bi-encoder / cross-encoder / late interaction (ColBERT, MaxSim)**: embed separately and dot product; joint query-doc pair, more precise and expensive; per-token embeddings, intermediate with a much larger index.
- **HyDE**: embed an LLM-generated hypothetical answer instead of the query.
- **Pre-filtering vs post-filtering**: filtering metadata before ANN preserves recall; filtering afterward can destroy it.
- **Unsupported claim rate (RAGChecker)**: fraction of atomic claims without support in the retrieved evidence.
- **Retrieval bundle**: query, chunks, evidence spans, model versions and answer, stored per answer for audit.

## Mental Models
- Use chunking as a hyperparameter when you don't know the optimal size: evaluate recall on a grid (256, 512, 768, 1024) before committing.
- Use hybrid search when queries mix exact entities (tickers, codes) with concepts.
- Use a fine-tuned classifier for systematic breadth (thousands of stocks per day) and RAG for fundamental depth (due diligence on one company); sophisticated firms use both.
- Use long context only as a fallback for complex multi-hop; 5-10K tokens of precise chunks typically beat 100K+ of loose material.
- Use an agent only for truly open-ended exploration; well-designed LLM workflows usually win on structured, predictable tasks.
- Use "best on a public benchmark" as a hypothesis, not as a deployment conclusion.

## Anti-patterns
- **Fixed-size chunking of 500-1000 characters**: separates headers from values in tables and sections from their paragraphs.
- **Trusting fluency or post-hoc fact-checking**: it does not scale or fix the root cause.
- **Ingesting without temporal metadata**: retrieves 2022 guidance for a question about 2024; plausible, cited and obsolete.
- **Post-filtering on small subsets**: severely reduced recall.
- **Letting the LLM do arithmetic**: the evidence is correct but the derived figure is wrong; invisible to faithfulness metrics.
- **Assuming citations are faithful**: the LLM can cite passages that do not support the claim or invent pages.
- **Putting the entire 10-K into the context**: "lost in the middle", cost and time-to-first-token grow.
- **Tuning without measurement** or evaluating with 3 anecdotal queries.
- **Security as a separate audit**: it must live in the same evaluation harness.

## Code Examples
No inline code in the chapter; the core formula reconstructed:

```
RRF(d) = sum_{r in R} 1 / (k + rank_r(d))      # k = 60
```
What it demonstrates: fusing BM25 + semantic rankings is a sum of smoothed reciprocal ranks, with no training.

## Reference Tables
**Chunk size trade-off**

| Size | Advantage | Cost |
|---|---|---|
| 200-300 tokens | high retrieval precision | no context (segment, period) |
| 400-600 tokens + overlap 50-100 | default for financial docs | hyperparameter to evaluate |
| 1,000-2,000 tokens | self-contained context | diluted embedding, consumes window |

**RAGAs (Table 22.2)**

| Metric | Measures | Failure mode |
|---|---|---|
| Context precision | retrieved chunks are relevant | Context failure |
| Context recall | everything needed was retrieved | Retrieval failure |
| Faithfulness | answer grounded in the context | Synthesis failure |
| Answer relevance | answers the question | Synthesis failure |
| Abstention quality | declines if evidence is insufficient | Unanswerable query |

**Fine-tuned classifier vs RAG (Table 22.3)**

| Dimension | Classifier | RAG Q&A |
|---|---|---|
| Output | scores, labels | narrative with citations |
| Scalability | high (batch) | medium (per query) |
| Flexibility | low (fixed taxonomy) | high |
| Verifiability | indirect | direct (citations) |
| Updating | retrain | update corpus |
| Best use | systematic factors | fundamental due diligence |

**Vector DB**: native hybrid (Weaviate, Pinecone, Milvus, Qdrant) for production; ChromaDB + BM25 for prototyping; pgvector if PostgreSQL is already in place; OpenSearch/Elasticsearch with first-class RRF.

## Worked Example
Query "AAPL supply chain", k = 60, three documents (BM25 rank / semantic rank): A (exact ticker match) 1 / 15; B (supply chain discussion) 10 / 2; C (loose mention) 50 / 8.
- RRF(A) = 1/61 + 1/75 ≈ 0.0297
- RRF(B) = 1/70 + 1/62 ≈ 0.0304
- RRF(C) = 1/110 + 1/68 ≈ 0.0238
Decision: B ranks first; its consistent presence in both lists beats A's BM25 top-1 with weak semantics. Notebook results: Hybrid RRF MRR 0.443 vs BM25 0.349 (+27%) on a corpus of 200 passages and 15 queries; bge-large-en-v1.5 (1024-d) vs all-MiniLM-L6-v2 (384-d) MRR 0.675 vs 0.592, leading on technical queries (0.90 vs 0.73) and risk queries (0.61 vs 0.48), MiniLM wins only on general-business (0.56 vs 0.51). Structured parser: table preservation 0.45 to 0.85, layout fidelity 0.40 to 0.88, latency +70%. Security: simple defenses cut the unsafe-action rate from 0.50 to 0.00 and the unsupported-claim rate from 1.00 to 0.25.

## Key Takeaways
- Design ingestion first: structured parser, tables always, figures only if they carry quantitative data absent from the text; ingestion errors are irreversible.
- Tag every chunk with company, fiscal year/quarter, doc type, filing date, section, page, structural role; filter by publication date (point-in-time) and with pre-filtering.
- Select embeddings with 50-100 of your own queries, top-10 ground truth, Recall@k, MRR and latency; repeat when the corpus or distribution changes.
- Use hybrid (BM25 + semantic + RRF k=60), then a cross-encoder (top-50 to top-5) validated on your corpus.
- Enforce a restrictive prompt + programmatic validation (JSON schema, guardrails) + citation verification (supporting sentence, claim-citation similarity with a threshold, human review).
- Delegate arithmetic to code; measure unsupported claim rate in addition to RAGAs; calibrate the LLM-judge with a human-labeled subset.
- For production, version corpus + embedding + reranker + prompts + LLM together; incremental indexing, tombstoning, caching; involve compliance early.

## Connects To
Ch 10 (FinBERT, Loughran-McDonald, classification); Ch 4 (filing structure); Ch 23 (knowledge graphs, bipartite 13F, GNN); Ch 24 (agents, ReAct, memory); Ch 26 (governance, audit logging, retention); embeddings, point-in-time/leakage, LLM-as-judge.

## Code Reference

Official repository: [22_rag_financial_research/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/22_rag_financial_research). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_sec_filing_pipeline.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/01_sec_filing_pipeline.ipynb) + `.py` | 22.3 | - Compare two practical approaches to SEC filing ingestion for downstream RAG systems. |
| [`02_domain_embeddings_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/02_domain_embeddings_comparison.ipynb) + `.py` | 22.4 | This notebook compares embedding models for financial document retrieval: Uses synthetic data. |
| [`03_hybrid_retrieval.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/03_hybrid_retrieval.ipynb) + `.py` | 22.5 | This notebook demonstrates hybrid retrieval combining: Uses cross_encoder data. |
| [`04_ragas_evaluation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/04_ragas_evaluation.ipynb) + `.py` | 22.7 | This notebook implements a finance-oriented evaluation harness that... |
| [`05_10k_rag_assistant.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/05_10k_rag_assistant.ipynb) + `.py` | 22.6, 22.8 | This notebook demonstrates a production-grade RAG (Retrieval-Augmented Generation) system for analyzing SEC 10-K filings. |
| [`06_esg_rag_vs_finetune.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/06_esg_rag_vs_finetune.ipynb) + `.py` | 22.8 | This notebook compares two approaches to ESG (Environmental, Social, Governance) analysis: Uses finbert_pipeline data. |
| [`07_institutional_holdings_graph.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/07_institutional_holdings_graph.ipynb) + `.py` | 22.8 | Build a bipartite institution-stock graph from the 13F holdings artifact and derive co-ownership similarity, institutional momentum, and crowding signals for alpha resea… |
| [`08_rag_security.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/22_rag_financial_research/08_rag_security.ipynb) + `.py` | 22.7 | This notebook demonstrates attack and defense evaluation for document-grounded finance assistants, a critical concern when RAG systems operate on untrusted or adversaria… |
