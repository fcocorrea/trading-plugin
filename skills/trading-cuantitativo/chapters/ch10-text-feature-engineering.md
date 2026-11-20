# Chapter 10: Text Feature Engineering

## Core Idea
Turning text into a tradable feature depends less on the NLP model than on temporal discipline: when the document was available, whether the model was trained before the backtest period, and whether revisions contaminate the historical series. The progression of representations (lexical → static embeddings → sequential → transformers) adds context, but each layer demands a leakage-safe pipeline.

## Frameworks Introduced
- **Progression of representations**: lexical (term counts, no order) → static embeddings (one vector per word, semantic similarity) → sequential RNN/LSTM (memory, but a sequential bottleneck) → transformers (contextual embeddings with self-attention). *When*: choose based on the information each representation keeps/loses. *Failure mode*: using a complex encoder without comparing it against a strong TF-IDF; if the complex one does not win, the problem may be label quality, temporal alignment, or task definition.
- **Text-to-signal pipeline contract (7 points)**: (1) timestamp definition (publication vs scrape vs vendor: only one defines when you could have acted; index the backtest on it); (2) publication lag (measure it empirically; SEC by form type/filer status, news seconds-hours, transcripts same/next day); (3) entity resolution (ticker, CIK, FIGI; subsidiaries, M&A, ambiguous names); (4) deduplication (syndication/wires; TF-IDF similarity within (ticker, date)); (5) sampling unit (sentence/paragraph/document, with firm-day aggregation defined before modeling); (6) universe construction (endogenous coverage); (7) pre-training horizon (each model's cutoff precedes the backtest). *Failure mode*: a shortcut (approximate timestamp, unversioned embedding, unvalidated schema) only surfaces in live trading.
- **Four walk-forward safeguards for text**: *Purged CV* (exclude documents within k days of the boundary; k=1 for a 1-day signal, up to ~20 for monthly topic drift); *Rolling fine-tuning windows* (no test document in train windows; log document dates and model training dates); *Revision leakage* (snapshot at first availability, exclude revisions; amendment_leakage = "reject" by default); *Event-study alignment* (a document at t predicts [t+δ, t+δ+h]; δ = actual availability).
- **Pre-train → adapt → fine-tune cascade**: Stage 1 general (downloaded checkpoint), Stage 2 domain adaptation (continue MLM on 10-Ks, transcripts, reports, news; GPU hours, produces FinBERT), Stage 3 task fine-tuning (classification/regression/tagger head; thousands of examples, not millions). *Failure mode*: distribution shift between the fine-tuning and inference corpora.
- **Fine-tuning recipe**: Hugging Face Trainer; lr ≈ 2×10⁻⁵, 3-5 epochs, batch 16-32; head on [CLS]; mean pooling for retrieval/similarity; LoRA (updates a small fraction of weights, cuts memory by 60-80% with a modest loss).
- **Embedding-based factor construction**: aggregation unit = firm-day embedding (mean weighted by recency/source reliability) and baseline = rolling mean of the embedding over the previous 20 news-days. Three patterns: *Semantic novelty* (news surprise = cosine distance between the daily embedding and the 20-day baseline; measures topic change, not tone), *Peer-relative positioning* (distance to the peer centroid; cross-sectional dispersion as a second factor), *Narrative drift* (period-over-period changes, rolling volatility of the trajectory, similarity decay). Safeguards: do not fit global PCA/clustering on the whole corpus; version the model and preprocessing; cache by document hash + model id.
- **Extract-to-schema**: structured output with four field groups: *Entity* (CIK/FIGI/ticker), *Event* (controlled vocabulary: guidance, buyback, issuance, litigation, regulatory action, executive change), *Attribute* (direction, magnitude, units, horizon), *Uncertainty* (modality, confidence, abstention flag: no record if confidence is low). Example: "Board authorizes $10B share repurchase through 2027" → {event: buyback, magnitude: 10e9, currency: USD, horizon: 2027, confidence: 0.95}; buyback intensity = announced buyback / market cap over 90 days. Implement with fine-tuned encoders, an LLM with JSON mode, or a cascade (cheap encoder + LLM for high novelty/low confidence). Two safeguards: schema validation at the pipeline boundary; separate extraction (document level, deterministic model version, availability timestamp) from aggregation (firm-day, only available records).
- **Evaluating text signals**: IC/ICIR/t-stats/quintile spreads (Ch 7) + (a) decay analysis (IC at 1, 5, 20 days), (b) coverage-conditional (high vs low coverage), (c) event-time alignment (shift the timestamp by 30 min, 1 h, next-day open).
- **Interpretability with SHAP**: per-token attribution to audit risk (does it rely on boilerplate/tickers/artifacts?), debug IC degradation, and discover alpha (modals, hedging). It is a diagnostic, not a causal explanation; noisy on long documents.

## Key Concepts
- **Loughran-McDonald Master Dictionary**: financial lexicon with 6 lists (Negative, Positive, Uncertainty, Litigious, Strong modal, Weak modal); ~3/4 of the "negative" words in the Harvard General Inquirer are not negative in finance.
- **TF-IDF / BM25**: rarity weighting (BM25 adds length normalization and saturation).
- **Distributional hypothesis**: words in similar contexts have similar meanings (Firth 1957).
- **Word2Vec (Skip-Gram, CBOW) / GloVe**: static embeddings (100-300 dims) via local prediction vs global co-occurrence factorization.
- **Polysemy**: a word with several senses ("margin", "prime", "interest", "guidance") that static embeddings conflate.
- **Self-attention (Q, K, V)**: Attention = softmax(QKᵀ/√d_k)V; O(T²) cost in length; the weights are not faithful explanations.
- **Encoder-only / decoder-only / encoder-decoder**: BERT/FinBERT/DeBERTa/ModernBERT (bidirectional; embeddings and classification) vs GPT/LLaMA (causal; generation, prompt-based extraction) vs T5/BART.
- **FinBERT checkpoints**: ProsusAI/finbert (Reuters TRC2 + Financial PhraseBank) vs yiyanghkust/finbert-tone (tone of analyst reports); not interchangeable.
- **ModernBERT**: native 8,192-token context.
- **Bi-encoder vs cross-encoder**: independent embedding (linear, retrieval) vs joint processing of pairs (O(n²), reranking the top 50-100).
- **Coverage bias**: heavily covered firms differ systematically; a ticker with <21 days of coverage in the window is dropped.

## Mental Models
- Use **TF-IDF/lexicon as baseline and diagnostic** when the sample is small, the domain narrow, or the document formulaic.
- Use **encoder-only (FinBERT/ModernBERT) or distilled models** for high-volume daily features; **decoder-only LLM** for rare/ambiguous extraction or synthetic labeling, only if it adds value after latency/cost/validation.
- Use **mean pooling / Sentence-BERT** for similarity and retrieval; **[CLS]** for classification.
- Use **a domain checkpoint** only when its corpus and labels match the task; otherwise, a strong modern encoder fine-tuned.
- Use **chunking with overlap** (256-512 tokens, 64-token overlap) or hierarchical encoding for long documents, always within each filing (never across filings from different periods).
- Use **an LLM for synthetic labeling** of thousands of examples and then fine-tune a small encoder when you only have a few hundred labels.

## Anti-patterns
- **Using general-purpose dictionaries** (Harvard) in finance.
- **Scrape/vendor timestamp instead of actual availability**.
- **Pretrained model with a corpus that postdates the backtest period**; batch LDA over the whole corpus (use online LDA, Bybee et al. 2023).
- **Ignoring duplicates/syndication** (counts the same information several times).
- **Fine-tuned on one distribution and applied to another**: FinBERT 87.0% in-domain on PhraseBank but 49.4% on FinMarBa (-37.6 pts).
- **Truncating ~512-token documents without noticing**; numbers fragmented into subwords ($94.8B can be 5+ tokens): extract numbers separately.
- **Interpreting attention weights as explanation**.
- **Unversioned embeddings**: semantics change silently with the model.
- **Reporting only benchmark accuracy** instead of IC by horizon, coverage, and event-time alignment.
- **A signal whose IC collapses with a 1-hour delay**: it pays for speed, not information.

## Code Examples
```
{event: "buyback", magnitude: 10e9, currency: "USD", horizon: "2027", confidence: 0.95}
```
What it demonstrates: the target output of extract-to-schema, one validatable record per document. (The chapter references .py notebooks but includes no executable code.) Novelty formula: surprise_{i,t} = 1 − cos(e_{i,t}, mean(e_{i,t-1..t-20})).

## Reference Tables
Financial PhraseBank results (sentences_allagree, 2,264 sentences; 1,391 neutral/570 positive/303 negative; majority baseline 61.4%):

| Method | Accuracy | F1 macro |
|---|---|---|
| TF-IDF + Logistic Regression | 83.2% | 0.742 |
| GloVe + Logistic Regression | 80.6% | 0.709 |
| FinBERT-tone (zero-shot) | 93.2% | 0.917 |
| FinBERT (fine-tuned) | 97.1% | 0.955 |
| DeBERTa-v3 (fine-tuned) | 94.4% | 0.932 |
| ModernBERT (fine-tuned) | 96.5% | 0.962 |

When to fine-tune (Table 10.3): a few hundred examples → frozen embeddings/prompt labels/parameter-efficient; 1,000-10,000 → fine-tune with early stopping; domain vocabulary → domain adaptation or SFT; latency-critical → small model (FinBERT-base); nuanced reasoning → LLM + prompting.

Pooling: [CLS] (classification), mean (retrieval/similarity, uses the attention mask), max, attention pooling (requires training). Lengths: 512 tokens standard; ModernBERT/GTE-large/BGE-M3 8,192; transcripts 8,000-12,000 words; 10-K 50,000+.

Signal families: sentiment (commoditized, edge in speed/coverage), novelty/surprise, attention (coverage; predicts volatility even with neutral tone), events, entity relations (Ch 23), topic exposure (Bybee: 180 topics explain ~25% of aggregate return variance).

## Worked Example
Signals on 10-Qs for 50 symbols (cluster bootstrap by symbol, B=1,000):
- FinBERT sentiment on MD&A: 5d horizon IC=+0.055 (95% CI [-0.036,+0.160], p=0.264, n=720); 20d IC=-0.082 ([-0.195,+0.034], p=0.172, n=719): not significant.
- Narrative change (cosine distance between consecutive filings): 5d IC=-0.151 ([-0.226,-0.074], p<0.001, n=671, 47 symbols), 20d IC=+0.059 (p=0.154, n=670): only the narrative change signal is significant, and it does not reverse into a positive signal at the medium term.
Decision: the choice of horizon matters; the same source produces different signals depending on the horizon. Another SHAP example: "Net loss narrowed significantly from the prior year" → FinBERT positive with 69.7% confidence; "narrowed" contributes +0.373 and dominates over "loss". 13F asset embeddings (500 institutions, 1.32M holdings, 9,167 stocks): Apple closest to Microsoft (0.914), Amazon (0.913), NVIDIA (0.906); Hits@5 = 30.6% vs 0.055% random (229x lift, 95% CI 217-241x); attenuates to ~6% at positions 11-50 and <1% at 51-200.

## Key Takeaways
- Define the pipeline contract (7 points) before choosing a model; no result is trustworthy if you cannot answer them.
- Keep a TF-IDF/Loughran-McDonald baseline; if the transformer does not win, review labels and alignment.
- Check the training cutoff, version embeddings and models, and use online (not batch) topic models.
- Build narrative factors (novelty, peer-relative, drift) with a rolling 20-news-day baseline and handle coverage explicitly.
- Evaluate by decay (1/5/20 days), coverage, and realistic delay; use SHAP to audit.
- Use extract-to-schema with validation and separate extraction from aggregation.

## Connects To
Ch 4 (data, entity mapping, vintages), Ch 5 (synthetic data, SV), Ch 6 (timestamp, admissible information), Ch 7 (IC/ICIR, triage), Ch 9 (fitted-object discipline), Ch 11 (pipeline), Ch 12 (SHAP), Ch 13 (transformers), Ch 22-24 (RAG, knowledge graphs, LLM agents), Ch 23 (graphs).

## Code Reference

Official repository: [10_text_feature_engineering/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/10_text_feature_engineering). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

This chapter has no notebooks; only `README.md` in the folder.

Other notebooks in the folder, not linked from the README: `01_word2vec_training.ipynb`, `02_asset_embeddings.ipynb`, `03_sentiment_evolution.ipynb`, `04_bert_finetuning.ipynb`, `05_financial_ner_finetuning.ipynb`, `06_finbert_cross_dataset.ipynb`, `07_news_return_signals.ipynb`, `08_text_feature_evaluation.ipynb`, `09_filing_text_signals.ipynb`.
