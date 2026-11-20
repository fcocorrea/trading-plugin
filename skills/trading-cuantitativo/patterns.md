# Patterns and techniques — ML4T 3rd Ed. (Jansen)

Grouped by part of the book. Each entry: When to use / How / Trade-offs (Ch N).

# Part I — Data (Ch 2-5)

## Evidence boundary (exploration vs confirmation)
**When to use**: any iterative research over features, models or parameters.
**How**: log every trial in a research ledger; reserve a sealed holdout; freeze spec and metrics; evaluate once with selection-adjusted inference.
**Trade-offs**: less data for development and more discipline; in exchange, the evidence is credible. (Ch 1, Ch 6)

## Bitemporal as-of query with restricted joins
**When to use**: any fundamental, macro or alternative data subject to revisions.
**How**: store valid time and `available_at` with `source_id`, `entity_id`, version; join only on `available_at <= decision_time` and mappings with `effective_date <= t < end_date`; macro via ALFRED/vintages; XBRL reconstructed from filings, not Frames.
**Trade-offs**: more storage and complexity; eliminates silent lookahead. (Ch 2, Ch 4)

## Point-in-time anti-survivorship universe
**When to use**: cross-sectional backtests or backtests on historical indices.
**How**: reconstruct membership by date, include delistings with correct terminal returns, CRSP-type sources.
**Trade-offs**: more expensive data; the corrected bias was +3.5-15.3 pp in 2014-2018. (Ch 2)

## Storage default: partitioned Parquet + DuckDB/Polars
**When to use**: research up to ~100M rows on one machine.
**How**: partitioned Parquet for raw/curated; DuckDB for SQL and ASOF; Polars for features; a server (Postgres, kdb+, QuestDB) only for concurrency/governance; validate with Great Expectations/Pandera (fail hard).
**Trade-offs**: no multi-writer ACID; move to a lakehouse (Delta/Iceberg/Hudi) if there are concurrent updates. (Ch 2)

## LOB reconstruction as a state machine
**When to use**: microstructure, OFI, impact and execution models.
**How**: parse → normalize → apply → invariants → snapshot → validate against BBO/NBBO; order registry and price levels.
**Trade-offs**: expensive (13 GB/day; Python ~23 min, Rust <5 min); only sees displayed liquidity from one venue. (Ch 3)

## Bar selection
**When to use**: defining the sampling unit for features and labels.
**How**: dollar bars by default; time bars if the horizon is clock-based; imbalance/run bars only if order flow is the object and aggressor side or Lee-Ready is available; α~0.001, warmup 100+ bars, monitor expected_imbalance.
**Trade-offs**: information bars require classification, initialization and watching for threshold spiral. (Ch 3)

## Jump detection and label conditioning
**When to use**: cleaning intraday labels and building volatility features.
**How**: RV − BV ≥ 0 per day; Lee-Mykland with K=12 on 5-min bars, α=0.01; features jump_count, jump_variance, jump_share; skip 30 min after a jump.
**Trade-offs**: the first K bars cannot be tested; sensitive to K and frequency. (Ch 3)

## Hierarchical entity resolution
**When to use**: combining sources with different identifiers.
**How**: deterministic → fuzzy (rapidfuzz) → manual review → entity/security/contract master with temporal mappings and an audit trail; monthly QA on a 1-5% sample.
**Trade-offs**: precision over recall; human review is costly. (Ch 4)

## Gated alt data evaluation
**When to use**: before buying or building an alternative dataset.
**How**: incremental uniqueness, half-life, quality/coverage/latency, legal and PIT as hard gates, TCO and operationalization test; compare only those that pass.
**Trade-offs**: kills ideas early; avoids legal and operational debt. (Ch 4)

## PIT text pipeline
**When to use**: NLP features from filings.
**How**: primary document, section extraction with checks, cleaning that preserves paragraphs, PK cik+accession+section with `accepted_at`.
**Trade-offs**: removing boilerplate favors novelty; keeping it favors level sentiment. (Ch 4)

## FUP validation of synthetic data + classical baselines first
**When to use**: any use of synthetic data.
**How**: fidelity (KS, correlation, ACF of r², tails), utility (TSTR/TRTR), privacy (duplicates, NN, membership inference, DP) and a real holdout; before a deep generator, stationary bootstrap (~22-day block) and GARCH(1,1); the deep model must beat them.
**Trade-offs**: the three criteria compete; bootstrap does not generate events outside the record. (Ch 5)

## Regime-conditional generation
**When to use**: regime-based stress testing.
**How**: HMM for labeling, classifier on noisy sequences, log-prob gradient during denoising, temperature/DDIM for diversity.
**Trade-offs**: inherits the detector's errors; strong guidance exaggerates extremes. (Ch 5)

# Part II — Research and features (Ch 6-10)

## Versioned setup (mechanics vs parameter)
**When to use**: starting or modifying a research line.
**How**: `setup.yaml` with universe, schedule, mapping, constraints and costs; parameters keep the version, mechanics bump it and require a new baseline.
**Trade-offs**: minimal bureaucracy in exchange for comparability. (Ch 6)

## Walk-forward with buffers + sealed holdout
**When to use**: evaluating any pipeline on time series.
**How**: fixed windows; label buffer = horizon (trading days); embargo (~1%) if there is training after the test period; holdout once; nested walk-forward or CPCV for variants.
**Trade-offs**: high compute, less effective data. (Ch 6)

## Baseline checkpoint and run logging
**When to use**: before large experiment grids.
**How**: sanity checks on timing, coverage and trading intensity; one label, compact families, one simple model, one walk-forward design; log provenance, data/eval, config, artifacts, decision.
**Trade-offs**: narrow first run; the search stays accountable (the basis for the Deflated Sharpe). (Ch 6)

## Split-aware preprocessing with auditable masks
**When to use**: any pipeline that estimates means, standard deviations, quantiles or imputations.
**How**: classify steps as fitted vs stateless; refit fitted steps at each walk-forward boundary using only the past; one pipeline object for training and inference; store the mask and effective N; winsorize/robust-scale instead of truncating.
**Trade-offs**: more compute and bookkeeping (demo: test mean 0.110 train-only vs 0.101 full-panel). (Ch 7)

## Execution-consistent label with overlap audit
**When to use**: defining any forward return or event label.
**How**: fix the convention (next-open-to-open or close-to-close) and repeat it in the PnL; vol-scaled threshold or cross-sectional percentile; triple-barrier with widths from MFE/MAE and a rule for both barriers hit in one bar; uniqueness and N_eff; purge ≥ H_max, sample weights or subsampling; audit record (anchor, horizon, resolution, overlap, base rate).
**Trade-offs**: narrow barriers = noise; wide ones = more overlap and fewer observations. (Ch 7)

## Univariate Proceed/Revise/Stop triage with accounted search
**When to use**: filtering many candidate features before modeling.
**How**: correctness (coverage, timing, mask, staleness); per-fold association (median, IQR, worst fold, sign consistency); shape by decile; feasibility (turnover, spread ≥ 2x costs, capacity); record the searched set and correct with Holm or BH; exploration by stability, then frozen confirmation.
**Trade-offs**: does not see duplicates or conditional value; FWER is conservative, FDR permissive. (Ch 7)

## Minimal DAG + plausibility checks
**When to use**: every candidate that survives triage, before complex modeling.
**How**: 3-5 nodes; identify confounder/mediator/collider; timing placebo, shared-driver, regime heterogeneity, event-time; decide PROCEED/REVISE/STOP.
**Trade-offs**: only rejects mechanisms, does not confirm them; REVISE is the usual outcome. (Ch 7)

## Three-step filter and feature spec
**When to use**: turning a hypothesis into a feature.
**How**: align lookback/cadence to the label horizon and execution lag; name the driver (persistence, reversion, risk compensation, predictable clock); signal or state role; fill in the spec; record it as a trial.
**Trade-offs**: friction per feature; prevents indicator sprawl and patterns without a mechanism. (Ch 8)

## Signal × State (gating, scaling, conditional variant)
**When to use**: a relevant state (vol, liquidity, cross-asset correlation) modulates the signal.
**How**: one signal and one state; one template at a time; terciles with expanding-window percentiles; IC per tercile; verify structure, per-fold stability and real-time robustness.
**Trade-offs**: multiplies the searched set (5×3×3 = 45); gating changes turnover and capacity. (Ch 8)

## Feature dedup by cluster
**When to use**: a search with many nearly equivalent variants.
**How**: per-fold rank correlation on train; hierarchical clustering on 1−|ρ|, cut at |ρ| 0.6-0.8; one representative per cluster (best median IC with low dispersion, or the simplest).
**Trade-offs**: reduces redundancy but does not replace model selection; MDI and permutation importance agree only 49% of the time. (Ch 8)

## Fitted-object discipline
**When to use**: any feature from a fitted model (Kalman, GARCH, HMM, FFT, embeddings).
**How**: fit only on the fold's training window; use filtered outputs; choose order/d/states within training; fix a refit cadence and update online between refits; version parameters and window.
**Trade-offs**: operational cost and complexity; without it there is silent look-ahead. (Ch 9)

## Volatility models as feature extractors
**When to use**: returns that are barely predictable but volatility that clusters.
**How**: ARIMA residual as prewhitening; GARCH → cond_vol, α, β, half-life, long_run_vol (only if α+β<1); EGARCH/GJR → leverage; HAR → β per horizon; rolling Hurst over 252 days.
**Trade-offs**: refit weekly to monthly (GARCH), rolling HAR, SV with MCMC monthly or less often. (Ch 9)

## Regimes as continuous conditioning
**When to use**: other signals change across market environments.
**How**: baseline with observable rules (VIX threshold, price vs MA); then a 2-state HMM with filtered probabilities, entropy and expected duration; relabel by variance; pass them as inputs to a single model. As a risk lens: cluster (GMM, low K), measure vol/Sharpe/drawdown by regime and map to sizing/caps.
**Trade-offs**: per-regime models only with an economic rationale and enough data; hard switches are fragile; 267 transitions in 98 years make timing noisy. (Ch 1, Ch 9)

## Panel features: temporal order → align → cross-sectional
**When to use**: taking per-asset features into multi-asset tasks.
**How**: feature with trailing history per asset; align on decision date; rank/z-score, benchmark adjustment, pairwise (Kalman hedge ratio, spread z-score) or universe-level aggregates.
**Trade-offs**: reversing the order leaks information; changing universe composition moves ranks. (Ch 9)

## Text-to-signal pipeline contract
**When to use**: before choosing any text model.
**How**: seven questions (timestamp, lag, entity resolution, dedup, sampling unit, universe, pre-training horizon); purged CV according to decay; snapshot without revisions; event-time with actual availability.
**Trade-offs**: more discipline; no notebook result is trustworthy without it. (Ch 10)

## Pre-train → adapt → fine-tune
**When to use**: financial sentiment/classification/extraction with a few thousand labels.
**How**: checkpoint; MLM on a financial corpus if there is a vocabulary gap; fine-tune lr≈2e-5, 3-5 epochs, batch 16-32 (LoRA if memory is short); validate out of distribution.
**Trade-offs**: FinBERT 97.1% in-domain but 49.4% on FinMarBa; with a few hundred labels, frozen embeddings or synthetic labeling with an LLM. (Ch 10)

## Narrative factors from embeddings
**When to use**: capturing topic shifts invisible to sentiment.
**How**: embedding per document → firm-day (weighted mean) → 20 news-day baseline; novelty (cosine), peer-relative, narrative drift; version the model and cache by hash.
**Trade-offs**: bias toward high-coverage firms; do not fit global PCA/clustering without walk-forward. (Ch 10)

## Extract-to-schema with extraction/aggregation separation
**When to use**: events with magnitude (buybacks, guidance, litigation).
**How**: Entity/Event/Attribute/Uncertainty schema; extraction by encoder or LLM with guided decoding and abstention; validate at the boundary; aggregate by firm-day.
**Trade-offs**: valid format does not guarantee factual accuracy; allows re-extraction without rebuilding history. (Ch 10)

# Part III — Models (Ch 11-15)

## Regularized linear baseline ladder
**When to use**: first ML model on any feature set; the floor for Ch 12-14.
**How**: standardize train-only with winsorization; sweep λ from 10⁻² to 10⁹ with large n; Ridge by default with correlated features; evaluate IC, ICIR, turnover and spreads; uniqueness × recency weights and N_eff.
**Trade-offs**: regularization gives numerical stability, not signal (6 of 9 case studies with no significant IC). (Ch 11)

## Nested walk-forward HPO
**When to use**: tuning hyperparameters with a weak signal.
**How**: inner Optuna within each training window, evaluation in the outer loop; report trials, search space and per-fold dispersion; distrust a best trial that depends on a single fold; 50-200 trials, pruning, multi-objective IC vs turnover (NSGA-II).
**Trade-offs**: more compute; single-loop can flip the sign of the IC. (Ch 11, Ch 12)

## Classification with calibration and mapping to positions
**When to use**: long/short/flat decisions with a threshold or quantile.
**How**: regularized logistic (C = 1/λ); class_weight='balanced' if imbalanced; Platt/isotonic on validation; signal by threshold (0.6/0.4), probability-weighted or rank-based.
**Trade-offs**: regularization compresses probabilities; rank-based is robust to miscalibration. (Ch 11)

## Conformal layer (split, ACI, CQR)
**When to use**: moving from a point forecast to a calibrated interval and sizing by uncertainty.
**How**: proper-train/calibration split (80/20); score |residual|; ACI with γ ∈ [0.005, 0.05]; CQR if dispersion is heterogeneous; sets with a singleton-or-abstain rule; monitor coverage by fold, regime, magnitude and asset.
**Trade-offs**: split is auditable but fixed-width; CQR is adaptive but depends on the quantile model; ACI covers on average, not per regime. (Ch 11, Ch 13)

## Auditing and monitoring with SHAP
**When to use**: pruning, auditing and drift for any model.
**How**: LinearSHAP/TreeSHAP per fold; mean |SHAP| ≈ 0 across all folds = candidate for removal; high variance across folds = regime-dependent signal; waterfall on high-conviction predictions; compare with PFI/MDI and across families (GBM vs TabM).
**Trade-offs**: descriptive attribution, not causal; correlation creates impossible coalitions; Rashomon limits confidence in any single model. (Ch 11, Ch 12)

## Choosing a GBM library by operational constraint
**When to use**: financial tabular data with a defined compute budget and categoricals.
**How**: LightGBM by default on CPU with massive walk-forward (~50s vs XGBoost ~100s vs CatBoost ~135s on 4.9M rows); CatBoost for high-cardinality categoricals, GPU (~14x) or inference latency; XGBoost with CUDA GPU and the SHAP/Optuna ecosystem; small datasets on CPU (CUDA FP64 is counterproductive).
**Trade-offs**: after tuning the gap is ~0.002 IC; hyperparameters matter more than the library. (Ch 12)

## Initial GBM config
**When to use**: getting trees started for the cross-section of returns.
**How**: shallow trees, MAE loss (8 of 9 case studies), learning rate 0.01-0.05, early stopping as a tuning outcome (~50-350 trees), emphasis on reg_alpha/reg_lambda, monotonic constraints if there is an economic sign, evaluate finalists on untouched data.
**Trade-offs**: more trials = more validation overfitting; depth rarely changes an IC that crosses zero. (Ch 12)

## Learning to Rank for tail strategies
**When to use**: long-short on the extreme decile/quintile of a broad universe (e.g. 500 stocks, 50 per tail).
**How**: lambdarank with group by date, label_gain by intra-date quintile, NDCG@10; compare vs MSE regression by NDCG and IC.
**Trade-offs**: scores are not calibrated to magnitudes; the advantage dilutes with full cross-section portfolios. (Ch 12)

## Baseline ladder before time-series DL
**When to use**: before any sequence architecture.
**How**: seasonal naive → N-Linear/D-Linear → ARIMA/ETS/VAR → LightGBM with lags; move up only if temporal order adds robust OOS signal (<5K Ridge/LightGBM; 5-20K LightGBM; >20K try DL).
**Trade-offs**: across 8 case studies DL credibly beat the best tabular model in 1 (Crypto), lost in 2, and tied in 5 at higher cost. (Ch 13)

## Sequence target: horizon-label vs path
**When to use**: choosing the target of a sequence model.
**How**: horizon-label (LSTM/NLinear) for cross-sectional rank IC; path forecasting (N-BEATS, TSMixer, PatchTST) if the trajectory matters; direct allocation (Ch 17) for economic objectives.
**Trade-offs**: the compounded path accumulates error (TSMixer ETFs 21d IC +0.029 < NLinear/Ridge). (Ch 13)

## Calibrated uncertainty for sizing
**When to use**: forecasts that scale positions or trigger no-trade.
**How**: ensemble of ~5 models (or MC dropout with 50-100 passes) + split-conformal; validate empirical coverage, width, and that more uncertainty = more OOS error.
**Trade-offs**: ensembles are robust but cost scales linearly; without calibration, nominal 95% coverage ends up under-covered. (Ch 13)

## Staged adaptation of foundation models
**When to use**: considering a TSFM after beating the baselines.
**How**: zero-shot → ICL → PEFT/LoRA → domain training; fine-tuned for vol/VaR; verify no pretrain/test overlap.
**Trade-offs**: zero-shot on returns gives negative OOS R² and IC; leakage through pretraining. (Ch 13)

## Choice of latent estimator + three-stage adapter
**When to use**: deciding among PCA, RP-PCA, IPCA, CAE, SDF, SAE and turning them into forecasts.
**How**: PCA = risk; RP-PCA = priced factors; IPCA = linear conditional betas; CAE = nonlinear; SDF = direct pricing; SAE = predictive signal; estimate several in the same walk-forward (agreement = signal, divergence = model risk). Adapter: estimate factors → premium forecaster → μ = β_t'λ_hat.
**Trade-offs**: PCA led in 0 of 5 case studies, SDF/SAE in 4 of 5; the reconstructive CAE gave IC −0.030. (Ch 14)

## Two-stage production PCA
**When to use**: covariance for risk and portfolio construction.
**How**: Stage 1 half-life ~20 days for idiosyncratic vol; Stage 2 normalize by that vol, half-life ~120 days, PCA + BBP shrinkage + Procrustes; rebuild Ω at the original scale.
**Trade-offs**: separates fast vol from slow correlation; more complexity. (Ch 14)

## DML with refutation
**When to use**: effect of a continuous treatment (e.g. momentum) with nonlinear confounders.
**How**: DAG → pre-treatment adjustment set → estimand → DML with temporal cross-fitting and ≥2 classes of learners → naive vs DML → placebo/block-permutation, sensitivity, stability.
**Trade-offs**: weak nuisance models = fragile estimate; refutation does not prove causality. (Ch 15)

## BSTS for discrete events
**When to use**: impact of FOMC, regulation, CEO departure.
**How**: learn target-control in the pre-period, project the counterfactual, take the difference; spillover (control-as-target), placebo dates, alternative controls.
**Trade-offs**: in macro there are rarely unaffected controls; with 58% of placebos significant, the design is rejected. (Ch 15)

## Causal discovery as a hypothesis generator
**When to use**: unknown causal structure.
**How**: Granger (screening) + PCMCI + VAR-LiNGAM + NOTEARS; prioritize edges confirmed by several methods; bootstrap stability (50% threshold).
**Trade-offs**: 1 to 42 edges on the same data depending on the method. (Ch 15)

# Part IV — Implementation (Ch 16-20)

## Explicit backtest protocol
**When to use**: every backtest, before interpreting results.
**How**: fix signal timing (close → next open), score-to-signal, rebalance, sizing, fills, constraints, costs; vectorized or event-driven depending on whether there is state; document the engine semantics.
**Trade-offs**: more realism = more assumptions and bookkeeping. (Ch 16)

## Non-ML baseline + layered economic diagnostics
**When to use**: before accepting any ML strategy.
**How**: auditable baseline (risk-adjusted 6m momentum + 10Y-2Y>0.5% filter, top 3 or 60% AGG/40% TLT, 5 bp per side); report gross/net, turnover, break-even cost, cost sweep, comparison against a simple baseline, and pre-specified vol × trend regimes.
**Trade-offs**: the book's baseline came in below 60/40 (Sharpe 0.65 vs 0.80). (Ch 16)

## Search-adjusted Sharpe inference
**When to use**: before promoting a Sharpe selected from a sweep.
**How**: CI with SE (~1/√T), autocorrelation adjustment (Lo) or HAC/bootstrap, minimum track record, DSR with effective trials, Reality Check/SPA, FDR/FWER, Rademacher Anti-Serum, trial accounting, exploration separated from confirmation.
**Trade-offs**: counting trials as independent over-penalizes; ignoring them under-penalizes. (Ch 16)

## Allocator ladder + term sheet + FLAM
**When to use**: choosing an allocator after fixing signal, costs and constraints.
**How**: document the term sheet; baseline equal weight + inverse vol + vol targeting; score-weight if magnitude is informative; MVO only with Ledoit-Wolf, caps and a turnover penalty; HRP/NCO if covariance is the bottleneck; same forecasts, costs, constraints and window for all; IR = IC × √(effective breadth) as a sanity check.
**Trade-offs**: more information = more sensitivity to error; Sharpe spreads across allocators range from 0.1 (ETF) to 0.7 (crypto); FLAM assumes unbiased, independent forecasts. (Ch 17, Ch 20)

## End-to-end allocator (vol-targeted, pooled-Sharpe)
**When to use**: the objective does not factor into forecast + optimization.
**How**: shared encoder → tanh → vol-scaling → pooled Sharpe with turnover cost; SoftMin over sub-windows; two-pass for minibatches.
**Trade-offs**: aligns the loss with the portfolio but loses diagnosability, seed-sensitive; must beat heuristics after costs. (Ch 17)

## Cost model ladder by participation
**When to use**: backtests and universe design.
**How**: <0.5% ADV spread; 0.5-2% linear; >2% sqrt; without execution data use the quoted spread + wide slippage in vol + η in the upper half; recalibrate with TCA.
**Trade-offs**: conservative rejects viable ideas; optimistic approves false positives. (Ch 18)

## Cost mitigation cascade (options)
**When to use**: instruments where cost scales with the premium rather than the notional.
**How**: naive round-trip → hold-to-expiry (eliminates the exit spread) → liquidity filter (bottom quintile of relative half-spread).
**Trade-offs**: a very thin universe concentrates the top-5; holdout intervals still cross zero. (Ch 18, Ch 20)

## TCA cycle
**When to use**: production and recalibration.
**How**: estimate → trade → measure IS → decompose (spread, impact, timing, fees) → classify market-driven vs execution-driven with a regime tag → update parameters.
**Trade-offs**: requires fill data; without regime tags, churn gets mistaken for evidence. (Ch 18)

## Risk term sheet with graduated kill switches
**When to use**: before deploying any strategy.
**How**: per trigger: metric, threshold, action, escalation and reinstatement; levels 5/10/15/20/30% DD; context (VIX, cross-asset confirmation); regime-conditional VaR/CVaR with liquidity haircuts.
**Trade-offs**: rigid rules can lock in losses; calibrate to the strategy's vol. (Ch 19)

## Cyclical stress testing
**When to use**: quarterly and when exposures change.
**How**: historical replay + cost stress + combined matrix + factor stress + reverse stress; one-page memo with changes to limits, hedges or monitoring.
**Trade-offs**: without a resulting action it is wasted effort. (Ch 19)

## Leakage-free adaptive controls
**When to use**: vol targeting, per-regime caps, stops, pauses.
**How**: only information at t−1; GARCH+EWMA, one-step lag, cap on the scaling factor, predefined thresholds, walk-forward with nested tuning; traceability of every position change.
**Trade-offs**: regimes from hidden-state models can leak the future; tuning on the evaluation window overfits. (Ch 19)

## Four-stage strategy diagnosis
**When to use**: after a disappointing holdout or when evaluating a case study.
**How**: signal (IC, ICIR, checkpoints) → allocator (best-worst spread with a fixed signal) → costs (breakeven, capacity) → overlays and stability; identify the drift mechanism before re-running; top-3 ensemble (cross-sectional z-score per timestamp) if the top cluster falls within the bootstrap.
**Trade-offs**: n=9 heterogeneous observations; heuristic; the ensemble gives up peak Sharpe for stability. (Ch 20)

# Part V — Advanced AI (Ch 21-24)

## Execution / Market-making / Hedging as an MDP
**When to use**: the action is a target with a measurable reward.
**How**: state (LOB + inventory + regime), continuous action with clipping/constraints, reward (IS, PnL − λ·inv², −CVaR), γ 0.99-0.999; PPO/SAC; analytical benchmark.
**Trade-offs**: sensitive to simulator, reward and seed; no proven dominance. (Ch 21)

## Sim-to-real pipeline
**When to use**: moving from simulation to real trading.
**How**: simulator with impact/tails/latency (ABIDES) → domain randomization → OPE → paper → limited capital → hard-coded limits and kill switch → retraining and versioning.
**Trade-offs**: reduces but does not eliminate the gap; offline RL suffers from distribution shift. (Ch 21)

## Hybrid retrieval with RRF and reranking
**When to use**: queries over filings that mix exact entities (tickers) and concepts.
**How**: BM25 and semantic search in parallel; RRF (k=60); top-50 to a cross-encoder; top-5 to the LLM; pre-filter by metadata (company, fiscal year, doc type, filing date).
**Trade-offs**: +27% MRR over BM25; a generic cross-encoder degraded MRR by 36%, so validate on your own corpus; reranking adds 100-200 ms. (Ch 22)

## Retrieve-extract-compute-narrate
**When to use**: numerical questions over tables (margins, ratios, YoY).
**How**: retrieve tables; extract to a schema (value, unit, period, segment); compute in code; narrate with citations and a trace.
**Trade-offs**: eliminates arithmetic failures that faithfulness cannot see; adds extraction and validation. (Ch 22)

## RAG diagnosis by failure mode
**When to use**: the RAG answers poorly and it is unclear what to fix.
**How**: eval set of 50-100 queries (10-20% unanswerable); RAGAs; low recall = retrieval, low precision = ranking, low faithfulness = synthesis; fix only the weak component; unsupported claim rate and adversarial safety.
**Trade-offs**: LLM-as-judge requires human calibration; curating ground truth is costly. (Ch 22)

## Governed KG extraction + Graph RAG with safe text-to-Cypher
**When to use**: filings → auditable graph (supply chain, holdings, 8-K); multi-hop or co-ownership questions.
**How**: identity/schema/provenance contracts; slicing (Items 1, 1A, 7); canonicalization; validation; human review queue; idempotent writes; gold subset. Querying: graph/document/hybrid routing; parameterized templates; read-only, allowlist, limits; dual citation (row + text).
**Trade-offs**: 10-K is high-precision/low-recall; recall 1.00 vs 0.13 and ~82 vs 267 tokens on holdings; higher upfront cost than vector RAG. (Ch 23)

## Leakage-safe temporal graph
**When to use**: graph features for backtests.
**How**: three timestamps per edge; filter by disclosure time before aggregating; splits by disclosure time with embargo; log snapshot hash and extractor version; compare against tabular and static-graph baselines.
**Trade-offs**: more evaluation infrastructure; reusable across families. (Ch 23)

## Research agent (bounded ReAct)
**When to use**: evidence-rich, read-only forecasting.
**How**: one tool (search) with the cutoff at the boundary; 2-action schema (search/forecast); step limit 5; quality gates; forced default; serializable trace; accept after stability, groundedness and a low forced-default rate.
**Trade-offs**: less flexible than multi-tool but isolates reasoning quality. (Ch 24)

## AIA-style multi-agent forecaster
**When to use**: calibrated probabilities on events with contested evidence.
**How**: intake, parallel agents, aggregation (mean/median/trimmed/Neyman), optional debate, supervisor with override only at high confidence, Platt on a disjoint window, scoring.
**Trade-offs**: debate and roles add cost without value on one-directional questions; extremization + Platt can overfit confidence. (Ch 24)

## Replay with frozen evidence
**When to use**: debugging nondeterministic systems.
**How**: checkpoint at decision boundaries; freeze tool outputs; rerun with the new prompt/policy; compare trace and eval deltas.
**Trade-offs**: all artifacts must be persisted; avoids mistaking evidence drift for prompt improvement. (Ch 24)

# Part VI — Production (Ch 25-27)

## Unified dual-mode strategy
**When to use**: taking any validated strategy live.
**How**: abstract data/execution interfaces, deterministic signals, shared event-driven core, persistent state; LiveEngine with broker adapters and SafeBroker around the same Strategy.
**Trade-offs**: safety and connectivity must be built; eliminates the class of bugs caused by rewrites. (Ch 25)

## Stage-by-stage parity verification
**When to use**: before committing capital and as continuous regression.
**How**: same inputs to both systems; compare features, predictions, signals and orders; per-commit tests plus a morning check; on mismatch, halt and bisect by stage.
**Trade-offs**: test infrastructure and ground truth; EXPECTED_DIFFERENCE for warm-up. (Ch 25)

## Order lifecycle with idempotency
**When to use**: any system with asynchronous orders.
**How**: state machine with PENDING_CANCEL; persisted client_order_id; recovery by querying the broker; daily and startup reconciliation; kill switch persisted in a state file.
**Trade-offs**: more state to maintain; zero duplicates after a crash. (Ch 25)

## Layered drift detection
**When to use**: diagnosing what changed when performance drops.
**How**: data gates; daily PSI and K-S, weekly SHAP, continuous ADWIN/DDM; cross-check with performance (drift × decay matrix); monitor the detector's health; 4-way monitoring (data, signal, execution, risk) with predefined retrain/pause/retire actions.
**Trade-offs**: false positives and alert fatigue; PSI requires binning. (Ch 1, Ch 26)

## Safe model rollout
**When to use**: replacing a decayed model.
**How**: shadow ~60 days; test with a minimum effect size; A/B with 5-10% of capital; steps of 10-25-50-100% over 60-90 days; explicit criteria; archived and tested rollback.
**Trade-offs**: slow, but avoids trading one failure for another. (Ch 26)

## Circuit breakers in depth
**When to use**: protecting capital against sudden failures.
**How**: 4 independent levels; loss, position and anomaly breakers; software breaker CLOSED/OPEN/HALF_OPEN; recovery protocol; override with logging and expiry.
**Trade-offs**: a fixed drawdown is crude (combine with exposure and stress); false positives require disciplined overrides. (Ch 26)

## Systematic career plan
**When to use**: designing quant professional development.
**How**: skills assessment against the 5 roles; a daily 30-minute habit; apply each technique to a project; knowledge base including failures; visibility (open source, preprints, competitions).
**Trade-offs**: requires consistency; advanced degrees only for top-tier research. (Ch 27)
