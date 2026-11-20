# Judgment Cheatsheet — ML4T 3rd Ed. (Jansen)

Decision layer: what the author would do and why. Definitions in glossary.md; techniques in patterns.md.

## 1. Decision rules

**Process discipline and multiple testing**
- When iterating over features/models, log every trial and seal a holdout; use it once with the spec frozen, because the process is the edge (Ch 1, Ch 6)
- When you pick a parameter after seeing results, count it as a new trial, because it is the most common accounting failure (Ch 7)
- When selecting the best of N, adjust with DSR/Reality Check; with 100 no-skill candidates the best Sharpe reached 2.54, because the maximum of noise grows with N (Ch 16, Ch 5)
- When testing many candidates, use Holm if you will promote few and BH for a large screen; t>3 for new factors, because 150 bundles yield ~7-8 false positives at 5% (Ch 7)
- When success gets redefined after seeing results (goalpost drift), pre-register thesis, signal and criteria, because otherwise it is rationalization (Ch 1, Ch 6)
- When an idea is a published anomaly, assume ~50-58% decay and an implementation gap, because it gets arbitraged away after publication (Ch 4, Ch 6)
- When you cannot name a plausible mechanism, treat the pattern as provisional with the holdout untouched, because there is no guide to failure or regime (Ch 8)
- When a result comes from a single split or seed, do not trust it: TCN went from IC +0.051 to −0.013 between identical reruns (Ch 13)

**Temporal validation and leakage**
- When evaluating time series, use walk-forward with label buffer = horizon in trading days; never random k-fold, because labels overlap (Ch 6)
- When there is training after the validation period (k-fold, combinatorial), add a feature buffer; standard walk-forward does not need one (Ch 6)
- When a transform estimates parameters, refit it in each fold using only the past, because fitting on the full panel leaks the future (Ch 7)
- When querying fundamentals/macro, use as-of with `available_at <= t` and vintages, because the restated value did not exist at decision time (Ch 2, Ch 4)
- When building the universe, use historical membership and delistings; never "current constituents", because it injects survivorship (Ch 2)
- When you suspect leakage in fundamentals, run an as-known view and a "latest today" view; if the second wins, there is later information (Ch 4)
- When using 13F, the info exists from the filing date (45-day lag), not from quarter end (Ch 23)
- When the LLM could have seen the backtest period, reject the result or use a model with an earlier cutoff, because there is look-ahead (Ch 10, Ch 24)
- When using HMMs, regimes or wavelets, use filtered probabilities/trailing windows, because smoothed probabilities and offline coefficients use the future (Ch 9)
- When a number comes from a regime labeled ex post, do not use it in trading rules (Ch 1)

**Signal evaluation**
- When evaluating IC, summarize per fold (median, IQR, worst fold, sign consistency) with N_eff/HAC, because pooling hides regimes (0.001 pooled vs 0.064 per fold) (Ch 7)
- When OOS AUC >0.65 in finance, suspect leakage; 0.55-0.60 is already significant (Ch 11)
- When the gross spread does not exceed 2x costs, treat it as caution/stop (Ch 7)
- When a new model does not beat Ridge or D-Linear/N-Linear, discard it, because the complexity adds nothing (Ch 11, Ch 13)
- When IC improves but turnover rises, evaluate net of costs (Ch 11)
- When Sharpe is 1.0 over 5 years of daily data, remember the 95% CI is ~[0.12, 1.88]: it is not precise evidence (Ch 16)
- When you have done HPO in low-SNR, cap it at 50-200 trials and reserve untouched data, because 500 trials exploit validation noise (Ch 12)

**Models**
- When you need a baseline tabular model, shallow LightGBM + MAE + early stopping (8 of 9 case studies); apply monotonic constraints if the sign is known (Ch 12)
- When using importances, use SHAP, not gain or split-count, because those are biased and unstable across seeds (Ch 12)
- When features are correlated by construction, prefer Ridge/Elastic Net over LASSO, because LASSO picks arbitrarily (Ch 11)
- When evaluating by cross-sectional IC, use a direct horizon label rather than path forecasting, because it avoids compounding (Ch 13)
- When N approaches T, apply Ledoit-Wolf before PCA; a factor is recoverable only if its eigenvalue exceeds 1+√(N/T) (Ch 14)
- When selecting a CAE, do it by the workflow's IC/IC IR/spreads/turnover, not by reconstruction loss (Ch 14)
- When a causal effect is ≤0 but the predictive signal is positive, use it to predict without claiming causality (Ch 15)
- When claiming causality, require |t_HAC|>1.96 and p_ref<0.05; failing either downgrades it to a hypothesis (Ch 15)

**Costs, sizing and risk**
- When setting cadence, run horizon feasibility: if the typical move does not exceed the round-trip cost, no edge is possible (Ch 6)
- When execution data is missing, calibrate conservatively (quoted spread, wide slippage, high η), because underestimating costs destroys capital (Ch 18)
- When the regime is ambiguous, use stressed parameters, because costs multiply under stress (Ch 18)
- When the cost test fails: reduce turnover → improve execution → filter to liquid names → accept less capacity; if it still fails, abandon (Ch 18)
- When costs dominate, try a slower cadence before tuning the model (Ch 20)
- When alpha decays fast, execute aggressively; if slowly, passively (Ch 18)
- When signals are weak, use equal weight or inverse vol; MVO only with Ledoit-Wolf, caps and a turnover penalty (Ch 17)
- When using Kelly, go fractional 1/2-1/4, because overestimating μ produces extreme leverage (Ch 17)
- When long-short, scale the short leg by vol, not equal-notional (Ch 17)
- When comparing allocators, fix forecasts, costs, constraints and covariance, because otherwise you confound sizing with signal (Ch 17)
- When a neural allocator wins gross, require a breakeven curve, multiple seeds and comparison vs EW/inverse vol (VLSTM 0.66 → 0.22 Sharpe at 20 bps) (Ch 17)
- When reporting risk, show VaR and CVaR by regime with liquidity haircuts, because the unconditional figure hides tails (Ch 19)
- When an adaptive control uses a signal, lag it to t−1 and fix the rules ex ante (Ch 19)
- When a stress test reveals a vulnerability, map it to accept/hedge/reduce/kill switch, because without an action it is worthless (Ch 19)
- When the realistic cost approaches the break-even cost, the strategy is fragile (baseline: ~166 bp/side vs 5 bp assumed) (Ch 16)

**Production**
- When live differs from the backtest, verify identical inputs and then compare stage by stage (Ch 25)
- When something fails live, classify technical vs statistical first: fix vs retrain/retire (Ch 26)
- When the execution feed fails validation, do not trade (fail-closed) (Ch 26)
- When a submission times out, query by client_order_id before retrying (Ch 25)
- When reconciliation fails, halt and review manually; do not auto-correct (Ch 25)
- When a hard limit is breached, halt immediately; do not try to dig out of the hole (Ch 26)
- When live costs exceed simulated ones, do not retrain: it is microstructure or infrastructure (Ch 26)
- When promoting a model, require Sharpe ≥ incumbent + 0.2, p<0.10 and no new max drawdown after ~60 days of shadow (Ch 26)
- When the kill switch or staleness needs to persist, store it in a state file, because a restart resets the counter and fails open (Ch 25)
- When an agent would touch orders, do not allow it: read-only at L1-L2; cutoff at the tool boundary (Ch 24)

## 2. Decision trees

**Model family (by effective n and structure)**
- n < 5K → Ridge or LightGBM (Ch 13)
  - <10K and quick prototype → TabPFN (non-commercial license) (Ch 12)
- 5K-20K → LightGBM (Ch 13)
- 10K-100K production → LightGBM/XGBoost; >100K with categoricals → CatBoost (Ch 12)
- >20K → compare DL without substituting it by default (Ch 13)
  - sequence >10K steps → Mamba; >50 cross-asset series → iTransformer; heterogeneous covariates → TFT; long horizon → PatchTST (Ch 13)
- Always: Ridge and D/N-Linear floor; move up only with a robust OOS gain (Ch 11, Ch 13)

**GBM library**
- CPU + massive walk-forward → LightGBM (Ch 12)
- High-cardinality categoricals, GPU or latency → CatBoost (Ch 12)
- CUDA GPU + SHAP/Optuna ecosystem → XGBoost (Ch 12)

**Bar type**
- Clock-time horizon → time bars (Ch 3)
- ML default → dollar bars (Ch 3)
- Order flow is the object + aggressor side/Lee-Ready → imbalance/run bars (Ch 3)
- Trades only, no quotes → dollar bars, never imbalance (tick test ~78-80% vs ~95%) (Ch 3)

**Label**
- Fixed-horizon return → vol-scaled threshold (Ch 7)
- Multi-asset → cross-sectional percentile (Ch 7)
- Events with barriers → triple-barrier calibrated by MFE/MAE (Ch 7)
- Unknown horizon → trend scanning (Ch 7)
- Filter a primary signal → meta-labeling (Ch 7)

**Volatility estimator**
- Closes only → close-to-close; OHLC with small gaps → Garman-Klass; material overnight gaps → Yang-Zhang (Ch 8)

**Latent estimator**
- Risk → PCA; priced factors → RP-PCA; linear conditional betas → IPCA; nonlinear → CAE; direct pricing → SDF; predictive signal → SAE (Ch 14)

**Sizing/allocation method**
- Weak/noisy signal → equal weight or inverse vol (Ch 17)
- Vol-heterogeneous universe → risk parity/inverse vol (Ch 17, Ch 20)
- Score magnitude is informative → score-weighted (Ch 17)
- Covariance is the bottleneck → HRP (no views, long-only) or NCO (with views) (Ch 17)
- You want to scale by uncertainty → conformal sizing (Ch 17)
- Non-factorizable objective → end-to-end, but it must beat heuristics after costs (Ch 17)

**Cost model by participation**
- <0.5% ADV → spread; 0.5-2% → linear; >2% → square-root (Ch 18)
- Dispersed prices → bps-of-notional, not per share (Ch 18)
- Options → cost as % of premium (not bps of notional) (Ch 18)

**Backtest engine**
- No relevant state → vectorized (Ch 16)
- Cash, margin, orders, stops → event-driven (Ch 16)
- Going live → dual-mode framework (same Strategy for backtest/paper/live) (Ch 25)

**Multiple testing correction**
- You will promote few → Holm (FWER) (Ch 7)
- Large screen → BH (FDR) (Ch 7)
- Sharpe of the best of N → DSR / Reality Check / SPA (Ch 16)

**When to use RL**
- Execution, market making, hedging (continuous action, measurable reward) → RL (Ch 21)
  - little data → PPO; high-fidelity simulator → SAC/TD3 (Ch 21)
- Alpha → predictive pipeline, not RL (Ch 21)

**Retrieval / graph**
- Exact entities + concepts → hybrid BM25 + semantic + RRF + rerank (Ch 22)
- Numerical question → compute in code (Ch 22)
- Single attribute or broad narrative → no graph; multi-hop/co-ownership → Graph RAG (Ch 23)
- Hand-crafted first; GNN only if it improves OOS net of costs (Ch 23)

**Text: how many labels**
- A few hundred → frozen embeddings or synthetic labeling with an LLM (Ch 10)
- 1,000-10,000 → fine-tune with early stopping (Ch 10)
- Document >512 tokens → ModernBERT (8,192) or chunking 256-512 with overlap 64 (Ch 10)

**After a disappointing holdout**
- Weak signal → label/horizon/breadth/ensembling (Ch 20)
- Alpha fine but Sharpe drops → translation to portfolio/allocator (Ch 20)
- Costs dominate → slower cadence, liquidity filter (Ch 20)
- Parity OK + negative holdout → regime/factor exposure; monitoring decision (Ch 25)

**Model replacement / drift**
- PSI <0.10 stable; 0.10-0.25 investigate; >0.25 retrain if the feature has high importance (Ch 19, Ch 26)
- Decay vs data failure vs costs indistinguishable → do not intervene (Ch 1)

## 3. Trade-off matrices

| Choice | Wins on | Loses on | Ch |
|---|---|---|---|
| Ridge vs LASSO | diffuse signal, stability | sparsity | 11 |
| Nested walk-forward vs single-loop | IC honesty | compute | 11 |
| Split conformal vs CQR vs ACI | auditability / adaptivity / long-run coverage | fixed width / model dependence / no per-regime coverage | 11 |
| Dollar vs imbalance bars | robustness | capturing informed flow | 3 |
| Holm vs BH | false-positive control | power | 7 |
| LightGBM vs CatBoost vs XGBoost | CPU / GPU+categoricals / ecosystem | categoricals / CPU speed / speed | 12 |
| Zero-shot TSFM vs fine-tuned | zero cost | zero-shot gives negative IC on returns; useful fine-tuned on vol/VaR | 13 |
| HRP vs NCO vs MVO | stability / views / use of μ | ignores μ and is long-only / complexity / estimation error | 17 |
| Passive vs aggressive (execution) | low impact / alpha capture | alpha decay / impact | 18 |
| Slow shadow vs fast swap | avoids trading one failure for another | deployment velocity | 26 |
| Hard-switch vs continuous regime | simplicity | fragile in transitions | 9 |
| More trials vs untouched data | better apparent tuning | validation overfitting | 12 |

## 4. Thresholds and numerical defaults

- Label buffer = label horizon; retrain with ≥1-5% new data or systematic OOS degradation (Ch 6)
- 21 calendar days ≈ 15 trading days (Ch 3)
- Gross spread ≥ 2x costs (Ch 7); t>3 for new factors (Ch 7)
- Feature dedup: |ρ| 0.6-0.8 (Ch 8)
- Reference IC: 0.03-0.05 small; IC ceiling ~+0.085 in the case studies (Ch 17, Ch 20)
- AUC 0.55-0.60 significant; >0.65 suspicious (Ch 11)
- GBM: lr 0.01-0.05, 100-200 trials (50-100 if SNR is very low) (Ch 12)
- Conformal: 80/20 split; ACI γ ∈ [0.005, 0.05]; coverage <80% with a 90% target during vol spikes = stale calibration (Ch 11)
- Uncertainty ensemble: ~5 models; MC dropout 50-100 passes (Ch 13)
- Net ≈ gross − 2 × turnover × one-way cost (5-10 bps large cap; 50+ microcaps) (Ch 14)
- Breakeven cost per leg: ETFs 50+, CME ~30, crypto ~15, FX ~7 bps (Ch 20)
- Capacity: minimum net Sharpe typ. 0.5 (Ch 18)
- Kelly fractional 1/2-1/4 (Ch 17)
- Drawdown kill-switch levels 5/10/15/20/30%; gross caps ~200% / ~150% / ~100% by regime (Ch 19)
- PSI <0.10 / 0.10-0.25 / >0.25 (Ch 19, Ch 26)
- Alerts: live Sharpe persistently 0.5+ below backtest = overfitting or regime (Ch 26)
- Rollout: shadow ~60d; A/B 5-10% of capital; steps of 10-25-50-100% over 60-90 days (Ch 26)
- Promotion: ΔSharpe ≥ 0.2, p<0.10 (Ch 26)
- RAG: chunk 400-600 tokens, overlap 50-100; RRF k=60; eval 50-100 queries (Ch 22)
- KG: schema-valid >90%, provenance >95%; mean degree <2-3 = poor extraction (Ch 23)
- Agents: step limit 5 (Ch 24)
- Entity resolution: ~95% auto / 85-95% review / <85% escalate (Ch 4)
- Diff. privacy: ε 5-10 practical; ε=1 degrades heavily (Ch 5)
- Cantelli: P(losing period) ≤ 1/(1+SR²) (Ch 19)
- Drawdown: −30% requires +43%; −50% requires +100% (Ch 19)

## 5. Tells & smells

- IC concentrated in one episode, sign that flips between folds, IC that does not decay or rises at distant lags → aggregation artifact or leakage (Ch 7)
- Unadjusted Sharpe >2.0 in a diversified, scalable strategy → scrutinize leakage, costs, capacity and multiple testing; 1.5-2.0 → check for search bias (Ch 16)
- Training Sharpe >12 with validation collapsing within a few epochs → end-to-end overfitting (Ch 17)
- Linear baseline beats the GBM, or the GBM improves by a point on a single split → require persistence in rolling IC (Ch 12)
- Encoder that does not beat TF-IDF → check labels, alignment and task before blaming the model (Ch 10)
- Text IC that collapses with a 1-hour delay → speed, not information (Ch 10)
- Benchmark that drops >30 pts out of distribution → label/corpus shift (Ch 10)
- Daily lead-lag between liquid names with correlation ≈ −0.06 to −0.13 → microstructure, not information (Ch 8)
- Daily jumps in rolling Hurst or daily GARCH/HAR refits → estimation noise (Ch 9)
- Trade nulls in extended hours (20-23% of bars) → no execution occurred; not missing data (Ch 3)
- SHAP dominated by a single feature or with signs opposite to documented factors → fragility (Ch 11)
- High PROCEED rate with a bad holdout (Crypto 71% → −0.13) → FDR triage is a screen, not a predictor (Ch 20)
- Top-1 that falls apart out of sample (NASDAQ-100 +2.11 → −0.21) → top cluster; ensemble (Ch 20)
- Factor R² ~0.995 with alpha 0.47% → factor exposure disguised as alpha (Ch 19)
- High Var(d²) with a mean above expected → unstable precision matrix; shrinkage before MVO (Ch 19)
- Bps-of-notional for options → underestimates by 15-75 bps (Ch 18)
- RL state with future info (daily close in an intraday agent) → state leakage (Ch 21)
- RL policy with wealth ≈ benchmark but more dispersion/inventory → no dominance (Ch 21)
- Citations with plausible page numbers that do not exist → verify with a supporting sentence (Ch 22)
- Agents with identical searches and forecasts → broken temperature or a bug (Ch 24)
- A threshold that fires often without changing decisions → noise (alert fatigue) (Ch 26)
- Backtest with signal and execution on the same bar → lookahead (Ch 16)
- Best model with ΔIC 0.02 that vanishes on clean data → validation overfitting (Ch 12)
