# Chapter 1: The Process Is Your Edge

## Core Idea
Durable trading performance depends less on choosing the "right model" and more on maintaining a disciplined research-to-production workflow (ML4T Workflow) that withstands non-stationarity, costs, and operational frictions. The separation between exploration and confirmation (evidence boundary) is the central integrity mechanism.

## Frameworks Introduced
- **ML4T Workflow**: two layers. (1) Data infrastructure (Ch 2-5): reproducible, auditable, point-in-time correct data. (2) Strategy research (Ch 6-21): iterative loop scoping -> features/labels -> models -> strategy design -> deployment/monitoring, with artifacts flowing downstream and live results feeding back into hypotheses.
  - **When to use**: any systematic strategy; does not require ML (discretionary signals, rules, or learned models use the same structure).
  - **How**: (1) fix scoping invariants; (2) iterate features and models as a nested loop; (3) turn signals into trades (signal-to-trade, portfolio, backtest as falsification, execution/risk); (4) deploy in phases and monitor; (5) feed findings back.
  - **Why it works / failure mode**: the most damaging errors in trading do not break the code, they inflate the backtest (leakage, revisions, unrealistic execution assumptions). Fails if treated as a linear sequence or if invariants are redefined after seeing results.
- **Evidence boundary (Exploration vs Confirmation)**: separation between exploration mode (use data for diagnostics, record every trial in a research ledger) and confirmation mode (untouched sealed holdout, frozen specification, predefined metrics, selection-adjusted inference that counts trials).
  - **When to use**: whenever there is iterative search (features, hyperparameters, variants).
  - **How**: log trials; reserve a holdout; freeze the spec; evaluate once with inference adjusted for the number of trials.
  - **Why it works / failure mode**: credibility comes from being able to count and characterize the search. Fails if the holdout is "peeked at" or no records are kept (impossible to distinguish learning from selection).
- **Scoping invariants (what gets fixed before iterating)**: decision-time correctness, tradability/universe rules, label and horizon, cost-model class, evaluation protocol (walk-forward, holdout, trial logging). Cost parameters can be estimated; the class is fixed.
- **Phased deployment**: shadow (log orders without executing) -> canary (minimal capital, end-to-end test) -> full scale only with demonstrated stability.
- **Four-way monitoring**: data health, signal health (feature/label drift, calibration), execution health (slippage, fill rates, impact), risk behavior (breaches, exposure creep, drawdown). Explicit retrain / pause / retire rules.
- **Prediction-first vs Mechanism-first**: prediction-first = target + many features + high capacity with regularization and ensembles; mechanism-first = economic rationale that bounds the search and makes monitoring actionable. Litmus test: if being wrong means "doesn't make money after costs," it is a signal; if it means "biased estimate," it is measurement (risk of "factor mirages").
- **Regime lens (3 uses)**: ex post labeling (explanation, full sample), backtest conditioning (robustness, no label leakage), live monitoring (decision-time information only, walk-forward). Regimes = risk lens, not alpha timing.
- **Solo checkpoints**: Scoping check, Data integrity check, Signal check, Tradability check (fragility audit), Monitoring check.

## Key Concepts
- **Structural break**: abrupt change in the data-generating process (the "when").
- **Regime**: persistent state with properties (vol, correlations) stable within and distinct across states.
- **Data drift / Concept drift**: shift in the feature distribution / change in the feature-target relationship; they are flags that trigger investigation, not diagnoses.
- **Online detection**: real-time change detection using only information available at decision time (distinct from ex post labeling).
- **Adaptive Markets Hypothesis (Lo, 2004)**: efficiency is an evolutionary outcome that varies with the "market ecology."
- **Alpha factory**: repeatable pipeline that generates, evaluates, deploys, and monitors strategies (López de Prado).
- **Research ledger**: record of trials used to count and characterize the search.
- **Goalpost drift / Assumption stacking**: redefining success after seeing results / accumulating mildly favorable assumptions that manufacture performance.
- **Leakage by construction**: features, labels, or prompts that incorporate the future, even via LLM training data.
- **CRISP-ML(Q)**: adaptation of CRISP-DM to ML with handoff checks and post-deploy monitoring.

## Mental Models
- Use **drift as an investigation flag** when a metric moves; do not conclude a cause until you separate data, signal, execution, and regime.
- Use **regimes as a risk lens** when you need to know which states hurt the strategy; not as a timing signal.
- Use **mechanism-first** when the effective sample is short and the winner's curse looms; use **prediction-first** only with anti-leakage protocols and a statistical budget.
- Use a **fragility audit** (double costs, delay, size caps, intra-bar shift) when evaluating whether an edge survives; gradual degradation is worth more than an optimal point.
- Use **automated output as a candidate** when an AI generates code/features/explanations: it must pass the same safeguards.

## Anti-patterns
- **Data mining and narrative overfitting**: post-hoc hypotheses after seeing results; no pre-commitment to what would falsify them.
- **Leakage and non-point-in-time data**: revisions, survivorship, corporate actions, timestamp errors.
- **Multiple testing / selection bias**: searching variants until one "works" and treating noise as signal (Harvey, Liu, Zhu 2016).
- **Ignoring implementability**: confusing predictability with edge after costs and frictions.
- **Sunk cost / late exits**: keeping a decaying strategy because it used to work.
- **Using ex post regime labels in trading logic**: look-ahead bias.
- **Sharing degrees of freedom without counting them**: results sensitive to features, windows, number of states.
- **Competing on speed/microstructure or with expensive institutional data** as an independent.

## Code Examples
Not applicable (conceptual chapter; notebooks factor_regimes.ipynb and macro_regimes.ipynb are referenced without code in the text).

## Reference Tables
| Concept | Question it answers | Use |
|---|---|---|
| Structural break | When did the process change | Ex post explanation |
| Regime | Which persistent state are we in | Risk lens |
| Data drift | Did the input distribution change? | Monitoring flag |
| Concept drift | Did the feature-target relationship change? | Monitoring flag |
| Online detection | Can I detect it with info at t? | Live operation |

| Regime mode | Sample | Evaluation |
|---|---|---|
| Ex post labeling | Full | Descriptive |
| Backtest conditioning | Full, no leakage into features | Stress test |
| Live monitoring | Decision time only | Walk-forward |

| Entry point | Starts with | Main risk |
|---|---|---|
| Prediction-first | Target + many features | "Lucky" models get promoted |
| Mechanism-first | Economic rationale | Misspecification / factor mirages |

## Worked Example
Factor regimes with "Century of Factor Premia" (AQR, 1927-2024, monthly market + factor returns): standardize the series, fit a GMM with 2-6 components. AIC prefers K=6 but silhouette ~0 and the partition fragments after 1950, so K=2 is chosen (more stable and interpretable); states are labeled ex post "Risk-On"/"Risk-Off" by the worse equity outcome. Results: vol 19.1% (Risk-Off) vs 8.6% (Risk-On), ratio 2.2x; market Sharpe 1.11 -> 0.12; max drawdown -77% vs -23%; value countercyclical (+5.3% vs +1.6% annualized), carry (-0.6%) and defensive (-0.5%) negative in Risk-Off. 267 transitions in 98 years (~every 4 months): too noisy as a tactical signal. Decision: use the map to adjust risk posture (sizing, caps, buffers), not to "buy in Risk-On."
Macro regimes: UNRATE, DFF, T10Y2Y, CPIAUCSL (YoY); monthly resampling, standardization, 4-component GMM; cophenetic correlation 0.710 in hierarchical clustering; Ward, GMM, and K-Means yield comparable structure. Macro regimes separate vol and drawdown better than mean returns.

## Key Takeaways
- Fix scoping invariants (decision point, universe, label/horizon, cost class, protocol) before optimizing and do not change them after seeing results.
- Record every trial and confirm with a sealed holdout and selection-adjusted inference.
- Design monitoring that separates data, signal, execution, and risk, with retrain/pause/retire triggers written in advance.
- Use regimes for risk and stress testing; never use ex post labels in trading rules.
- As an independent, impose your own governance: stop criteria, fragility audit, fast post-mortems; invest in reusable infrastructure.
- Treat generative AI and causal inference as amplifiers: without guardrails they produce false signals faster.

## Connects To
Ch 2-5 (data infrastructure), Ch 6 (research framework), Ch 7-10 (features/labels), Ch 11-15 (models, HMMs, conformal prediction), Ch 16 (backtest harnesses), Ch 18 (cost models), Ch 19 (risk, measurement), Ch 21 (RL), Ch 22-24 (RAG, knowledge graphs, agents), Ch 25-26 (live ops, MLOps, governance).

## Code Reference

Official repository: [01_process_is_edge/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/01_process_is_edge). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`factor_regimes.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/01_process_is_edge/factor_regimes.ipynb) + `.py` | 1.4 | Demonstrates unsupervised learning for market regime detection using Gaussian Mixture Models (GMM) on factor returns from the AQR Century of Factor Premia dataset. |
| [`macro_regimes.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/01_process_is_edge/macro_regimes.ipynb) + `.py` | 1.4 | Demonstrates unsupervised learning for market regime detection using macroeconomic indicators from FRED, validated against S&P 500 volatility and drawdowns. |
