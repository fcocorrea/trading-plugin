---
name: trading-cuantitativo
description: "Knowledge base from \"Machine Learning for Trading, Third Edition\" by Stefan Jansen. Use when applying the ML4T workflow for quantitative trading: backtesting and walk-forward validation, labels and feature engineering, multiple-testing control (DSR), gradient boosting and deep learning for returns, portfolio construction, transaction costs, risk management, live trading and MLOps; or studying the book and referencing its concepts."
argument-hint: "[topic, framework or chapter number]"
---

# Machine Learning for Trading — Third Edition
**Author**: Stefan Jansen | **Pages**: ~827 | **Chapters**: 27 | **Generated**: 2026-10-27

## How to use this skill

- **No arguments** — use the Core Frameworks below.
- **With a topic** (`triple-barrier`, `DSR`, `HRP`, `kill switch`) — look it up in the Topic Index and read the chapter it points to before answering.
- **With a chapter** (`ch16`) — load `chapters/ch16-*.md`.
- **Making a decision** — start with [cheatsheet.md](cheatsheet.md) (rules, thresholds, decision trees).
- **Writing code or proposing a design** — before proposing anything, find the reference implementation in the book's repository: delegate to the `ml4t-code-scout` agent (or use `/trading:ml4t <problem>`). Work on the local copy that `ml4t-ref` maintains (`ml4t-ref status`, `ml4t-ref path`, `ml4t-ref url <path>`), not on live GitHub. Base the solution on that code, cite the permalink and explain every deviation. The **Code Reference** table of each chapter and [code-reference.md](code-reference.md) are the index to find your way.

If the topic is not in the Core Frameworks, read the matching chapter before answering. Formulas came out corrupted when the PDF was extracted: they were rebuilt from the surrounding text, and the code snippets inside the chapters are illustrative. The real code is in the repository notebooks (see Code Reference).

## Relationship with the plugin's other skills

This skill is the **methodology**. The plugin's engine skills describe what the engine **does today**: `experiment-design` (experiment layout and validation), `data-schema`, `execution-model`, `trading-metrics`, `market-profiles`, `add-source`.
- When designing research, follow this skill; `experiment-design` is how that design is written down for the engine.
- When the engine lacks something the book requires (funding, market impact, CPCV, the runner itself), say so explicitly and propose the implementation from the book's code. Never present the engine's simplification as the book's method, nor the book's method as something the engine already does.

---

## Core Frameworks & Mental Models

**The process is the edge (ML4T Workflow, ch01, ch27).** Durable performance comes from a disciplined research-to-production workflow, not from the "right model". Use the **evidence boundary**: separate exploration from confirmation with a sealed holdout and count every trial you ran. Deploy in phases and monitor regimes (non-stationarity is the norm).

**Data: two failures dominate (ch02, ch04).** *Time consistency* (point-in-time, bitemporal, restatements) and *entity consistency* (entity resolution, identifiers). Before blaming the model, check survivorship, corporate actions, futures rolls and identifier joins. Every dataset must pass vendor due diligence and the five quality dimensions.

**Microstructure (ch03).** Data is the output of a matching engine: L1/L2/L3, LOB reconstruction, trade classification (Lee-Ready). Sampling in bars (dollar/volume/imbalance/run) changes the statistical meaning of each observation.

**Synthetic data (ch05).** Use synthetic data to evaluate robustness as a distribution, not as a substitute for historical evidence. Measure with **Fidelity-Utility-Privacy** and TSTR/TRTR; its value is judged by the downstream task.

**Research framework (ch06).** Four commitments: a versioned trading setup, objectives and metrics separated by role (model, signal, strategy), a temporal protocol (walk-forward with purging/embargo, CPCV), and a **baseline checkpoint** with run logging of every trial.

**Learning task and features (ch07-ch10).** Preprocess *split-aware*; define labels consistent with execution (fixed-horizon, **triple-barrier** with overlap/uniqueness, meta-labeling). Evaluate per fold with rank IC / ICIR and control the search with Holm/BH and search accounting. A feature is a point-in-time economic hypothesis; combine **Signal × State** (gating/scaling) and apply the *three-step filter*. Model-based features (Kalman, GARCH/HAR, HMM) are re-estimated inside the walk-forward. For text, temporal discipline matters more than the NLP model.

**Model ladder (ch11-ch14).** Regularized linear baseline (Ridge/LASSO/Elastic Net) with walk-forward, SHAP and conformal prediction; every later model must beat it. GBM (XGBoost/LightGBM/CatBoost) is the tabular default, justified only if the IC with a HAC interval excludes zero. In deep learning no architecture wins by default: define the target well first and compare against a linear baseline (LTSF-Linear). Factor models: a factor that explains covariance is not necessarily priced.

**Causality (ch15).** Pipeline DAG → admissible adjustment → estimand → estimation (DML with cross-fitting) → refutation (placebo, sensitivity, stability). ML does not create identification.

**Backtest = falsification (ch16).** Explicit protocol (timing, fills, costs, sizing, constraints), a non-ML baseline, cost sensitivity and regime slicing. Correct the selected Sharpe with the **Deflated Sharpe Ratio**, Reality Check and trial accounting.

**Implementation (ch17-ch20).** The allocator is one more model (**allocator term sheet**): simple heuristics (equal weight, inverse vol) are a high bar; HRP, Kelly and robust MVO with shrinkage. Costs explain the gap between theoretical and realized alpha: estimate, trade, measure (TCA), recalibrate; compute break-even turnover and capacity. Risk is system design: regime-conditional VaR/CVaR, stress testing, adaptive controls without lookahead, and kill switches. Synthesis: predictive quality ≠ performance; signal quality, translation into a portfolio, cost survival and stability are separate stages.

**Advanced AI (ch21-ch24).** RL fits where the action is the object of optimization (execution, market making, hedging), not directional alpha; the sim-to-real gap is the barrier. RAG with hybrid retrieval and tool-verified computation; knowledge graphs only for genuinely relational questions, with temporal discipline; agents as engineered systems with bounded, read-only autonomy.

**Production (ch25-ch26).** Most live failures are self-inflicted technical divergence: the same code runs in backtest, paper and live; order state machine, parity verification, a 4-level kill switch and a preflight checklist. Distinguish **technical failure** (fix the bug) from **statistical failure** (decay); drift with PSI/K-S/SHAP/ADWIN, safe rollout and circuit breakers.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|--------|----------------|
| [ch01](chapters/ch01-process-is-your-edge.md) | The Process Is Your Edge | ML4T Workflow, evidence boundary, phased deployment |
| [ch02](chapters/ch02-financial-data-universe.md) | The Financial Data Universe | data taxonomy, PIT/bitemporal, vendor due diligence |
| [ch03](chapters/ch03-market-microstructure.md) | Market Microstructure | LOB reconstruction, dollar/imbalance bars, Lee-Ready |
| [ch04](chapters/ch04-fundamental-alternative-data.md) | Fundamental and Alternative Data | as-of query, entity resolution, alt data evaluation |
| [ch05](chapters/ch05-synthetic-financial-data.md) | Synthetic Financial Data | Fidelity-Utility-Privacy, TSTR/TRTR, TimeGAN, diffusion |
| [ch06](chapters/ch06-strategy-research-framework.md) | Strategy Research Framework | strategy map, versioned setup, walk-forward, run logging |
| [ch07](chapters/ch07-defining-learning-task.md) | Defining the Learning Task | triple-barrier, rank IC, Proceed-Revise-Stop triage |
| [ch08](chapters/ch08-financial-feature-engineering.md) | Financial Feature Engineering | three-step filter, Signal × State |
| [ch09](chapters/ch09-model-based-feature-extraction.md) | Model-Based Feature Extraction | Kalman, GARCH/HAR, HMM regimes |
| [ch10](chapters/ch10-text-feature-engineering.md) | Text Feature Engineering | text-to-signal contract, FinBERT, extract-to-schema |
| [ch11](chapters/ch11-ml-pipeline.md) | The ML Pipeline | Ridge/LASSO, SHAP, conformal (ACI, CQR) |
| [ch12](chapters/ch12-advanced-tabular-models.md) | Advanced Models for Tabular Data | GBM, LambdaMART, Optuna, TreeSHAP |
| [ch13](chapters/ch13-deep-learning-time-series.md) | Deep Learning for Time Series | baseline ladder, N-BEATS, PatchTST, Mamba |
| [ch14](chapters/ch14-latent-factor-models.md) | Latent Factor Models | PCA, IPCA, conditional autoencoder, adversarial SDF |
| [ch15](chapters/ch15-causal-machine-learning.md) | Causal Machine Learning | DAG, DML, BSTS, PCMCI |
| [ch16](chapters/ch16-strategy-simulation.md) | Strategy Simulation | backtest protocol, DSR, Reality Check |
| [ch17](chapters/ch17-portfolio-construction.md) | Portfolio Construction | allocator term sheet, HRP, Kelly, DeePM |
| [ch18](chapters/ch18-transaction-costs.md) | Transaction Costs | square-root impact, Almgren-Chriss, TCA, break-even |
| [ch19](chapters/ch19-risk-management.md) | Risk Management | VaR/CVaR, stress testing, kill switches, PSI |
| [ch20](chapters/ch20-strategy-synthesis.md) | Strategy Synthesis | FDR triage, cost survival, dual ML |
| [ch21](chapters/ch21-reinforcement-learning.md) | Reinforcement Learning | PPO/SAC, execution RL, deep hedging |
| [ch22](chapters/ch22-rag-financial-research.md) | RAG for Financial Research | hybrid retrieval + RRF, constraint prompting |
| [ch23](chapters/ch23-knowledge-graphs.md) | Knowledge Graphs | Graph RAG, three-timestamp model |
| [ch24](chapters/ch24-autonomous-agents.md) | Autonomous Agents | ReAct, tool contracts, Warden pattern |
| [ch25](chapters/ch25-live-trading-systems.md) | Live Trading Systems | order state machine, kill switch, preflight |
| [ch26](chapters/ch26-mlops-governance.md) | MLOps and Governance | drift detection, safe rollout, circuit breakers |
| [ch27](chapters/ch27-systematic-edge.md) | The Systematic Edge | alpha factory, T-shaped expertise |

## Topic Index

- **Agents / ReAct / Warden** → ch24
- **Alpha decay** → ch01, ch04, ch06
- **Alternative data** → ch04
- **Bars (dollar/imbalance/run)** → ch03
- **Backtest (vectorized vs event-driven)** → ch16
- **Break-even turnover / capacity** → ch18, ch20
- **Causal inference / DML / DAG** → ch07, ch15, ch20
- **Circuit breakers / kill switch** → ch19, ch25, ch26
- **Conformal prediction (ACI, CQR)** → ch11, ch13, ch17
- **CPCV / purging / embargo / walk-forward** → ch06, ch07
- **Deflated Sharpe / Reality Check** → ch16
- **Deep hedging / RL execution** → ch21
- **Drift (PSI, K-S, ADWIN)** → ch19, ch26
- **Entity resolution** → ch02, ch04
- **Evidence boundary / sealed holdout** → ch01, ch06
- **Feature engineering / Signal × State** → ch08
- **Fractional differencing / stationarity** → ch08, ch09
- **GARCH / HAR / volatility** → ch08, ch09, ch19
- **GBM (XGBoost/LightGBM/CatBoost)** → ch12
- **HMM / regimes** → ch01, ch09, ch16, ch19
- **HRP / Kelly / MVO** → ch17
- **IC / ICIR** → ch07, ch12, ch20
- **Knowledge graph / Graph RAG** → ch23
- **Labels (triple-barrier, meta-labeling)** → ch07
- **Latent factors (PCA, IPCA, autoencoders)** → ch14
- **LOB / Lee-Ready / microstructure** → ch03
- **Live trading / IBKR / Alpaca** → ch25
- **LLM / FinBERT / embeddings** → ch10, ch22
- **Model governance (SR 11-7)** → ch26
- **MLOps / feature store** → ch26
- **Multiple testing / Holm / BH / FDR** → ch05, ch06, ch07, ch16, ch20
- **Point-in-time / lookahead / survivorship** → ch02, ch04, ch06
- **Portfolio construction / allocator** → ch17
- **RAG** → ch22
- **Risk (VaR/CVaR, stress testing)** → ch19
- **SHAP** → ch11, ch12
- **Synthetic data (TimeGAN, diffusion, TSTR)** → ch05
- **Sharpe inference** → ch16
- **Transaction costs / impact / TCA** → ch18
- **Transformers / N-BEATS / Mamba** → ch10, ch13

## Supporting Files

- [glossary.md](glossary.md) — key terms with definitions
- [patterns.md](patterns.md) — techniques and patterns by part of the book
- [cheatsheet.md](cheatsheet.md) — decision rules, trees, thresholds and tells
- [code-reference.md](code-reference.md) — the book's code repository (GitHub): folders per chapter, 9 case studies, `utils/`, installation and execution. Each `chapters/chNN-*.md` ends with a **Code Reference** table of its `.ipynb`/`.py` notebooks

---

## Scope & Limits

Covers only the book's content (synthesized summaries, not the text). Formulas and some PDF tables were rebuilt from the surrounding text, and the code snippets are illustrative; for exact implementations use the repository notebooks (github.com/stefan-jansen/machine-learning-for-trading, listed in Code Reference). The repository is a living project: its file names may change. For topics outside the book, use other skills or ask directly.
