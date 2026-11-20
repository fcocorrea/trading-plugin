# Chapter 6: Strategy Research Framework

## Core Idea
Turning an idea into reliable evidence requires a research program with four commitments: a versioned trading setup, explicit economic objectives with metrics separated by role, a temporal evaluation protocol with a sealed holdout, and a baseline checkpoint with run logging that keeps the search accountable and recoverable.

## Frameworks Introduced
- **Live trading loop (5 steps)**: (1) observe a decision snapshot, (2) compute features and produce a score, (3) translate the score into positions (trading policy, limits), (4) execute and realize net outcomes (slippage, spreads, fees, financing, funding, roll), (5) monitor, update, and repeat. Only meaningful with well-defined timing.
- **Research loop (5 steps)**: (1) define the evaluation environment (universe, cadence, constraints, material costs); (2) minimal end-to-end baseline pipeline; (3) evaluate under a time-series protocol with a reserved holdout; (4) diagnose and revise by changing one component at a time; (5) log.
- **Strategy map**: two complementary lenses. Strategy families (cadence, information source, dominant frictions) = feasibility filters; Sources of edge = narratives of why it persists = durability filters. A strategy is well posed if it answers: family and cadence, PnL decomposition, edge narrative, 2-3 top failure modes.
  - **Why it works / failure mode**: a credible story with an unworkable implementation is not a strategy; a feasible implementation without a story is a backtest with uninterpretable failures. ML is not a family: it is a layer that increases degrees of freedom and therefore raises the value of a tightly bounded definition.
- **Six sources of excess returns (Paleologo, 2025)**: see table. Rules: edges based only on information erode; those based on risk tolerance, constraints, or capacity can persist even when they are known.
- **Versioned trading setup** (`setup.yaml`): fixes universe/tradability, decision schedule and admissible information, cadence/horizon, score-to-trade mapping (state space, entry, exit, sizing, order timing), constraints and risk controls, cost model class and components.
  - **When to use**: before iterating; every run records the version.
  - **How**: explore freely within invariants; changing mechanics = version bump (v2) + new baseline; changing parameters = same version.
  - **Why it works / failure mode**: preserves comparability. Fails if the evaluation environment drifts unannounced (instruments, snapshot, execution timing, cost treatment).
- **Horizon feasibility analysis**: for each cadence, what fraction of typical moves exceeds the round-trip cost? Three zones: hard floor (costs dominate), gray zone (depends on signal and turnover), comfortable zone (costs are not a binding constraint).
- **Three metric layers**: Model diagnostics (does it learn the label and generalize across splits? loss, calibration, stability; no costs), Signal diagnostics (does it behave like a tradable signal? IC = Spearman between score and subsequent return), Strategy outcomes (net risk-return, vol, drawdown, exposure, turnover; late-stage). One primary selection metric for development and a separate reporting set.
- **Walk-forward CV with temporal buffers**: fit on a past window, predict a future window, score, advance; restrict admissibility at the boundary.
  - **How (7 steps)**: choose the training window; the validation window; enforce admissibility (labels already resolved); fit; predict; score with the selection metric; advance a fixed step and repeat.
  - **Why it works / failure mode**: replicates deployment. Fails from leakage via overlapping windows, buffers counted in calendar days instead of trading days, or random k-fold.
- **Label buffer (purging) and Feature buffer (embargo)**: label buffer = label horizon (in trading days); feature buffer only when there is training after the validation window (k-fold, combinatorial).
- **Sealed holdout + nested walk-forward**: holdout used only once after freezing the pipeline; fixed holdout (tune once, test once) or rolling retuning (outer test loop, inner selection loop; Bates et al. 2021 show better CIs with nested CV).
- **CPCV (Combinatorial Purged CV)**: choose k of n contiguous groups -> C(n,k) train/test configurations with both buffers; a distribution of OOS results instead of a single path. Expensive, diminishing returns with short series or a single regime (details in Ch 17).
- **Baseline checkpoint + three preflight sanity checks**: timing sanity, coverage sanity, trading-intensity sanity. Narrow first run: one label and horizon, compact feature families, one simple model class, one walk-forward design.
- **Search accounting / trial taxonomy**: Strategy > Trial family > Trial > Run; run log with 5 categories (provenance, data and evaluation, configuration, artifacts, decision and gates). Hypothesis pre-registration separates confirmatory (counts toward deflation) from exploratory.

## Key Concepts
- **Decision cadence / Holding period / Forecast horizon / Lookback window**: four notions of time that must be kept separate (Box 6.1).
- **Edge**: repeatable advantage that makes the distribution of net per-trade outcomes favorable.
- **Alpha**: performance relative to a benchmark (index or factor model).
- **Capacity**: capital deployable before the edge degrades from impact, fees, financing.
- **Label leakage / Standardization leakage / Threshold leakage**: using future returns in features / computing means and standard deviations over the full sample / setting thresholds with validation or test data.
- **Admissible sample at t**: features computable at or before t and label already known at t.
- **Setup version**: incremented on changes to universe, schedule, mapping, constraints, or cost class.
- **Hypothesis class**: ML explores a family of measures, not a single hypothesis.
- **Prediction vs profit**: Palm/3Com (2000): obvious mispricing but scarce shorts, extreme borrow fees, and open funding risk.

## Mental Models
- Use **the family as a feasibility checklist** when choosing cadence: what breaks first.
- Use **risk/constraint edges** when seeking persistence; **informational edges** should expect decay (~50-58% post-publication).
- Use **mechanics vs parameter** when unsure whether something requires a new setup version.
- Use **model > signal > strategy diagnostics in that order**: strategy outcomes come late, or you overfit to the simulator.
- Use **"which labeled samples were available at t?"** as the single admissibility test.

## Anti-patterns
- **Moving the goalposts**: changing universe, snapshot, timing, or costs between iterations.
- **Using strategy outcomes as the objective of every micro-decision**: overfitting to the simulator's degrees of freedom.
- **Random k-fold on time series**: trains on timestamps after the validation window; overlapping windows.
- **Buffers in calendar days**: a 21-calendar-day label buffer covers ~15 trading days and leaves 6 of leakage.
- **Setting thresholds or standardizing with validation/test data.**
- **Looking at the holdout during development**.
- **A backtest of a published anomaly as evidence**: it is a hypothesis (decay ~50%, implementation gap, sensitivity to details).
- **Changing the evaluation design mid-research** and calling it an "improvement".
- **Optimizing around a fragile definition** instead of revising the setup.
- **Not logging runs**: selection becomes invisible and the Deflated Sharpe cannot be computed.
- **Assuming prediction = profitable trade**: verify short access, borrow, and funding.

## Code Examples
Compact reconstruction of the versioned setup described for ETF momentum v1 (`setup.yaml`):
```yaml
setup_version: v1
universe:
  source: 100 cross-asset ETFs
  eligibility: trailing_annual_ADV >= 10_000_000
decision:
  schedule: month_end_close
  execution: next_day_open
  variant: weekly_5d
mapping:
  state_space: long_only
  entry: rank_top_n
  sizing: equal_weight
constraints:
  leverage: none
costs:
  commission_per_share: 0.0035
  half_spread_cents: {mega_etf: 0.5, sector: 1, default: 2}
```
What it demonstrates: the invariants that define a setup (universe, schedule, mapping, constraints, costs); changing `state_space` to long/short requires `v2`, while changing top-N from 10 to 20 stays in `v1`. Illustrative reconstruction in the book's style, not literal code.

## Reference Tables
| Source of edge | Mechanism | Durability | ML4T example |
|---|---|---|---|
| Risk compensation | Taking exposures others avoid (tail, vol, illiquidity) | High | cme_futures carry, sp500_options VRP |
| Liquidity/immediacy provision | Premium for providing immediacy or absorbing inventory | High | nasdaq100_microstructure reversals |
| Funding constraints | Dislocations when capital is scarce | Moderate (episodic, under stress) | Mean reversion after forced liquidations |
| Flow predictability | Positioning ahead of mandated demand (indices, hedging) | Moderate | Index reconstitution, ETF creation/redemption |
| Informational advantage | Better inference from public data | Low to moderate | ML feature engineering |
| Pure arbitrage | Mispricing of identical assets | Very low | Not represented |

| Same version (parameters) | New version (mechanics) |
|---|---|
| Threshold 1.8 -> 2.0 | Rank selection -> threshold gating |
| Top 10% -> top 20% | Long-only -> long/short |
| Normalization window 20d -> 60d | Add a stop rule where none existed |
| Different lookback | Change cadence or execution delay |
| Cost 10 -> 15 bps per leg | Add funding/roll as a cost |

| Family | Dominant | First screens | Case studies |
|---|---|---|---|
| Price-based | Turnover and execution (short); episodic drawdowns (long) | Robustness to lookback/rebalance; turnover-aware costs | etfs, us_equities_panel, fx_pairs |
| Fundamental/valuation | Meaning of inputs at decision time; small effective sample | PIT and coverage | us_firm_characteristics |
| Flow/microstructure | Timing, latency, fills vs depth | Observability and execution | nasdaq100_microstructure |
| Market mechanics/payoff | Roll, funding, margin, option payoff | Attribution by component, stress | cme_futures, crypto_perps_funding, sp500_options |

Nine case studies: etfs (~100 symbols, daily, 20 years), us_equities_panel (~3,000, daily, 50 years), us_firm_characteristics (~2,500, monthly, 1996-2016, ~57 characteristics), fx_pairs (20 pairs, 4h/daily NY 5PM), cme_futures (30 contracts, 15 years), crypto_perps_funding (~20, 8h, 5 years), nasdaq100_microstructure (~114, minute to 15 min, 2 years), sp500_equity_option_analytics and sp500_options (daily, 5 years).

Design commitments to log: train/val/test windows, step size, retraining cadence, label horizon, longest feature lookback, test period. Retraining rule: retrain when new data is >= 1-5% of the training window or OOS performance degrades systematically.

## Worked Example
ETF cross-asset momentum setup v1: universe of 100 ETFs with PIT eligibility by trailing annual ADV >= $10M (70-95 eligible ETFs per year; residual survivorship due to lack of historical constituents); monthly decision at the month-end close with execution at the next day's open (weekly 5-day variant); long-only mapping, top-N rank, equal-weight (long-only because shorting ETFs is expensive, equal-weight to isolate the signal from sizing); no leverage; costs: commission $0.0035/share (IBKR Pro Tiered) plus tiered half-spread (0.5¢ mega-ETFs, 1¢ sector, 2¢ thematic/regional). Horizon feasibility with a 5-15 bps per leg reference line: monthly moves exceed a 30 bps round-trip cost in >90% of observations (comfortable zone); daily trading falls in the gray zone. Decision: moving top-N from 10 to 20 keeps v1; moving to long/short requires v2 and a new baseline.
Label buffer: with 5-day forward labels and a decision on Tuesday Feb 10, the last eligible timestamp for training is Tuesday Feb 3. Other numbers: Palm/3Com 2000 as a case of prediction without a viable trade; passive ~35-40% of US market capitalization for reconstitution flows.

## Key Takeaways
- Define and version the trading setup before iterating; distinguish mechanics from parameters.
- Before modeling, answer family/cadence, PnL decomposition, edge narrative, and 2-3 failure modes.
- Run horizon feasibility against round-trip costs to decide cadence.
- Use walk-forward with a label buffer (and a feature buffer if applicable) in trading days; reserve a sealed holdout used once; consider nested walk-forward or CPCV for multiple variants.
- Start with a narrow baseline and three preflight sanity checks; if it fails, revise the setup instead of optimizing around it.
- Log every run (provenance, data/eval, config, artifacts, decision/gates) and use the Strategy/Trial family/Trial/Run taxonomy to count the search and deflate the Sharpe.
- Pre-register hypotheses to separate confirmatory from exploratory work.

## Connects To
Ch 1 (scoping invariants, evidence boundary), Ch 2/4 (PIT and universes), Ch 3 (intraday timing and execution), Ch 5 (Deflated Sharpe, multiple testing), Ch 7 (labels, IC, diagnostics), Ch 8-10 (features), Ch 11 (conformal prediction), Ch 16 (Rademacher Anti-Serum, Section 16.7), Ch 17 (CPCV, strategy metrics), Ch 22-24 (LLM-generated hypotheses), Ch 25-26 (live, MLOps, Section 26.6).

## Code Reference

Official repository: [06_strategy_definition/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/06_strategy_definition). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

This chapter has no notebooks; only `README.md` in the folder.

Other notebooks in the folder, not linked from the README: `01_where_ideas_come_from.ipynb`, `02_cv_foundations.ipynb`, `03_case_study_overview.ipynb`.
