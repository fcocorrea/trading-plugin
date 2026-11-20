# Chapter 8: Financial Feature Engineering

## Core Idea
A feature is the computable expression of an economic hypothesis (point-in-time, with explicit lookback, aggregation, and reference frame) that must be aligned with the Ch 7 label horizon; the practical gain comes from combining signal features with state features and from controlling the degrees of freedom of the search.

## Frameworks Introduced
- **Three-step filter (hypothesis → driver → measurement)**: (1) *Horizon alignment*: lookback, cadence, and aggregation consistent with the label horizon and execution lag (very short horizons → microstructure/order flow; medium → trend, reversal, volatility state, event drift; slow → carry, term structure, fundamentals). (2) *Driver hypothesis*: persistence, reversion, risk compensation, or predictable clocks; if no plausible mechanism can be named, it is a data pattern, not a feature. (3) *Role separation*: signal (predicts mean/direction) vs state variable (conditions when the signal works, how aggressively to trade, cost/risk; acts through interaction rather than marginal association). *Why it works*: forces you to declare mechanism, failure modes, and horizon before testing. *Failure mode*: feature-label horizon misalignment, a common source of unstable estimates.
- **Box 8.1 – Patterns without a mechanism**: higher evidence standard: strict multiple-testing discipline, untouched OOS holdouts beyond the CV folds, robustness across instruments/regimes/variants, monitor post-deployment decay, and retire.
- **Three configuration choices**: *reference frame* (absolute time-series vs cross-sectional/peers), *representation* (raw, ranks, vol scaling, winsorization, residualization), *aggregation* (responsiveness vs variance reduction = effective bandwidth).
- **Feature specification template (Table 8.1)**: record name, family, role (signal/state), driver, edge source, inputs and observability, lookback and aggregation, reference frame, representation, failure modes. If these cannot be filled in, it is an input waiting for a hypothesis; it is equivalent to a feature registry entry. Each spec is a trial entering the Ch 7 workflow.
- **Meaning-changing vs noise-reduction choices**: meaning-changing (reference frame, anchor, peer set, estimator) create new hypotheses; noise-reduction (smoothing, shrinkage, winsorization) trade variance for bias.
- **Four questions per family**: economic claim and role; plausible horizons under the assumed execution; which choices change the hypothesis; how it fails in backtest/live.
- **Signal × state interactions (Table 8.4)**: *Gating* (trade only in a favorable state, otherwise flat), *Scaling* (continuous adjustment by state), *Conditional variant* (register "signal in regime" as a versioned feature). *How*: one signal and one state, vary one template at a time, each combination = a trial. Prioritize economically relevant states, asymmetric interaction (conditional IC differs across state values), and states observable at decision time.
- **Interaction diagnostics**: Structure (conditional IC varies systematically), Fold stability (no repeated flips), Real-time robustness (noisy/revised states hurt hard gates).
- **Event study (MacKinlay 1997)**: event date τ=0 set by a signal rule (only available information), normal-return model on a pre-event window, AR = R − E[R], CAR over [τ1,τ2], AAR/CAAR; inference with HAC or clustered SE. It is a falsification tool: correct shape in event time and robustness to the benchmark.
- **Degrees-of-freedom discipline**: budget per family and per role; one choice at a time (fix driver and representation, vary lookback, keep a few survivors); deduplicate within families first. *Dedup how*: hierarchical clustering with distance 1−|ρ| (cut at |ρ| ≈ 0.6-0.8 as a starting point), rank ρ per fold on the training window, one representative per cluster (highest median IC with low dispersion across folds, or the simplest); if there is a conflict, favor stability across folds.

## Key Concepts
- **Signal vs State variable**: a signal predicts the mean/direction; a state conditions when/how much to trade.
- **Time-series vs cross-sectional momentum**: the asset's own past vs relative performance within a peer set.
- **Anchor (reversion)**: level to which the price is expected to revert (MA, VWAP, peer mean, parity); defines the economic content.
- **Volatility estimators**: Parkinson, Garman-Klass, Yang-Zhang as range-based alternatives to close-to-close.
- **ATR / NATR**: typical range in price units; for stops, sizing, and gating, not a variance estimator.
- **Variance ratio / fractal efficiency**: <1 reversion, >1 persistence; net displacement / path length = trending vs choppy.
- **OFI (order-flow imbalance)**: cumulative signed volume over a window, normalized by volume, depth, or z-score; Kyle's λ = price impact per unit of flow.
- **Roll yield / carry**: return from holding a position with no price change (annualized difference between contracts of different maturities); backwardation = carry earned, contango = cost.
- **Funding rate (perps)**: periodic payments between longs and shorts that anchor the perpetual to spot; market-determined, not mechanical.
- **IV, VRP, risk reversal, skew**: implied volatility; implied-realized spread as a proxy for the variance risk premium; RR = 25Δ put IV − 25Δ call IV; OTM put-call skew.
- **Point-in-time (PIT)**: each value reflects only what was available at decision time (reporting lag, vintage, release timestamp).
- **Time-to-event**: periods until a scheduled event (capped and binned pre/event/post).

## Mental Models
- Use **cross-sectional rank** when the hypothesis is relative positioning; **time-series with vol scaling** when it concerns the asset's own dynamics.
- Use **range-based estimators (Yang-Zhang)** when overnight gaps are material; **Parkinson/Garman-Klass** if overnight is small; close-to-close only if only closes are available or for 24/7 cadences.
- Use **volatility features as state** (gating/scaling the signal) unless the label is a volatility label; then they are signals and are evaluated against a vol label.
- Use **cross-asset relationships as state** (e.g. 63d rolling SPY-TLT correlation) before using them as a direct signal; lead-lag requires strict validation (it may be asynchrony or bid-ask bounce).
- Use **treat funding as signal AND as state** (reversion at extremes; crowding that degrades directional signals) and register both roles.
- Use **effective sample size/uniqueness** when the feature is slow (quarterly book equity repeated for ~63 days multiplies nominal N by ~63).

## Anti-patterns
- **Technical indicator sprawl**: many are parameterizations of the same claim.
- **Feature without a nameable mechanism**: a data pattern, higher evidence standard.
- **Scanning many windows without selection correction**: search inflation.
- **Lookback misaligned with the label horizon**: too short = noise that decays; too long = smooths away the variation the label is after.
- **Treating funding as roll yield**: it can move 100 bps in hours; fails in exactly the transitions that matter. Check the interval per venue (8h typical; 4h/1h on some): 8h windows over 4h data mix settlements.
- **Fundamentals with backfilled revisions or quarterly values treated as fresh daily values**; assuming the reporting lag is constant (60 vs 90 days is a different feature).
- **Estimated earnings dates treated as confirmed**; using realized post-event amounts in pre-event windows.
- **Macro with revised data** instead of vintages (use ALFRED or equivalent); surprise expectations unstable across vendors.
- **Peer set changing mid-backtest**: changes the hypothesis.
- **IV-minus-RV with a realized leg that overlaps the label horizon** (look-ahead); a surface with a changing moneyness/quote policy.
- **Dense grids of lookbacks/transforms/interactions**: spurious winners; ratios without handling denominator ≈ 0 (clip vol, ratio = 0 or missing).
- **Using full-sample regime thresholds/terciles**: use expanding-window percentiles.
- **Relying on a single importance measure**: MDI and permutation importance agree on only 49% of rankings.

## Code Examples
The chapter contains no source code. Reconstructed key formulas:
```
Parkinson:     sigma^2 = (1/(4 ln2)) * (ln(H/L))^2
Garman-Klass:  sigma^2 = 0.5*(ln(H/L))^2 - (2 ln2 - 1)*(ln(C/O))^2
Rolling:       sigma_n^2 = mean(sigma^2_bar, n bars)
AR_{i,t} = R_{i,t} - E[R_{i,t} | normal model]
CAR(t1,t2) = sum_{t=t1..t2} AR_{i,t};  CAAR = mean of CAR across events
```
What it demonstrates: range-based estimators extract more information per bar than close-to-close; and the event study measures cumulative abnormal return relative to a benchmark.

## Reference Tables
Feature families (Table 8.2):

| Family | Economic claim | Role | Horizons |
|---|---|---|---|
| Trend/momentum | Recent performance persists | Signal | Hours to months |
| Reversal | Price reverts to an anchor | Signal | Seconds to days |
| Volatility/tail risk | Risk conditions sizing/gating | State | All |
| Liquidity/tradability | Costs and capacity condition signals | State | All |
| Microstructure/order flow | Informed flow moves prices | Signal/state | Seconds to hours |

Volatility estimators (Table 8.3):

| Estimator | Inputs | Efficiency vs close-to-close | Best when |
|---|---|---|---|
| Close-to-close | Close | 1x | Only closes are available |
| Parkinson | High, Low | ~5x | Daily bars, negligible overnight |
| Garman-Klass | OHLC | ~7x | Full OHLC, small gaps |
| Yang-Zhang | OHLC | ~8-14x | Material overnight gaps |

Use by case study: crypto perps close-to-close on 8h returns; ETFs/us_equities close-to-close + NATR; fx_pairs and sp500 options Garman-Klass; cme_futures Yang-Zhang; nasdaq100 microstructure intraday minute-level RV.

Options surface (5 features): ATM IV (constant 30d, regime state), implied-realized spread (VRP), 25Δ risk reversal, term-structure slope (inversion = imminent stress), OTM put-call skew.

Ratio interactions (Table 8.5): momentum/volatility (risk scaling), carry/implied_vol (risk-adjusted carry), momentum/bid-ask spread (cost-adjusted signal).

Spec template example (Table 8.1): mom_21d (Trend, Signal, Persistence, close, 21d, time series, vol-scaled return; fails: crowding/regime shift); rvol_20d (Volatility, State, risk state, annualized stdev; fails: stale in low-vol); carry_roll (Term structure, Signal, risk compensation, near/far futures, cross-section; fails: roll-date artifacts).

Contextual families: fundamentals (value, profitability, investment, quality; weeks to years), calendar (cyclical sin/cos or Fourier, time-to-event), macro (standardized 10Y-2Y yield-curve slope, credit/funding spreads, VIX percentile).

## Worked Example
ETFs: 126d momentum (cross-sectional rank) × volatility regime (SPY 42d RV, terciles from expanding-window percentiles: low <33, medium 33-67, high >67), label = 20d forward return, IC per tercile with HAC:

| Vol tercile | Momentum IC | HAC t | p |
|---|---|---|---|
| Low | +0.071 | 2.8 | 0.006 |
| Medium | +0.025 | 1.3 | 0.21 |
| High | -0.034 | -1.1 | 0.26 |

Monotonic decay (10.5 pp swing between low and high) → gating rule: take the signal only if vol is not in the top tercile. Search: 5 signals × 3 states × 3 templates = 45 interactions → report the number searched, BH-FDR, and treat best-in-search as exploratory until confirmed on held-out folds.
Other numbers: event study of momentum breakouts on ETFs (window -3 to +10): mean CAR 1.11% (t=3.28, p=0.001), no pre-event reversal; long signals CAAR 0.87% (t=3.30), short -0.25% (t=-1.31) → asymmetry. ETF feature selection: 57 candidates → 0.9 correlation filter removes 14 → IC ranking retains 18 → final 10 (82.5% reduction); best feature is distance to the 52-week low (IC 0.076); after BH-FDR, 4 of 34 significant at raw 5% do not survive. corr_spy_tlt_63d: highest normalized per-fold importance (0.73) among 27 LightGBM inputs. SPY→sector ETF lead-lag: lag-1 correlations from -0.06 to -0.13 (reversion, not continuation). Microstructure: contemporaneous OFI-return +0.053 vs lagged -0.011 (TSLA highest Kyle λ 0.003).

## Key Takeaways
- Write the full spec (family, role, driver, lookback, frame, representation, failure modes) before computing any feature.
- Align lookback and cadence with the label horizon, and test sensitivity to execution latency first in microstructure.
- Separate choices that change the hypothesis from those that reduce noise; change one at a time.
- Combine signal × state (gating/scaling/conditional) but count each interaction as a trial and control with BH-FDR.
- Deduplicate by cluster (|ρ| 0.6-0.8) within each family, choosing stability across folds.
- For fundamentals/macro/calendar, PIT is the binding constraint: lags, vintages, and confirmed dates.
- If the structure is hidden (shock persistence, latent regimes, cycles, path geometry), move to model-based features (Ch 9).

## Connects To
Ch 3 (OFI, order book, information bars), Ch 4 (PIT, COT, on-chain, vintages), Ch 6 (strategy narrative), Ch 7 (labels, triage, search accounting, uniqueness), Ch 9 (fractional differencing, GARCH, HMM, FFT, signatures; Sec 9.5 regimes), Ch 11-16 (modeling), Ch 13 (foundation models), Ch 14 (factor zoo).

## Code Reference

Official repository: [08_financial_features/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/08_financial_features). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_price_volume_features.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/01_price_volume_features.ipynb) + `.py` | 8.2 | This notebook demonstrates the core feature families derived from a single asset's price and volume history. |
| [`02_microstructure_features.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/02_microstructure_features.ipynb) + `.py` | 8.2 | Microstructure features capture market dynamics invisible in daily OHLCV data. |
| [`03_structural_cross_instrument_features.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/03_structural_cross_instrument_features.ipynb) + `.py` | 8.3 | This notebook demonstrates features that require data beyond a single asset's price series: term structures, cross-instrument relationships, and derivatives-implied quan… |
| [`04_fundamentals_macro_calendar.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/04_fundamentals_macro_calendar.ipynb) + `.py` | 8.4 | Slow-moving features that condition faster signals: SEC XBRL fundamentals (value/quality factors with point-in-time ASOF alignment), FRED macro indicators (yield curve,… |
| [`05_feature_selection.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/05_feature_selection.ipynb) + `.py` | 8.6 | A feature engineering pipeline produces many candidates — different lookbacks, transforms, and interaction variants. |
| [`06_robustness_sensitivity.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/06_robustness_sensitivity.ipynb) + `.py` | 8.6 | A robust signal maintains performance across reasonable variations in parameters, regimes, and implementation choices. |
| [`07_event_studies.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/07_event_studies.ipynb) + `.py` | 8.6 | Event studies measure abnormal returns around specific events (signal triggers, macro announcements, earnings) to assess their predictive power. |
| [`case_study_feature_summary.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/08_financial_features/case_study_feature_summary.ipynb) + `.py` | 8.6 | Cross-case-study feature inventory: feature counts per case study, family heatmap (momentum/volatility/return everywhere; carry on futures/FX; options-implied on the opt… |
