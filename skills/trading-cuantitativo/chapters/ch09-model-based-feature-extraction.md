# Chapter 9: Model-Based Feature Extraction

## Core Idea
Model-based features are the outputs of fitted procedures (filtered states, innovations, conditional variance, regime probabilities, uncertainty) that summarize hidden structure better than retrospective formulas; every estimate must be point-in-time, re-estimated within the walk-forward protocol, and versioned together with the feature.

## Frameworks Introduced
- **Fitted-object discipline**: fit the procedure only on the training window; use *filtered* outputs (not smoothed); refit cadence aligned with the walk-forward; version parameters and training window. *Why it works*: prevents look-ahead in any estimate. *Failure mode*: using smoothed probabilities, model-order searches over the full sample, or non-causal offline wavelet coefficients.
- **Diagnostics-as-features (9.1)**: *ADF* (H0: unit root) + *KPSS* (H0: stationarity). Joint reading: ADF rejects and KPSS does not → stationarity; ADF does not reject and KPSS rejects → non-stationary; both reject → conflicting signals (deterministic trend, breaks, changing vol); neither rejects → inconclusive (short window, high persistence, low power). They are computed over rolling windows, and the lagged statistics/p-values are stability features.
- **Structural break features**: Zivot-Andrews (one endogenous break), Bai-Perron (multiple breaks), CUSUM (online monitoring), a supervised classifier that combines weak signals. Outputs as features: break dates, time since the last break, pre/post means, CUSUM, classifier probability.
- **Fractional differencing (FFD)**: (1−L)^d with real d in (0,1) (ARFIMA, long memory for 0<d<0.5); weights decay slowly, preserving more memory than d=1. *How*: (1) evaluate a bounded grid of d on the training window; (2) compute ADF for each transformed series; (3) measure retained correlation with the original; choose the smallest d with an acceptable stationarity diagnostic (not a full-sample "optimal" d). Convention: full-window (warmup + validity mask) vs boundary-partial (keeps rows, shorter filter at the start; ml4t.engineer.features.fdiff.ffdiff); the choice is part of the feature definition.
- **Kalman filter (local linear trend)**: x_t = F x_{t-1} + w_t, y_t = H x_t + v_t; predict + update; the gain adapts to the signal-to-noise ratio, with no fixed lookback. Outputs: *level* (adaptive anchor), *trend/slope* (lookback-free momentum), *innovation* (surprise), *state uncertainty* (confidence). Extends to dynamic hedge ratios (β_t). Failure: none; be careful with MLE fitting of covariances inside the walk-forward.
- **Spectral features**: FFT/Welch over a trailing window [t−L+1, t] (causal). Features: spectral energy, dominant period, spectral entropy (low = a few strong cycles), low-frequency ratio (energy below a cutoff, e.g. 1/21). Useful windows 21/63/126 days on returns. They complement deterministic calendars by asking whether the cycle actually exists.
- **Wavelets**: multi-resolution decomposition (detail bands from 2-4 days to 32-64 days + slow approximation); localized in time. They are an exploratory tool (non-causal offline): use them to discover horizons and then build a causal rolling proxy at that scale.
- **Path signatures**: iterated integrals; depth-1 = net displacement, depth-2 = interactions/lead-lag; time augmentation distinguishes "rose early and faded" from "fell and recovered"; log-signatures are more compact. Start with a depth-2 log-signature on daily data; only if path shape matters.
- **Volatility models as feature extractors (9.3)**: ARIMA (residual and one-step forecast; low orders such as ARIMA(1,0,1)/(2,0,1); order search within each training window); GARCH(1,1) σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} (cond_vol, shock_impact=α, vol_persistence=β, half-life = ln(0.5)/ln(α+β), long_run_vol = ω/(1−α−β) only if α+β<1; if ≥1 it behaves like IGARCH, flag it); EGARCH (log-variance; γ<0 = leverage_effect) and GJR-GARCH (negative-shock indicator); HAR (Corsi 2009): RV_{t+1} on daily, weekly, monthly RV, β_d/β_w/β_m measure each horizon's contribution (with intraday RV; if only OHLC, Garman-Klass/Yang-Zhang proxies).
- **Re-estimation cadence**: GARCH and EGARCH weekly/monthly; HAR on a rolling schedule (more regime-sensitive); SV via MCMC monthly or less; between refits, update the online recursions (cond. variance, RV inputs, SV filters). Large panels: re-estimate overnight and update recursively; EWMA alternative (less interpretable).
- **Rough volatility / Hurst**: H=0.5 Brownian, >0.5 persistence in increments, <0.5 anti-persistence. Slow regime descriptor: 252-day rolling window refreshed weekly; daily jumps = estimation noise.
- **Uncertainty-as-feature**: posterior std, credible interval width, tail probabilities, forecast SE, prediction interval width and its relative ratio (width/level). Distinguishes (a) low and well-identified vol, (b) high and well-identified, (c) high and poorly identified. Better as conditioning than as a signal. Stochastic volatility (Taylor 1982): posterior mean, posterior std, vol-of-vol, persistence; use filtered posteriors. ARIMA on log RV (asymmetric intervals when transformed back to the original scale).
- **Regime features (9.5)**: (a) observable rules (VIX above a threshold, price vs long MA, choppiness/efficiency): transparent, "fail visibly", a good baseline; (b) HMM: filtered probability P(S_t=k | information up to t) (not the smoothed one), transition matrix, expected duration 1/(1−p_k), entropy of the probabilities; relabel states by a stable characteristic (e.g. variance), keep few states (2 is usually more stable OOS); (c) MS-AR (Hamilton 1989): AR coefficients per regime, useful if persistence/direction change and not just vol; (d) distribution-based: K-means clustering with Wasserstein distance on rolling windows (assignment, distance to centroid, disagreement with the moment-based label).
- **Regime as conditioning, not hard switch**: pass filtered probabilities/entropy/duration as inputs to a single model; hard switches are unstable near transitions (Shu and Mulvey 2025: continuous probabilities outperform hard switching rules). Separate models per regime only with an economic rationale and sufficient data.
- **Panel features (9.6)**: mandatory order: (1) temporal feature per asset with trailing history → (2) align at the decision date → (3) rank/benchmark-adjust/pairwise/aggregate. Transformations: cross-sectional rank/percentile/z-score; benchmark-relative (momentum − market/sector momentum, vol/market vol); pairwise (Engle-Granger/Johansen cointegration, Kalman hedge ratio, spread z-score, half-life); universe-level (mean cond. vol, dispersion, breadth = fraction with elevated stress probability: universe_crisis_prob, crisis_breadth, regime_dispersion).

## Key Concepts
- **Filtered vs smoothed**: filtered uses only information up to t (deployable); smoothed uses the future (retrospective only).
- **Innovation**: Kalman prediction error; a surprise variable.
- **ACF/PACF/Q-Q**: total autocorrelation, partial autocorrelation, and distribution comparison (fat tails).
- **Vol shock half-life**: ln(0.5)/ln(α+β); SPY α≈0.17, β≈0.80 → α+β≈0.97, ~23 days.
- **Leverage effect**: γ<0 in EGARCH, negative shocks raise future vol more.
- **Hurst exponent H**: roughness/persistence; index log-vol ~0.1 (rough).
- **Expected duration**: 1/(1−p_k) for a Markov chain state.
- **Regime entropy**: classification uncertainty over the probability vector.
- **Wasserstein distance**: distance between distributions (in 1D, between quantile functions).
- **Time augmentation**: monotonic time coordinate added to the path for signatures.
- **Dominant period / spectral entropy / low-freq ratio**: spectral features.

## Mental Models
- Use **the model as a feature extractor, not as a predictor** when returns are hard to predict but vol clusters (GARCH/HAR perform where mean models do not).
- Use **uncertainty and regime as conditioning** features (gates/interactions), not as stand-alone signals.
- Use **wavelets to discover horizons** and **causal rolling proxies** for production.
- Use **deterministic regime rules as a baseline** (they fail in a diagnosable way) before HMMs.
- Use **continuous filtered probabilities** instead of hard labels when regime uncertainty is highest (transitions).
- Use **cross-sectional rank** for robustness, **z-score** if distance matters, **percentile** as a bounded compromise.

## Anti-patterns
- **Using smoothed probabilities or full-sample smoothed states**: leaks the future.
- **Choosing the fractional differencing d or the ARIMA order on the full sample**.
- **Deploying offline wavelet coefficients** (they depend on observations on both sides).
- **Interpreting long_run_vol when α+β ≥ 1** (unconditional variance is not finite).
- **Re-estimating HAR/GARCH daily and running MCMC frequently**: unnecessary cost and noise.
- **Interpreting daily jumps in Hurst as breaks**.
- **HMM states without relabeling across windows**; too many states (they fit historical noise).
- **Hard-switching between models by regime** (unstable at the edges).
- **Computing ranks before temporally aligning each asset** (leak across assets/time); a universe with changing composition without controls; later sector classifications instead of the ones in effect at the time.
- **Maximal feature catalog**: compact baseline (1-2 volatility, 1 regime, 1 signal-transform, a few cross-sectional transforms).
- **Treating a short half-life or cointegration as a trading rule**: they are state summaries.

## Code Examples
The chapter includes no code blocks (only references to notebooks 01-14). Reconstructed GARCH(1,1) sketch:
```
sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
half_life = log(0.5) / log(alpha + beta)
long_run_var = omega / (1 - alpha - beta)   # only if alpha + beta < 1
```
What it demonstrates: a single recursion yields cond_vol, shock_impact, vol_persistence, half-life, and long_run_vol.

## Reference Tables
Joint ADF/KPSS reading:

| ADF (H0 unit root) | KPSS (H0 stationary) | Reading |
|---|---|---|
| Rejects | Does not reject | Stationary |
| Does not reject | Rejects | Non-stationary |
| Rejects | Rejects | Conflict: deterministic trend, breaks, changing vol |
| Does not reject | Does not reject | Inconclusive (short window, low power) |

Methods by hidden structure: Kalman → latent state; Spectral/FFT → cycles; Wavelets → scales; Signatures → path shape; GARCH/HAR → vol persistence; HMM/MS-AR/Wasserstein → regime.

Refit cadence: GARCH/EGARCH weekly-monthly; HAR rolling (more sensitive); SV MCMC monthly or less often; online recursions between refits.

## Worked Example
SPY/ETF case: (1) Kalman local-linear-trend on daily returns: the slope has IC = −0.16 with the 5d forward return, ~9x larger than the 20d rolling OLS slope (−0.02); both negative (short-term reversal); more stable than 20d momentum in choppy markets; the innovation spikes around earnings/macro releases. (2) GARCH(1,1) on SPY: α≈0.17, β≈0.80, α+β≈0.97, half-life ≈ 23 days. (3) HAR on SPY: the weekly component dominates (β_w≈0.43); OOS RMSE 0.061 vs 0.111 for GARCH (~half). (4) SPY Hurst: returns H≈0.5; log-vol increments H between 0.12 and 0.23 (rough regime). (5) Wasserstein K-means on a synthetic 2-regime benchmark: adjusted Rand index 0.87 vs 0.31 for moment-based K-means. Decision: compact baseline + conditioning on filtered probabilities in a single gradient boosting model.

## Key Takeaways
- Every fitted feature lives inside the fold: re-estimate, version parameters and window, use filtered outputs.
- Run ADF+KPSS and treat their rolling statistics as stability features; prefer fractional differencing with the smallest acceptable d.
- For volatility, use GARCH/EGARCH/HAR as extractors (cond_vol, persistence, leverage) and match refit cadence to cost.
- Use uncertainty and regimes as continuous conditioning (probabilities, entropy), not hard switches.
- Wavelets and offline exploration are for discovery; produce causal proxies.
- In a panel: temporal per asset → align → rank/benchmark/pairwise/aggregate, with a controlled universe.
- Keep a compact, non-redundant set.

## Connects To
Ch 5 (stochastic volatility, MCMC), Ch 7 (triage, walk-forward protocol), Ch 8 (direct features, vol estimators, calendars; fractional differencing and stability), Ch 10 (text: timing/decay/embedding), Ch 13 (sequence models and learned embeddings), Ch 11-16 (pipeline and modeling), Ch 12 (tree-based).

## Code Reference

Official repository: [09_model_based_features/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/09_model_based_features). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_visual_diagnostics.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/01_visual_diagnostics.ipynb) + `.py` | 9.1 | The diagnostic sequence for a financial series, from plotting it through stationarity tests and autocorrelation to the rolling versions of the same statistics as feature… |
| [`02_structural_breaks.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/02_structural_breaks.ipynb) + `.py` | 9.1 | Where a series changes regime, found by the classical tests and by a change-point search, and how a break date becomes a column. |
| [`03_fractional_differencing.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/03_fractional_differencing.ipynb) + `.py` | 9.1 | Differencing by a fractional order, which reaches stationarity while keeping more of the memory than a first difference does. |
| [`04_kalman_filter.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/04_kalman_filter.ipynb) + `.py` | 9.2 | The filter as a feature extractor: level, trend, the innovation as a surprise measure, and a hedge ratio that moves. |
| [`05_spectral_features.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/05_spectral_features.ipynb) + `.py` | 9.2 | Frequency-domain features, from a wavelet decomposition through a rolling transform to a power spectral density estimate. |
| [`06_path_signatures.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/06_path_signatures.ipynb) + `.py` | 9.2 | The signature of a path as a feature set, and what its terms say that a set of moments does not. |
| [`07_arima_features.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/07_arima_features.ipynb) + `.py` | 9.3 | ARIMA read as a feature extractor rather than a forecaster: its residuals, its forecasts and the width of its intervals. |
| [`08_garch_volatility.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/08_garch_volatility.ipynb) + `.py` | 9.3 | Conditional volatility, the persistence parameters, and the asymmetry between a rise and a fall. |
| [`09_har_rough_volatility.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/09_har_rough_volatility.ipynb) + `.py` | 9.3 | Volatility at three horizons in one linear model, the range-based estimators it is built from, and the roughness exponent. |
| [`10_uncertainty_features.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/10_uncertainty_features.ipynb) + `.py` | 9.4 | A stochastic volatility model refit forward, the sampler diagnostics that decide whether its posterior is worth reading, and the forecast interval as a column. |
| [`11_hmm_regimes.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/11_hmm_regimes.ipynb) + `.py` | 9.5 | Regime inference from first principles, filtered against smoothed, measured against two rules that estimate nothing. |
| [`12_wasserstein_regimes.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/12_wasserstein_regimes.ipynb) + `.py` | 9.5 | Each window treated as a distribution and clustered under the Wasserstein distance, following Horvath et al. |
| [`13_regime_as_feature.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/13_regime_as_feature.ipynb) + `.py` | 9.5 | The regime probability as a column against a model per regime, with the regime model refit inside every fold. |
| [`14_panel_features.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/14_panel_features.ipynb) + `.py` | 9.6 | Cointegration and a filtered hedge ratio on one pair, then cross-sectional ranks, benchmark-relative features and universe aggregates on a panel. |
| [`case_study_temporal_summary.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/09_model_based_features/case_study_temporal_summary.ipynb) + `.py` | 9.7 | What the nine case studies' model-based feature stages wrote, read from the artifact schemas, and what a schema cannot tell you about it. |
