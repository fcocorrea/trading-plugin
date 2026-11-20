# Chapter 13: Deep Learning for Time Series

## Core Idea
Deep learning for time series is a competition among inductive biases (recurrence, decomposition, attention, convolutions, SSMs, linear baselines); no architecture wins by default. The central conclusion: architecture matters only after the target is properly defined (recursive, multi-horizon, horizon-label or direct allocation), and every model must justify its complexity against a single-layer linear baseline and against GBMs.

## Frameworks Introduced
- **Three forecast formulations (Fig 13.4)**: (1) recursive one-step: a 1-step model rolled forward, errors accumulate; (2) direct multi-horizon path: predicts the whole trajectory in one forward pass; (3) direct horizon-label: predicts only the investment label (5d/21d return, direction, rank). A fourth (Ch17): learn positions/weights with an allocation loss (Sharpe, drawdown, costs).
  - When to use: (3) for cross-sectional ranking evaluated with Spearman IC; (2) when the trajectory matters (risk); (4) when the objective is economic.
  - How: define the target before choosing the architecture; align the objective with the evaluation metric.
  - Why it works / failure mode: (3) matches the metric and avoids compounding; on ETFs 21d and CME 5d, N-BEATS/TSMixer via Darts (compounded path) gave a substantially weaker signal than direct LSTM/NLinear.
- **Baseline ladder**: (1) seasonal naive (does not apply to returns); (2) LTSF-Linear (D-Linear / N-Linear); (3) ARIMA/ETS (univariate) or VAR (multivariate); (4) LightGBM/XGBoost with lags, rolling stats and calendar features.
  - When to use: before any DL architecture.
  - How: if the sophisticated model does not beat D-Linear, its complexity is not justified; test whether temporal ordering adds signal over lag-features; keep the model simple unless OOS gains are robust; walk-forward (Ch6) with bootstrap intervals and IC/Sharpe/max drawdown.
  - Failure mode: skipping non-deep baselines inflates perceived gains.
- **LTSF-Linear (Zeng et al. 2022)**: Linear (one lookback->horizon matrix); D-Linear (trend/seasonal decomposition via moving average + one linear layer per component, summed); N-Linear (subtract the last input value, linear layer, add it back; stabilizes distribution shift).
  - Evidence: beat Transformers on 9 LTSF datasets, 20-50% improvements over Informer/Autoformer/FEDformer; Shuffle experiment: D-Linear MSE 0.345 -> 1.406 when shuffled, Transformer 0.379 -> 0.391 (attention was not learning temporal patterns); Transformer error grew with longer lookback.
  - Caveat: univariate study; carefully tuned Transformers or multivariate ones with covariates can match; rankings flip with small changes (budget, splits, metric).
- **N-BEATS (Oreshkin 2019)**: FC blocks that emit a forecast and a backcast; residual stacking (input - backcast passes to the next block); doubly residual (each stack receives the residual of the previous stack; final forecast = sum of all blocks); basis expansion (coefficients θ over basis vectors g). N-BEATS-I: Trend Stack with a degree-p polynomial basis, Seasonality Stack with a Fourier basis; Forecast = Trend + Seasonality. Extensions: N-BEATSx (exogenous), N-HiTS (hierarchical interpolation, MaxPool, multi-rate).
  - When to use: univariate settings where interpretability is wanted (N-BEATS-I).
  - Failure mode: sensitive to the lookback/horizon ratio; N-BEATS-I can lose to N-BEATS-G if the process does not decompose into polynomial+Fourier (regime changes); benchmark both.
- **Modern Transformers**: PatchTST (patching: L/P non-overlapping patches, complexity O((L/P)^2); channel-independence as a regularizer; RevIN); iTransformer (variate-as-token: N tokens, attention across variables, sidesteps the temporal-order critique); TFT (variable selection via gating, static/observed/known-future covariates, multi-horizon quantile output).
  - Example: lookback 512 with patches of 16 -> 32 tokens.
  - Failure mode: leakage in patching if normalization uses statistics from the whole window (normalize per patch or use only backward-looking statistics).
- **SSMs / Mamba**: latent state h_t = A h_{t-1} + B x_t, y_t = C h_t (+ D); Mamba makes B_t, C_t and the step Δ_t input-dependent (selective state space), complexity O(L). ms-Mamba: parallel Mamba blocks at different sampling rates.
  - When to use: sequences >10,000 steps (tick/minute bars).
- **Adapting foundation models (TSFM)**: (1) zero-shot; (2) in-context learning (Chronos-2 group attention); (3) PEFT/LoRA (progressive training that randomly deactivates adapters; prune-then-finetune up to 90% of parameters); (4) test-time model selection/ensembling. Ladder: zero-shot -> ICL -> PEFT/LoRA -> full domain training.
  - Why: FMs are initializations, not solutions; in finance the transfer gap is distributional (heavy tails, low SNR, non-stationarity vs electricity/weather/traffic).
  - Failure mode: leakage via the pretraining corpus; verify that evaluation data postdates the corpus or is excluded from it (TIME benchmark with "fresh datasets").
- **Uncertainty: MC Dropout and Deep Ensembles**: MC dropout = dropout active at inference, 50-100 forward passes, mean = forecast, std = uncertainty; ensembles = 3-10 models with different initializations (~5 is a good compromise), more robust but linear cost. Variant: heteroscedastic head (mean + variance).
  - How: validate with empirical coverage, width, sharpness, reliability plots; if coverage is poor, split-conformal over a held-out window (Mondrian/group-conditional if it fails by group; multi-fold if split noise dominates).
  - Result (nb 10): uncalibrated, Gaussian intervals under-covered by ~an order of magnitude at 95% nominal; after split-conformal, coverage within ~3 points of nominal (95% -> 0.97 for MC dropout and for the 5-LSTM ensemble).
- **Architecture selection (Steps 1-3)**: baselines -> diagnosis (uni/multivariate, known/unknown relationships, horizon, interpretability, covariates, length) -> Table 13.2 matrix.

## Key Concepts
- **LSTM gates**: forget/input/output regulate the cell state C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W·[h_{t-1}, x_t] + b).
- **GRU**: simplified LSTM (single update gate, no cell state), ~1/4 fewer parameters; inherits O(L) sequential computation.
- **Patching / RevIN**: tokenization by fixed segments / reversible instance normalization (subtract mean, divide by std, revert at the output).
- **Channel-independence**: each series is processed separately with shared weights; regularizes against spurious correlations.
- **Backcast**: reconstruction of the input by an N-BEATS block; its residual feeds the next block.
- **Permutation invariance**: attention without positional encoding treats the sequence as a set; order must be injected.
- **GAF (Gramian Angular Field)**: transforms series into image matrices for CNNs.
- **Hybrid stat-neural (ES-RNN)**: Holt-Winters removes non-stationary structure and the RNN learns corrections; M4 winner.
- **Diebold-Mariano / Giacomini-White**: tests of equal average predictive accuracy / conditional predictive ability.
- **Finance transfer gap**: drop in TSFM performance when moving from general benchmarks to financial returns.
- **Pretraining contamination**: the pretraining corpus as a new leakage channel.

## Mental Models
- Use a linear baseline (N-Linear/Ridge) and LightGBM first; only move up to complex architectures if temporal ordering adds signal over lag-features.
- Use direct horizon-label (LSTM/NLinear) when the metric is cross-sectional IC; use path forecasters (N-BEATS, TSMixer, PatchTST) when the trajectory matters.
- Use fine-tuned TSFMs for volatility/VaR (transferable structure: clustering, persistence) and GBMs with features for returns.
- Use iTransformer with >50 series and cross-asset dynamics; PatchTST for long horizons or unknown dependencies; TFT with heterogeneous covariates; Mamba with >10K steps.
- Use ensembles (~5) + conformal when sizing depends on uncertainty; MC dropout if it must be added cheaply to an existing model.

## Anti-patterns
- **"Using Transformers by default"**: Zeng et al. showed that a linear layer beats them; new models must beat LTSF-Linear.
- **Trusting long nominal lookbacks**: more lookback increases noise and instability; weak/non-stationary signal means effective use matters, not theoretical capacity.
- **Ranking architectures with a single split**: on ETFs, TCN went from IC +0.051 to -0.013 between reruns of the same code; PatchTST vs iTransformer swap order; non-deterministic cuDNN is large relative to the IC scale.
- **Interpretable N-BEATS-I without comparing to generic**: the decomposition may not fit financial data.
- **Normalizing with the whole window in patching**: leakage.
- **Zero-shot TSFM for returns**: Chronos large OOS R^2 -1.37% and TimesFM 500M -2.80% (US, 512d window) vs CatBoost -0.03%; direction ~50%.
- **Evaluating TSFMs without checking overlap with the pretraining corpus.**
- **SHAP/LIME on recurrent/attention models**: they ignore sequential dependencies; attributions are unreliable.
- **More data => replace the model**: a large sample size should trigger a comparison, not a replacement.

## Code Examples
The extracted text contains no code blocks (only references to notebooks 01_core_architectures ... 12_dl_weekly). Minimal reconstruction of N-Linear as described by the author:

```
last = x[:, -1:]
y = linear(x - last) + last
```
What it demonstrates: normalizing by subtracting the last value and adding it back stabilizes predictions under distribution shifts between train and test.

## Reference Tables
Table 13.1 (Transformers):

| Model | Innovation | Attention domain | Best for |
|---|---|---|---|
| Vanilla Transformer | Self-attention | Time steps | Short sequences, baseline |
| TFT | Variable selection + covariates | Time steps | Covariate-rich, multi-horizon with uncertainty |
| PatchTST | Patching + channel-independence | Patches | Long LTSF, uni/multivariate |
| iTransformer | Variate-as-token | Variables | Cross-asset dynamics, portfolios |

Table 13.2 (selection; primary / alternative / baseline): uni+interpretable: N-BEATS-I / Autoformer / Seasonal ARIMA; uni black-box: PatchTST / N-BEATS generic / D-Linear; multi with known graph: GNN (Ch23) / - / VAR; multi unknown correlations: PatchTST / iTransformer / D-Linear; >50 series: iTransformer / PatchTST / Linear; long horizon: PatchTST / D-Linear / Naive Repeat; covariate-rich: TFT / PatchTST / LightGBM; >10K steps: Mamba / TCN / rolling stats + GBM; limited data: ARIMA or LightGBM / N-Linear / Seasonal Naive; zero adaptation: D-Linear / TSFM zero-shot / Seasonal Naive; moderate adaptation: PatchTST+LoRA / TSFM ensemble / LightGBM.

Sample-size thresholds (daily, moderate SNR): <5,000 -> Ridge or LightGBM; 5,000-20,000 -> LightGBM typically better accuracy/compute; >20,000 -> try LSTM/DL at scale, no guarantee.

Table 13.3 (libraries): Raw PyTorch (full control, custom losses, cross-sectional; ideal for panel ranking with IC); Darts (broad coverage, unified fit/predict API; TimesFM); NeuralForecast (native panel with unique_id, fast); PyTorch Forecasting (TFT, covariates; TimeSeriesDataSet); GluonTS/AutoGluon (Chronos-2, probabilistic); sktime (composable pipelines, wraps backends).

Table 13.4 (best DL vs best tabular, IC; Δ): ETFs 21d NLinear +0.062 (t 3.0) vs Ridge +0.054 (+0.009); Crypto 8h NLinear +0.029 (t 4.6) vs GBM +0.011 (+0.018); SP500 Options PatchTST +0.013 (1.8) vs GBM +0.018 (-0.005); FX 1d NLinear +0.011 (1.3) vs TabM +0.007 (+0.004); SP500 Eq+Opt 5d PatchTST +0.011 (1.1) vs TabM +0.011 (~0); US Equities 1d LSTM +0.007 (5.4) vs GBM +0.032 (-0.025); NASDAQ-100 15m NLinear +0.005 (2.3) vs GBM +0.006 (-0.001); CME 5d LSTM -0.001 (-0.1) vs GBM +0.032 (-0.033).

Table 13.5 (US Equities weekly, mean IC over 4 folds): direct NLinear +0.018; direct LSTM +0.006; N-BEATS (Darts) one-step -0.013.

## Worked Example
Cross-case study (Sec 13.9): 8 of 9 case studies (excluding US Firms), primary label, daily Spearman IC with 95% HAC CI; paired day-by-day comparison (delta DL - best tabular) with HAC on the difference.
- The DL interval excludes zero in 4 (ETFs, Crypto, US Equities, NASDAQ-100); the delta excludes zero in 3: Crypto (+0.018 in favor of DL), US Equities (-0.025) and CME (-0.033) against. The remaining five are indistinguishable from the best tabular model, at higher training cost.
- Who wins: NLinear leads DL in 4 (ETFs, Crypto, NASDAQ-100, FX), LSTM in 2 (CME, US Equities), PatchTST in 2 (SP500 Options, SP500 Eq+Opt), TCN in none; N-BEATS/TSMixer/PatchTST/iTransformer have path/cross-variate biases misaligned with a rank label.
- Training trajectories: at high frequency (NASDAQ-100, SP500 Options, Crypto) IC peaks early and decays; in daily/longer ones (FX, CME, US Equities) it rises with training -> early stopping per case study.
- Uncertainty: split-conformal coverage at 90% nominal ~0.90 on CME, ETFs, US Equities; SP500 over-covers; Crypto and FX under-cover; NASDAQ-100 near zero. Credible IC + coverage close to nominal (ETFs, US Equities) is a better operational position than credible IC with unreliable coverage (Crypto, NASDAQ-100).
- Weekly experiment on US Equities (Table 13.5): even with compounding removed, direct regression > one-step forecasting and both < daily GBM (+0.032). Conclusion: cross-sectional ranking is direct prediction, not path forecasting. Exception: for sizing with Sharpe/downside/costs, Saly-Kaufmann et al. (2026) find hybrid recurrent architectures with feature selection more competitive.
- Foundation models on ETFs (nb 09): Chronos IC -0.015 (~20M params), TTM -0.017 (1-5M), 50K-param LSTM +0.025, 60-param Ridge +0.022. With financial pretraining (Rahimikia 2025): Chronos small 36.84% annual and Sharpe 5.42; TimesFM 20M 30.36% and 3.66 (512d window).

## Key Takeaways
- Start with Ridge and LightGBM; use DL only if temporal ordering adds robust OOS signal; only 1 of 8 case studies showed DL clearly better.
- Define the target (horizon-label vs path vs allocation) before choosing the architecture.
- Include N-Linear/D-Linear as a mandatory floor and report IC with HAC and paired deltas.
- Do not rank architectures with a single split; use walk-forward and multiple seeds.
- For risk (vol, VaR) consider fine-tuned TSFMs with Student-t/mixture distributions and conformal calibration; for returns, trees with features.
- Quantify uncertainty with a ~5-model ensemble + split-conformal and validate coverage before using it in sizing.
- Verify pretrain/test separation in any TSFM evaluation.

## Connects To
Ch9 (ARIMA, GARCH, Kalman, long memory), Ch10 (self-attention), Ch11 (linear baselines, SHAP/LIME, conformal), Ch12 (GBMs, TabM, TabPFN), Ch6 (walk-forward), Ch14 (latent factor models), Ch16-19 (simulation, turnover, costs), Ch17 (learning direct allocation with Sharpe/drawdown), Ch23 (GNNs for known relationship structure).

## Code Reference

Official repository: [13_dl_time_series/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/13_dl_time_series). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_core_architectures.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/01_core_architectures.ipynb) + `.py` | 13.1 | Compares an MLP, a 1D-CNN, an LSTM and a GRU on the same panel, under a purged three-way split anchored on each example's target date. |
| [`02_nbeats_interpretable.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/02_nbeats_interpretable.ipynb) + `.py` | 13.2 | Builds N-BEATS from scratch in PyTorch in both its generic and interpretable configurations, and reads the decomposition for what it is - what a block was *allowed* to s… |
| [`03_great_debate.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/03_great_debate.ipynb) + `.py` | 13.4 | Puts Linear, D-Linear and N-Linear against a vanilla transformer on daily ETF returns, and runs the shuffle diagnostic that made the original critique sharp - measuring… |
| [`04_transformers.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/04_transformers.ipynb) + `.py` | 13.5 | PatchTST makes a token a short run of days; iTransformer makes a token a whole feature's history. |
| [`05_tcn.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/05_tcn.ipynb) + `.py` | 13.6 | A temporal convolutional network: causal convolution by padding and trimming, dilation doubling per layer, residual blocks. |
| [`06_tsmixer.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/06_tsmixer.ipynb) + `.py` | 13.6 | Drops attention and convolution and keeps only dense layers, applied along one axis at a time. |
| [`07_mamba_ssm.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/07_mamba_ssm.ipynb) + `.py` | 13.6 | A selective state space model written as a readable Python loop, with its four departures from the reference implementation named and locatable in the code. |
| [`08_cnn_image_encoding.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/08_cnn_image_encoding.ipynb) + `.py` | 13.6 | Turns each window into a Gramian angular field and a Markov transition field and hands the pair to an image CNN. |
| [`09_foundation_models.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/09_foundation_models.ipynb) + `.py` | 13.6 | Runs Chronos and TinyTimeMixer zero-shot against an LSTM and a ridge fitted on the panel, all four reading the same univariate context. |
| [`10_uncertainty.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/10_uncertainty.ipynb) + `.py` | 13.8 | MC Dropout and deep ensembles produce a spread; neither produces a coverage property, which the empirical coverage table makes visible. |
| [`11_library_landscape.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/11_library_landscape.ipynb) + `.py` | 13.7 | Writes the same univariate forecast through raw PyTorch, sktime and Darts, and compares the implementation experience. |
| [`12_case_study_insights.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/13_dl_time_series/12_case_study_insights.ipynb) + `.py` | 13.7 | Reads the registry rather than training anything: LSTM, NLinear, TSMixer, TCN and PatchTST across whichever case studies carry deep-learning pipelines, against the linea… |
