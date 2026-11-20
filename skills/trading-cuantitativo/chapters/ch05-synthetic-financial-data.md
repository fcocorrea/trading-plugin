# Chapter 5: Synthetic Financial Data

## Core Idea
History is a single realized path; synthetic data (samples from a fitted generative model) lets you evaluate robustness as a distribution, not as a point estimate. It is a decision-support tool, not a substitute for historical evidence, and its value is measured by the downstream task, not by looking realistic.

## Frameworks Introduced
- **Fidelity-Utility-Privacy (FUP) framework**: three criteria that trade off against each other. Fidelity = does it preserve the relevant structure (marginals, dependence, temporal dynamics)? Utility = does it support the intended use (TSTR vs TRTR, task-specific metrics)? Privacy = does it leak training information?
  - **When to use**: every synthetic experiment in finance; there is no universal score.
  - **How**: (1) fidelity: KS or Wasserstein per feature, histograms/ECDF/QQ, correlation/rank correlation matrices, stylized facts (heavy tails, weak autocorrelation in returns, volatility clustering via the ACF of r² or |r|, leverage effect, cross-asset dependence); (2) utility: TSTR vs TRTR with the same model class, features, target, and period; (3) privacy: exact/near-exact duplicates, nearest-neighbor distances train vs held-out, membership inference, DP-SGD with an ε budget; (4) real holdout.
  - **Why it works / failure mode**: validation matters more than architecture. Fails if judged by visual/marginal similarity, or if privacy is inferred from realism.
- **TSTR / TRTR**: train on synthetic only, evaluate on real; error ratio = error(TSTR)/error(TRTR). ~1 = preserves information; >1 = loss of utility; <1 may be suppression of idiosyncratic noise or leakage/narrow evaluation. Do not mix error ratios and score ratios without explaining the convention.
- **Classical baselines first**: bootstrap (IID, block ~22 days as a starting point, stationary with geometric blocks) and parametric (GBM, Merton jump-diffusion, Ornstein-Uhlenbeck, Heston, GARCH(1,1)); a deep generator must beat the appropriate baseline on the diagnostics that matter.
- **Generative taxonomy**: VAE (stable, tends to oversmooth), GAN (sharp, unstable, underrepresents tails), Diffusion (stable, conditional, slow sampling), tabular LLMs (heterogeneous, risk of invalid records and leakage).
- **Generator by use case**: TimeGAN (regular multivariate sequences, baseline), Tail-GAN (portfolio-level VaR/ES risk), Sig-CWGAN (path-wise fidelity, low dimension), GT-GAN (irregular series where timing is informative), Diffusion-TS (general purpose, conditional), GReaT (mixed tables).
- **Regime-conditional generation (classifier guidance)**: (1) fit a regime model (HMM) and label; (2) train a classifier on noisy sequences at different timesteps; (3) during generation, use the gradient of the target regime's log-prob to adjust denoising. Strong guidance reduces diversity and exaggerates extremes; use temperature scaling and stochastic sampling (DDIM with η).
  - **Why it works / failure mode**: the conditional generator inherits the regime detector's errors; validate with regime-specific diagnostics.
- **GReaT**: (1) serialize each row to column-value text (random column order); (2) fine-tune an autoregressive LLM (GPT-2 scale); (3) sample, parse, and filter out invalid rows with a constraint layer (type, range, logic).
- **Synthetic-specific failure modes**: bias amplification, overfitting to the generator, limited novelty (it interpolates, it does not invent unprecedented events).

## Key Concepts
- **Selection inflation**: the best in-sample result grows mechanically with the number of trials even with a true Sharpe of 0.
- **Deflated Sharpe Ratio (Bailey & López de Prado)**: adjusts inference for non-normality and number of trials (Ch 17).
- **Stylized facts**: heavy tails, volatility clustering, leverage effect, weak return autocorrelation.
- **Mode collapse**: the generator covers only a subset of regimes.
- **Path signature**: iterated integrals that summarize the shape of a path; invariant to time reparameterization (which is why time is added as a channel); terms grow as O(d^n).
- **DDPM**: forward process that adds noise; denoiser that predicts the noise (MSE); stochastic reverse chain.
- **Diffusion-TS**: encoder-decoder transformer denoiser with output = polynomial trend + seasonal (top-k Fourier) and a Fourier loss.
- **DP-SGD**: per-sample gradient clipping + calibrated noise; privacy summarized by ε (lower = more privacy, less fidelity).
- **Stationary bootstrap (Politis-Romano)**: a better default than fixed blocks for volatility clustering without a parametric model.
- **OU half-life**: ln(2)/κ.

## Mental Models
- Use **synthetic data as simulation infrastructure** when evidence is path-limited: robust parameter selection, regime stress, privacy-preserving development.
- Use **the generator as part of the modeling stack**: fix it before selecting a strategy, validate it on held-out real data, use an outer validation loop.
- Use **block/stationary bootstrap** when you need volatility clustering without a model; use **parametric models** when you need paths "beyond history" and accept the assumptions.
- Use **Tail-GAN** when the downstream task is risk measurement and the tail dominates; **Sig-CWGAN** if it is path-dependent and low-dimensional; **GT-GAN** only if observation times carry information.
- Use **utility and fidelity separately**: high utility with low fidelity is possible (GReaT: TSTR 92% with distinguishable marginals).

## Anti-patterns
- **Judging by "it looks realistic"**: can understate drawdowns, clustering, and correlations under stress.
- **Selecting strategies by performance on synthetic paths without a real holdout**: overfitting to the generator.
- **IID bootstrap for risk/sizing**: destroys volatility clustering.
- **GBM as a returns model**: no clustering and thin tails.
- **Applying TimeGAN with a sigmoid output to unbounded returns** without mapping to a range or using a linear output.
- **GT-GAN with artificial irregularity** (random subsampling).
- **Inferring privacy from fidelity/utility**; using synthetic data outside a controlled environment without leakage tests.
- **Ignoring rejection rates** in tabular generation (sustained >10-15% indicates serialization/preprocessing problems).
- **Comparing toy generators as if they were equivalent**: each was designed for a different task.

## Code Examples
GReaT serialization (text format of one row):
```
income is 85000, debt_ratio is 0.32, emp_length is 7, home_status is MORTGAGE, approved is 1
```
TSTR ratio for error metrics:
```
TSTR_ratio = error(train=synthetic, test=real) / error(train=real, test=real)
```
What it demonstrates: how a tabular row is turned into text for autoregressive fine-tuning and how utility is summarized with a comparable ratio.

## Reference Tables
| Generator | Use when | Key limitation |
|---|---|---|
| TimeGAN | Regular multivariate sequences, baseline | Does not target tails; regular grid; bounded output |
| Tail-GAN | Risk (portfolio-level VaR/ES) | VaR/ES on given portfolios does not guarantee other tails |
| Sig-CWGAN | Path-wise fidelity, low dimension | O(d^n) terms: 50 assets depth 3 > 125,000 |
| GT-GAN | Informative irregular timestamps | Complexity; oversmooths with little data |
| Diffusion-TS | General purpose, conditional | Slow sampling; underestimated variance |
| GReaT (LLM) | Mixed tables (credit, profiles, fundamentals) | Invalid records, numeric fidelity, leakage |

| ε budget (DP-GAN) | Mean Abs Diff | Corr. Distance | Interpretation |
|---|---|---|---|
| 1.0 | 8.02 | 0.12 | Strong privacy, poor utility |
| 5.0 | 2.40 | 0.15 | Good balance |
| 10.0 | 2.67 | 0.06 | Good utility |
| 50.0 | 1.68 | 0.11 | Best utility, weak privacy |

| Bootstrap | Preserves | Fails at |
|---|---|---|
| IID | Marginal | Temporal dependence, clustering |
| Block | Within-block dependence | Artificial boundaries; long persistence if block is short |
| Stationary | Dependence in expectation | Still limited to the historical record |

Minimum checklist: one fidelity diagnostic, one utility benchmark, one privacy check, and one result on a real holdout.

## Worked Example
Diffusion-TS on 20 ETFs (daily returns 2018-2025): KS 0.06 (marginals), correlation error 0.04 (cross-asset dependence), ACF error 0.05 (volatility persistence) = reasonable fidelity; TSTR on extreme-move classification ~ parity with TRTR (utility); privacy not established without DP or leakage tests. Conditional generation: low-vol regime 0.93x and high-vol 1.08x of historical volatility; ratio between regimes 2.1x. Caveat: uncalibrated samples underestimate variance by ~1/3 in normalized space.
Comparisons: TimeGAN (6 stocks, windows of 24): discriminative accuracy 68%, ROC AUC 0.87 (separable), TSTR 1.76 (76% more error) vs Diffusion-TS 1.00. Tail-GAN (5 ETFs, 32 long-short portfolios): relative VaR error 13%, ES 11%; synthetic VaR ≈ -10.33 vs real ≈ -9.13 (overstates loss). Sig-CWGAN (S&P 500 2005-2020, depth 4, 16-day windows, 2,500 steps): TSTR 0.954 but r² autocorrelation 0.05 vs 0.49 real (does not preserve clustering). GReaT (distilgpt2, 50 epochs, ~13 min on an RTX 3090): TSTR AUC 0.70 vs TRTR 0.74 (92%); only volume_ratio matches well (KS 0.10), the rest KS 0.30-0.59; mode collapse (direction=down 94% vs 45% real). Selection inflation (Bailey et al.): with 10 configurations the expected maximum in-sample Sharpe is 1.57 even with a true Sharpe of 0; with 100 trials >2.5.

## Key Takeaways
- Treat the backtest as an estimate conditioned on the search; synthetic data expands paths but does not eliminate selection bias.
- Start with classical baselines (stationary bootstrap, GARCH) and require the deep model to beat them.
- Choose the generator by use case (tail, path, irregular, tabular), not by "the best."
- Always report fidelity, utility (TSTR/TRTR), privacy, and the real holdout.
- Validate finalists on real data not used to train the generator.
- If you need privacy guarantees, use DP-SGD (ε between ~5 and 10 as a practical range in the example) and leakage tests.

## Connects To
Ch 1 (evidence boundary, multiple testing), Ch 3 (OHLCV/dollar bars as input), Ch 4 (fundamentals with PIT checks in generation), Ch 9 (GARCH, HMM, regimes), Ch 17 (Deflated Sharpe, backtest overfitting), Ch 6 (research framework).

## Code Reference

Official repository: [05_synthetic_data/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/05_synthetic_data). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`00_classical_simulation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/00_classical_simulation.ipynb) + `.py` | 5.3 | Simulates return paths from GBM, jump diffusion, mean reversion, Heston and GARCH, and resamples real returns with the IID, block and stationary bootstraps, then measure… |
| [`01_timegan.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/01_timegan.ipynb) + `.py` | 5.5 | This notebook implements TimeGAN (Yoon, Jarrett & van der Schaar, NeurIPS 2019), the foundational architecture for synthetic financial time series generation. |
| [`02_tailgan_tail_risk.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/02_tailgan_tail_risk.ipynb) + `.py` | 5.5 | This notebook implements Tail-GAN (Cont, Xu, and Zhang 2022), a GAN architecture that uses differentiable sorting to preserve tail risk characteristics (VaR, ES) in synt… |
| [`03_sigcwgan_signatures.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/03_sigcwgan_signatures.ipynb) + `.py` | 5.5 | Builds the unconditional Sig-Wasserstein GAN, which replaces the learned discriminator with an analytic distance between expected path signatures, and explains how the c… |
| [`04_gtgan_irregular.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/04_gtgan_irregular.ipynb) + `.py` | 5.5 | This notebook implements a GT-GAN-inspired model (based on Jeon et al., NeurIPS 2022) using Neural ODEs to handle time series with naturally irregular timestamps. |
| [`05_diffusion_ts.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/05_diffusion_ts.ipynb) + `.py` | 5.6 | This notebook implements Diffusion-TS (Yuan & Qiao, ICLR 2024), a diffusion model that decomposes the denoising prediction into trend (polynomial regression) and seasona… |
| [`06_llm_tabular_great.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/06_llm_tabular_great.ipynb) + `.py` | 5.7 | This notebook implements GReaT (Generate Realistic Tabular Data) using the actual be-great library to generate synthetic financial tabular data with LLMs. |
| [`07_dp_gan.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/05_synthetic_data/07_dp_gan.ipynb) + `.py` | 5.8 | This notebook demonstrates Differential Privacy (DP) for training generative models using Opacus, PyTorch's official DP library. |
