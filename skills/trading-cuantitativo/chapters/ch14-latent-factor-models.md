# Chapter 14: Latent Factor Models

## Core Idea
Latent factor models summarize high-dimensional return variation in a few factors, and each method optimizes a different object (variance, variance+pricing error, conditional betas, no-arbitrage). A factor that explains covariance (attribution factor) is not necessarily a priced factor; choose the method according to the economic objective and evaluate with walk-forward.

## Frameworks Introduced
- **Three estimation objectives**: (a) maximize explained variance (PCA); (b) variance + pricing error (RP-PCA); (c) directly minimize pricing error (adversarial SDF). IPCA/CAE are conditional latent factor models with characteristic-dependent loadings.
  - When to use: PCA for risk decomposition; RP-PCA to discover priced factors; IPCA/CAE for conditional betas; SDF for direct pricing.
  - How: first decide the object (risk, pricing, prediction), then the method.
  - Failure mode: treating a high-variance factor as if it carried a risk premium.
- **Three-stage latent-factor forecasting adapter**: Stage 1 estimate factors F and the loading map from returns (SVD in PCA; penalty in RP-PCA; β(z) in IPCA/CAE); Stage 2 forecast the factor premium λ_hat from the F history; Stage 3 map μ_hat_{t+1} = β_t' · λ_hat. Implementation: LatentFactorForecastPipeline(model, forecaster, mapper) in ml4t-models.
  - When to use: turning IPCA, RP-PCA, CAE (and optionally PCA) into ex-ante asset-level forecasts.
  - How: Stage 2 baseline = training mean (ExpandingMeanFactorForecaster, "IID-BS"); forecaster ladder Tier 1: Constant/AR(1)/EWMA; Tier 2: Ridge/LightGBM (analogous to Q-Boost); Tier 3: LSTM/TCN (and ZS-Chronos). Stages 1 and 2 are separable.
  - Why it works / failure mode: premium forecast quality enters in Stage 2 and translates directly into signal; a CAE can reconstruct well and forecast poorly if the factors have no persistent premium. The SDF and the SAE are outside the adapter.
- **PCA / eigendecomposition**: solve Σv = λv on demeaned returns; explained variance = λ_i/Σλ; scree plot with an "elbow"; eigenvectors = weights of orthogonal eigenportfolios.
  - When to use: risk decomposition, eigenportfolios, yield curve.
  - How: choose correlation (equity cross-section) vs covariance (multi-asset with differing volatilities); consider normalizing by lagged idiosyncratic volatility (estimated from residuals over a prior window, no lookahead); apply shrinkage (Ledoit-Wolf) before PCA if N/T is large.
  - Failure mode: second-order moments only and linear; with N close to T the eigenvalues inflate and the eigenvectors are unstable.
- **IPCA (Kelly-Pruitt-Su 2019)**: r_{t+1} = Z_t Γ f_{t+1} + ε, β_t = Γ' z_t with lagged z_t of dimension P, Γ of dimension K×P (per the text), f latent factors. "Characteristics are covariances". Estimation by Alternating Least Squares (ALS).
  - When to use: interpretable conditional betas and a risk vs mispricing test (if the characteristic operates through risk, alphas are insignificant; if it is mispricing, alpha is significant).
  - How: lagged characteristics; impute missing values with the cross-sectional median (missingness is informative); report a K range of 3-8, not an optimum; KPS uses 94 characteristics, the book's dataset 57.
  - Failure mode: the linear map does not capture interactions; missing values are absorbed into residuals; sensitive to K and to the set of test assets.
- **RP-PCA (Lettau-Pelger 2020)**: PCA on the covariance plus a penalty that overweights mean-return information; parameter κ (gamma in the notebook, with κ = 1+γ; gamma=0 reproduces PCA).
  - When to use: low-variance but high-premium factors (e.g. credit: 2% of variance, Sharpe 0.8, rank >15 in PCA, rises to 3rd with high κ); OOS Sharpe more than double PCA's in US equities 1963-2017 (sample-dependent).
  - Failure mode: κ->∞ ignores variance entirely; test assets are a design decision (compare on a common set and report sensitivity).
- **Conditional Autoencoder (CAE; Gu-Kelly-Xiu)**: r_{t+1} = β(z_t)' f_{t+1} + ε with β(z)=neural network (strict generalization of IPCA; a single linear layer without activation = IPCA). Dual-network: beta network + factor side (x_t = (Z'Z)^-1 Z' r projection onto managed portfolios, f_t = h(x_t)); cross-sectional reconstruction loss with L1 and early stopping; seed ensemble.
  - When to use: nonlinear interactions among characteristics (momentum conditional on volatility).
  - How: rank + rescale characteristics to [-1,1] per month; pyramidal beta network with ReLU and batch norm (e.g. 64->32->16->8); single linear layer factor network; K 5-10.
  - Failure mode: reconstruction loss is not forecast quality; divergence from a high LR; correlated factors (large K / little regularization); beta-network collapse (nearly identical betas across assets).
- **Adversarial SDF (Chen-Pelger-Zhu 2021)**: E[M_{t+1} r_{t+1}] = 1 (or E[M r^e]=0), M parameterized with a network over characteristics + macro state (MacroLSTM with FRED); a second network builds the worst-case test portfolio; three-phase minimax training.
  - Evaluation: SDF-implied max Sharpe (Hansen-Jagannathan bound), pricing errors vs factor models, adversarial portfolio. Caveat: cross-sectional GMM can give a spurious fit because of the weighting matrix (Gospodinov-Kan-Robotti); repeat with different weights and test assets.
- **Supervised Autoencoder (SAE)**: encoder -> bottleneck, decoder reconstructs inputs (regularizer), supervised head predicts future returns; winner of the Kaggle Jane Street 2020-2021 competition. A predictive benchmark, not a structural factor model.
- **Hierarchical PCA (Avellaneda)**: PCA within each economic group (GICS) and then PCA on the correlation of the sector factors; factors interpretable as inter-sector bets.
- **Procrustes alignment**: orthogonal rotation to align loadings across windows: SVD of B_t' B_{t+1} = U S V', R* = U V', B̃_{t+1} = B_{t+1} R*.
- **Two-stage production PCA (Paleologo 2025)**: Stage 1 volatility (half-life ~20 days, preliminary PCA, idiosyncratic volatility per asset); Stage 2 correlation (normalize by that vol, half-life ~120 days, production PCA + BBP shrinkage + Procrustes); reconstruct Ω = D(BΩ_fB' + Ω_ε)D with D restoring the volatility scale.
  - Why: volatility changes over days/weeks, correlation over months/quarters.

## Key Concepts
- **Factor zoo**: hundreds of published factors (>400 by the mid-2010s); the evidence points to a recurring core (market, value, momentum, profitability, investment) amid fragile results.
- **Attribution vs priced factors**: the former explain co-movement without a necessary premium; the latter carry a risk premium (Idzorek-Kaplan-Ibbotson).
- **SDF / pricing kernel**: object linking future payoffs to prices; no-arbitrage E[M r]=1.
- **Eigenportfolio**: eigenvector read as portfolio weights; PC1 ~ market proxy (0.99 correlation with the equal-weighted portfolio across 500 liquid stocks).
- **Marchenko-Pastur / BBP**: eigenvalue benchmark for noise / recoverability threshold: a factor is recoverable only if its population eigenvalue exceeds 1+√γ with γ = N/T (eigenvalues normalized to unit idiosyncratic variance).
- **Eigenvector cosine similarity**: |v_t' v_{t+1}|; frequently <0.8 = unreliable component.
- **Level/Slope/Curvature**: three PCs of the rate curve.
- **Characteristic-managed portfolios**: cross-sectional projection of returns onto lagged characteristics.
- **ALS**: alternating least squares for IPCA (Γ and f).
- **Hansen-Jagannathan bound**: links the SDF's maximum Sharpe to the magnitude of pricing errors.
- **IC information ratio**: recommended validation metric alongside quintile spreads and turnover.

## Mental Models
- Use PCA when the objective is risk/covariance decomposition (eigenportfolios, yield curve hedging), not when seeking a risk premium.
- Use RP-PCA when you suspect low-variance, high-premium factors that PCA buries.
- Use IPCA when you want auditable linear conditional betas and a risk vs mispricing test; CAE when there are nonlinear interactions among characteristics.
- Use SDF when the objective is direct pricing error; SAE when you only want a strong predictive signal.
- Use several estimators under a single walk-forward protocol: agreement = likely signal, divergence = model risk.

## Anti-patterns
- **Equating explained variance with risk premium**: PCA does not distinguish priced factors.
- **PCA with N close to or greater than T without shrinkage**: inflated eigenvalues, noisy eigenvectors; choosing components by visual scree plot is unreliable with moderate/large γ.
- **Interpreting components of order >=3 as structural**: they rotate/flip sign across windows due to near-degenerate eigenvalues; treat them as descriptive.
- **Selecting the CAE by reconstruction loss**: it measures optimization, not OOS premia.
- **Reporting a single K in IPCA**: report a 3-8 range.
- **Comparing named vs latent factors with different universe/construction**: Hou-Xue-Zhang show that 65% of 452 anomalies fail |t|>1.96 with NYSE breakpoints and value-weighting.
- **Ignoring costs**: net ≈ gross - 2 × turnover × one-way cost (5-10 bps large cap; 50+ microcaps).
- **Taking SHAP on the beta network as economic validation**: it is only an interpretability diagnostic.
- **Using PCA on firm characteristics with IDs that are not stable month to month**: breaks the balanced panel.

## Code Examples
The extracted text includes no code; only formulas and notebooks (01_pca_equity_sectors ... 08_supervised_autoencoder). Compact reconstruction of the adapter described:

```
pipeline = LatentFactorForecastPipeline(model, forecaster, mapper)
mu_hat = beta_t @ lambda_hat
```
What it demonstrates: the structural model (Stage 1) and the premium forecaster (Stage 2) are interchangeable; the asset map is current exposure times forecast premium.

## Reference Tables
Table 14.1 (yield curve): Level PC1 >90% (parallel shift; inflation/growth); Slope PC2 ~5-8% (steepening/flattening; monetary policy); Curvature PC3 ~1-2% (butterfly; rate volatility; requires >=5-7 maturities). In notebook 03 (8 yields 1Y-30Y): 82.3%, 12.3%, 3.1% (97.8% cumulative). Litterman-Scheinkman: 3 factors explain 95-99%.

Table 14.2 (model / objective / loadings / adapter / use):

| Model | Objective | Loadings | Adapter | Best use |
|---|---|---|---|---|
| PCA | Maximize explained covariance | Static | Optional | Risk decomposition |
| RP-PCA | Variance + pricing error | Static latent | Yes (extractor) | Discover priced factors |
| IPCA | Conditional latent factor | Betas linear in characteristics | Yes | Interpretable conditional betas |
| CAE | Nonlinear conditional latent factor | Neural betas | Yes | Nonlinear conditional betas |
| Adversarial SDF | No-arbitrage restrictions | Pricing kernel weights | No | Minimize pricing error |
| SAE | Supervised prediction + reconstruction | Bottleneck | No | Predictive benchmark |

Stability rules: eigenvalue ratios >2:1 among the top 3-5 PCs = Procrustes unnecessary; clustered eigenvalues (common beyond the 3rd) = always Procrustes.

Effective CAE hyperparameters (us_firm_characteristics): 2-3 hidden layers, pyramidal widths 32-128, LR 1e-4 to 1e-2, L1 1e-5 to 1e-3, batch 1,000-5,000 (close to a full monthly cross-section), dropout >0.3 underfits and <0.05 overfits, initial K around 5-10 (lower bound illegible in the extraction), Optuna >=50 trials with validation IC or IC IR as the objective.

Table 14.3 (best latent vs best prior model; IC and Δ): ETFs 21d SDF +0.085 (t 4.9) vs NLinear +0.062 (+0.023); US Firm Char 1m SAE +0.062 (5.5) vs GBM +0.080 (-0.018); CME 5d SDF +0.037 (2.8) vs GBM +0.032 (+0.005); S&P500 Eq+Opt 5d SDF +0.012 (0.7) vs TabM +0.011 (+0.001); US Equities 1d IPCA +0.005 (2.2) vs GBM +0.032 (-0.027).

## Worked Example
Cross-case study (Sec 14.8, 5 of 9 case studies with balanced cross-sections): ETFs and S&P500 Eq+Opt run all 5 estimators; US Firm Char excludes PCA (unstable anonymized IDs); CME runs PCA and SDF; US Equities runs PCA and IPCA. Metric: daily IC with 95% HAC CI.
- The best latent model excludes zero in 4 of 5 (all but S&P500 Eq+Opt); PCA leads in none; SDF/SAE lead 4 of 5 and IPCA the fifth (US Equities).
- Paired against the best prior model (Ch11-13): credibly leads on ETFs (+0.023), credibly trails on US Equities (-0.027) and US Firm Char (-0.019); CME and Eq+Opt: point estimate in favor but the interval crosses zero. Result: matches or beats in 3 of 5, loses in 2.
- Objective ladder on US Firm Char (1m): SAE +0.062, CAE -0.030, SDF and IPCA near zero. Multi-task objective with label = positive ranking; pure contemporaneous reconstruction points in the wrong direction (reconstruction is a poor proxy for one-step forecasting on monthly characteristics).
- Eq+Opt: IPCA with a risk-adjusted 5d label reaches +0.041 with the interval away from zero (the only credible latent signal).
- Dimensionality: CME and ETFs (T >> N) favorable; US Equities and US Firm Char highly dimensional (~9x more assets than periods); those that achieve credible IC there (IPCA, SAE) avoid estimating loadings from the return covariance.
- Per-fold stability: US Firm Char (monthly, structural edge) positive in all folds; US Equities (daily) positive in ~3 of 5 folds -> use the HAC interval as the threshold, not the point estimate.
- CAE replication on us_firm_characteristics: IC -0.030 (shorter sample, smaller universe, fewer characteristics than GKX).

Dimensionality example with γ=N/T (γ values illegible in the extraction): ETFs = low threshold (several components); 500 stocks with 1 year of daily data = only market + a few sector factors; CME = most identifiable.

## Key Takeaways
- Define whether you are after risk (PCA/eigenportfolios), pricing (RP-PCA/SDF) or prediction (IPCA/CAE/SAE) before choosing.
- With high N/T apply shrinkage and expect to recover only the dominant factors; monitor cosine similarity and apply Procrustes.
- Estimate several methods under a single walk-forward and read agreement as signal, disagreement as model risk.
- Validate the complete workflow (IC, IC IR, quintile spreads, turnover, net vs gross), not the reconstruction.
- Report results with and without the bottom 20% by market cap to separate tradable signal from microcap artifacts.
- Keep characteristics point-in-time and lagged; fix the universe before estimating.
- PCA remains useful as a risk decomposition even if it does not lead on IC.

## Connects To
Ch4 (point-in-time, no survivorship bias), Ch7 (multiple testing; t threshold stricter than 2 for new factors, exact value illegible), Ch9 (Kalman, GARCH), Ch10 (text embeddings as characteristics), Ch11 (SHAP, IC/AUC, symmetric metric), Ch12 (Optuna, HPO), Ch13 (sequence models as Stage 2 forecasters), Ch15 (double-selection LASSO, debiased ML), Ch16 (signals), Ch17 (covariance, eigenportfolio betas for sizing, mean-variance, HRP), Ch18 (costs), Ch19 (risk management), Ch20 (multi-objective ensemble), Ch23 (GNNs).

## Code Reference

Official repository: [14_latent_factors/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/14_latent_factors). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_pca_equity_sectors.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/01_pca_equity_sectors.ipynb) + `.py` | 14.2 | PCA on sector ETF returns, with bootstrap confidence intervals on the loadings and a rolling decomposition of the factor structure. |
| [`02_eigenportfolios.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/02_eigenportfolios.ipynb) + `.py` | 14.3 | PCA on the 500 most liquid US equities, producing gross-normalized eigenportfolios, sector loading analysis, hierarchical PCA, residual persistence diagnostics and a two… |
| [`03_yield_curve_decomposition.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/03_yield_curve_decomposition.ipynb) + `.py` | 14.4 | Decomposes Treasury yield changes into three components and uses them for factor hedging. |
| [`04_ipca.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/04_ipca.ipynb) + `.py` | 14.5 | Alternating least squares on a synthetic panel with a known loading matrix, so subspace recovery can be checked against ground truth rather than asserted. |
| [`05_rp_pca.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/05_rp_pca.ipynb) + `.py` | 14.5 | Builds the modified covariance matrix and sweeps the pricing-error weight. |
| [`06_conditional_autoencoder.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/06_conditional_autoencoder.ipynb) + `.py` | 14.6 | A beta network over characteristics and a factor network over jointly estimated managed portfolios, trained as a contemporaneous reconstruction model, with a separate wa… |
| [`07_stochastic_discount_factor.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/07_stochastic_discount_factor.ipynb) + `.py` | 14.7 | A portfolio-weight network trained against an adversarial moment network, with a separate beta network for the asset-level predictive head. |
| [`08_supervised_autoencoder.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/08_supervised_autoencoder.ipynb) + `.py` | 14.7 | Reconstruction and two classification heads share one bottleneck. |
| [`09_case_study_insights.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/14_latent_factors/09_case_study_insights.ipynb) + `.py` | 14.8 | Reads the case-study registries and trains nothing. |
