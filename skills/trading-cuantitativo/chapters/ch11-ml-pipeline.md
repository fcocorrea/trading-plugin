# Chapter 11: The ML Pipeline

## Core Idea
Regularized linear models (Ridge, LASSO, Elastic Net, logistic regression) are the interpretable baseline of the ML signal workflow: they are fit with point-in-time preprocessing, validated with walk-forward (nested if there is tuning), audited with SHAP, and accompanied by calibrated uncertainty via conformal prediction. Every model in Ch 12-15 must beat this baseline.

## Frameworks Introduced
- **Inference vs prediction (Breiman, two cultures)**: data modeling (estimate the parameters of a DGP) vs algorithmic modeling (success = out-of-sample accuracy). *When*: algorithmic trading → predictive criterion. *Why*: Gauss-Markov only optimizes variance among unbiased estimators; MSE = Bias² + Variance + σ², and moderate bias with much less variance lowers MSE. *Failure mode*: OLS with 100 features and 240 observations: unstable coefficients.
- **Regularization (λ controls the tradeoff)**: *Ridge* (L2, β̂ = (XᵀX+λI)⁻¹Xᵀy; shrinks low-variance directions more; for diffuse signal and multicollinearity), *LASSO* (L1; automatic selection if the signal is sparse; unstable across correlated features: picks one at random, switches between windows), *Elastic Net* (L1/L2 mix; groups correlated features; in sklearn alpha + l1_ratio, 1.0 = LASSO, 0 = Ridge). *How*: CV over l1_ratio {0.1,0.3,0.5,0.7,0.9} and log-spaced alpha; l1_ratio near 1 = the data prefer sparsity, near 0 = diffuse signal; if unstable across windows, average predictions.
- **Leakage-safe standardization**: z-score with μ,σ from the train fold only (recompute at each refit); winsorize at the 1/99 percentiles (from train) before standardizing. Both errors are silent.
- **Optuna HPO + nested walk-forward**: Optuna (TPE, define-by-run) for tuning; risk = validation overfitting; fix = nested CV (inner HPO, outer evaluation), with nested walk-forward. Report the number of trials, search space, chosen hyperparameters, and per-fold dispersion. Calibrate the space: sklearn.Ridge uses unnormalized SSE (alpha scales with n); ElasticNet averages (a different n factor). With ~400k obs, searching λ only up to 10² explores the near-OLS zone.
- **Loss choice**: MSE (conditional mean), MAE (median, robust to outliers), Huber (quadratic for small errors, linear for large ones beyond threshold δ); SGDRegressor with loss='squared_error' / 'epsilon_insensitive' / 'huber'; quantile regression changes the estimand (percentiles) if you only trade the extreme deciles.
- **Sample weighting**: recency w = exp(−λ·age) + uniqueness (Ch 7); they compose multiplicatively; report N_eff = (Σw)²/Σw² (equivalently: N_eff of the combined weights) vs nominal N.
- **Signal-to-trade mapping (regression vs classification)**: threshold or quantile rule → classification; position proportional to magnitude → regression; it is empirical. Direction can be misaligned with PnL under asymmetric payoffs (55% hit rate while missing the large moves).
- **Logistic regression**: P(y=1|x)=σ(βᵀx), minimizes log-loss; coefficients linear in log-odds (standardized β=0.5: 50% → ~62%; 90% → ~94%); multinomial/softmax for long/neutral/short; with p ≈ n, perfect separation drives coefficients to ±∞ → regularization is critical (sklearn C = 1/λ). Regularization compresses probabilities toward the base rate (underconfidence); calibrate with Platt scaling or isotonic regression (CalibratedClassifierCV) using non-training data. Probability bin vs hit rate must be monotonic.
- **Prob→signal**: threshold (τ_upper=0.6/τ_lower=0.4 as a starting point), probability-weighted (requires calibration; size ∝ |p−0.5|), rank-based (top/bottom decile, robust to miscalibration). Thresholds/cutoffs are recomputed within each training window.
- **Class imbalance**: class_weight='balanced'; watch N_eff when combining weights (risk if the minority class clusters in crisis periods).
- **SHAP as a continuous diagnostic**: Shapley (axioms: Efficiency, Symmetry, Null player, Linearity); LinearSHAP φ_j = β_j(x_j − E[x_j]) exact and O(p); TreeSHAP (Ch 12), KernelSHAP (Ch 13). Four layers of economic narrative: sign consistency, magnitude plausibility, stability across walk-forward windows (SHAP stability chart), regime-conditional analysis; bootstrapping SHAP within a fold separates noise from regime change; right-vs-wrong comparison on high-conviction predictions; waterfall + concentration-risk flag (a single feature dominates |φ|). Three pillars: diagnosis, economic sanity-check, governance (SR 11-7, SS1/23).
- **Conformal prediction**: model-agnostic calibration layer; finite-sample marginal coverage guarantee under exchangeability. *Split-conformal*: (1) split into proper-train and calibration (do not touch until evaluation); (2) score s_i = |y_i − f̂(x_i)|; (3) q̂ = s_(k), k = ⌈(n_cal+1)(1−α)⌉ (conservative order statistic); (4) interval f̂(x) ± q̂ (fixed width). *Adaptations* (do not restore the theorem): rolling calibration windows (~250 days), weighted conformal (exponential decay), regime-conditional calibration (defined ex ante, with enough observations per regime), online coverage feedback. *ACI* (Gibbs-Candès): α_{t+1} = α_t + γ(α − 1{Y_t ∉ C_t(X_t)}); miss → α drops → wider interval; γ ∈ [0.005, 0.05]; clip α_t ∈ [0.001, 0.999]; guarantees long-run average coverage, not per regime. *CQR* (Romano et al.): lower/upper quantiles (5%/95% for 90%), score = max(q_lo − y, y − q_hi), correction q̂, interval [q_lo − q̂, q_hi + q̂]; adaptive width; depends on the quality of the quantile model. *Classification sets*: score s = 1 − p̂(y|x), set = {k: p̂(k|x) ≥ 1 − q̂}; conservative rule: trade only singleton sets and abstain if there are several; report coverage/hit rate/abstention/turnover separately.
- **Uncertainty-based sizing (preview)**: position ∝ 1/interval width (a 50 bps interval → twice the weight of a 100 bps one), before vol/correlation/turnover (Ch 17).
- **Baseline ladder**: every model in Ch 12-14 reports its improvement (or lack of it) over Ridge.

## Key Concepts
- **BLUE / Gauss-Markov**: OLS is the best linear unbiased estimator under linearity, strict exogeneity, no perfect multicollinearity, and spherical errors.
- **Bias-variance tradeoff**: MSE = Bias² + Variance + σ².
- **ICIR**: mean/standard deviation of the IC across folds (see Ch 7).
- **Platt scaling / isotonic regression**: parametric logistic / nonparametric monotonic calibration.
- **AUC**: 0.5 random; 0.55-0.60 meaningful predictive power in finance; >0.65 OOS warrants suspicion of leakage.
- **Brier score / log-loss**: calibration-sensitive metrics.
- **Exchangeability**: the joint distribution does not change under reordering; weaker than IID; markets violate it.
- **Marginal vs conditional coverage**: average guarantee vs coverage within a fold/regime/asset.
- **SHAP stability chart**: mean |φ| per fold over time.
- **SGDRegressor/SGDClassifier partial_fit**: incremental update (Box 11.1); drifts without periodic full retrains (Ch 25).
- **Nonconformity score**: observed error used to calibrate intervals/sets.

## Mental Models
- Use **Ridge** when many features each contribute a little and there is multicollinearity (diffuse signal); **LASSO** when the signal is sparse; **Elastic Net** with families of correlated indicators.
- Use **classification** when the decision is a threshold/quantile rule; **regression** when size scales with the predicted magnitude.
- Use **rank-based signal** when calibration is doubtful; **probability-weighted** only with calibrated probabilities.
- Use **CQR** if residual dispersion is heterogeneous and there is enough calibration data; **rolling split-conformal or ACI** if quantiles are unstable or there are abrupt shifts.
- Use **SHAP (not coefficients)** for attribution: shrinkage entangles importance with the penalty, and correlated features split the credit.
- Use **turnover and net Sharpe** alongside IC before claiming an edge: higher IC can lose after costs.

## Anti-patterns
- **Choosing a model by p-values**: significance ≠ predictive power.
- **Standardizing with full-sample moments** or not winsorizing first (an inflated σ compresses everything else).
- **λ range too narrow** (exploring only the near-OLS zone).
- **Single-loop CV with a weak signal**: flips the sign (single-loop mean IC +0.028 vs nested −0.032 on the same splits).
- **Interpreting logistic output as a calibrated frequency** without diagnosing it.
- **Interpreting SHAP as causal**; permutation importance inflates correlated features; LIME does not sum to the prediction.
- **Trusting marginal coverage** without stratifying by fold/regime/magnitude/asset.
- **Expecting conformal to re-center a biased forecast**: it calibrates coverage, not the center.
- **Applying the conformal guarantee to the traded subset** (singletons only) without measuring it separately.
- **Interpreting a small significant IC as a strong signal**: NASDAQ-100 clears zero because of sample size, not effect size.
- **Reporting only MSE improvements**: they may come from fitting the center of the untraded cross-section or from more turnover.

## Code Examples
```
from sklearn.linear_model import ElasticNet, SGDRegressor
ElasticNet(alpha=a, l1_ratio=r)          # 1/(2n)||y-Xb||^2 + a[r||b||_1 + (1-r)/2 ||b||_2^2]
SGDRegressor(loss="huber", penalty="l2", alpha=a)
model.partial_fit(X_batch, y_batch)
CalibratedClassifierCV(clf, method="isotonic")
```
What it demonstrates: the chapter's sklearn vocabulary (ElasticNet parameterization with l1_ratio, pluggable loss in SGD, partial_fit for incremental updates, and post-hoc calibration); the book describes but does not list the full code, so this is a compact reconstruction.

## Reference Tables
Regularization:

| Method | Penalty | Signal prior | Risk |
|---|---|---|---|
| Ridge | L2 | Diffuse, multicollinear | Does not select |
| LASSO | L1 | Sparse | Unstable across correlated features |
| Elastic Net | L1+L2 | Correlated groups | Two hyperparameters |

Metrics by task: regression → IC (primary), RMSE/MAE (vs a naive model predicting the cross-sectional mean), turnover, quintile/decile spreads; classification → AUC (0.55-0.60 meaningful; >0.65 suspicious), precision/recall (high costs favor precision; capacity favors recall), log-loss/Brier if the probability matters, turnover.

Conformal on ETFs (99 ETFs, 2006-2025, 8 walk-forward folds 2015-2023, 80/20 split, Ridge base, 90% target):

| Method | Marginal coverage | Mean width | Width σ |
|---|---|---|---|
| Split-conformal | 88.4% | 16.9% | 3.3% |
| CQR | 89.9% | 15.7% | 7.5% |
| ACI on SC (γ=0.01) | 89.5% | 16.3% | 6.8% |

Case studies (regression IC, HAC 95% CI): ETFs 21d +0.054 (t 2.4, [+0.009,+0.098], n=2,016); US Equities 1d +0.016 (t 9.1); NASDAQ-100 15min +0.005 (t 4.3, n=64,580): CI excludes zero. Crypto 8h +0.009, SP500 Options ~30d +0.007, FX 1d +0.005: positive but cross zero. CME Futures 5d −0.000, SP500 Eq+Opt 5d −0.006, US Firms 1m −0.005: negative and cross zero. Real signal in 3 of 9.
Direction vs magnitude (Table 11.6): Crypto 8h AUC 0.509 (cross-IC +0.033); US Firms 1m AUC 0.539 (cross-IC +0.074); SP500 Eq+Opt AUC 0.510/0.507; regression cross-AUC ≈ 0.49-0.52.

Coverage warning signs: <80% during volatility spikes for a 90% target (stale or poorly stratified calibration); monotonic deterioration (drift); asymmetric violations (directional bias).

## Worked Example
Ridge on ETFs, 21-day labels (n ≈ 394,000): for α < ~100 Ridge ≈ OLS (IC ≈ 0.028, ICIR ≈ 0.54); mean IC peaks at α ≈ 3×10⁴ (0.047), ICIR peaks at 1.30 at α ≈ 10⁶ = 2.4x over OLS, driven by +60% mean IC (0.046 vs 0.028) and −33% IC standard deviation across folds. LASSO, with 57 correlated features: α ≈ 0.0001 already zeroes out ~25% of features, larger penalties push IC below OLS, α ≥ 0.01 eliminates almost everything → the signal is diffuse, Ridge preserves information and LASSO discards it. λ sweep (10⁻² to 10⁹): mean IC slightly negative, minimum −0.039 near λ=10³, crosses zero at 10⁷ and reaches ≈ +0.007; per-fold standard deviation of 0.04-0.10 dominates the differences between λ values (a noise-dominated landscape). SHAP on ETFs (Ridge): vol_126d first (positive), vol_63d second (opposite), then sharpe_63d, vol_ratio_medium, yield_curve_slope; example waterfall: baseline 0.5% → momentum +0.8% → mean reversion −0.3% → volatility +0.2% → 1.4%; a wrong high-conviction prediction dominated by a liquidity/volume feature (+0.018) with an actual return of −11.7%. IC → profitability gap: simple 126d momentum has lower IC than Ridge but matches/beats its net Sharpe because Ridge trades much more.

## Key Takeaways
- Every new model must beat the Ridge baseline; otherwise, it is complexity without predictive power.
- Standardize and winsorize within the train portion of each fold, and widen the λ search (10⁻² to 10⁹ with large samples).
- If you do HPO, use nested walk-forward; report trials, search space, and per-fold dispersion.
- Choose regression vs classification based on the signal→trade rule; calibrate if sizes depend on the probability level.
- Audit with SHAP (sign, magnitude, stability, regime) and flag predictions dominated by a single feature.
- Add conformal (CQR/ACI/rolling) and monitor coverage by fold/regime; size ∝ 1/width.
- Report turnover alongside IC/AUC; economic significance requires costs and sizing (Ch 16-19).

## Connects To
Ch 6 (walk-forward, trading setup), Ch 7 (labels, IC/ICIR, uniqueness, triage), Ch 8-10 (features), Ch 12 (gradient boosting, advanced Optuna, TreeSHAP), Ch 13 (deep learning, KernelSHAP), Ch 14 (latent factors, multi-horizon), Ch 15 (causality), Ch 16-19 (simulation, portfolio, costs, risk), Ch 17 (allocator, uncertainty-based sizing), Ch 18 (net evaluation), Ch 25 (online learning).

## Code Reference

Official repository: [11_ml_pipeline/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/11_ml_pipeline). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_ols_inference.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/01_ols_inference.ipynb) + `.py` | 11.1 | This notebook shows what classical inference looks like before we leave |
| [`02_regularization_paths.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/02_regularization_paths.ipynb) + `.py` | 11.2 | This notebook compares OLS, Ridge (L2), LASSO (L1), and |
| [`03_logistic_classification.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/03_logistic_classification.ipynb) + `.py` | 11.3 | This notebook applies logistic regression to |
| [`04_nested_cv_hpo.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/04_nested_cv_hpo.ipynb) + `.py` | 11.2 | This notebook develops a systematic approach to hyperparameter |
| [`05_shap_analysis.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/05_shap_analysis.ipynb) + `.py` | 11.4 | This notebook uses SHAP (SHapley Additive exPlanations) to interpret a |
| [`06_conformal_prediction.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/06_conformal_prediction.ipynb) + `.py` | 11.5 | This notebook demonstrates conformal prediction methods |
| [`07_case_study_insights.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/07_case_study_insights.ipynb) + `.py` | 11.6 | This notebook compares linear model results across all 9 |
| [`08_ml_backtest_intro.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/11_ml_pipeline/08_ml_backtest_intro.ipynb) + `.py` | 11.6 | This notebook provides a pedagogical backtest comparing |
