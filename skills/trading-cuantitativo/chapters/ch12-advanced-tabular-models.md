# Chapter 12: Advanced Models for Tabular Data

## Core Idea
Gradient Boosting Machines (XGBoost, LightGBM, CatBoost) are the defensible default for financial tabular prediction: they capture interactions and thresholds without manual feature engineering. The choice between libraries is operational (speed, categoricals, ecosystem), and the extra flexibility over the linear baseline is only justified by walk-forward evidence (IC with a HAC interval that excludes zero).

## Frameworks Introduced
- **Gradient Boosting (Friedman 2001)**: F_m(x) = F_{m-1}(x) + η·h_m(x); initialize with a constant (target mean), fit a weak learner (shallow tree) to the pseudo-residuals (negative gradient of the loss), and add it scaled by the learning rate η.
  - When to use: tabular prediction with nonlinear relationships, thresholds, and interactions.
  - How: (1) F_0 = mean; (2) for m=1..M compute the negative gradient; (3) fit a tree; (4) add with η (0.01-0.3).
  - Why it works / failure mode: corrects systematic bias (e.g. predicts 2%, actual 5% -> the next tree learns the 3% residual), something bagging cannot do. Failure: overfits in low-SNR settings without shrinkage/regularization and disciplined validation.
- **GBM library selection**: XGBoost = general-purpose + ecosystem (SHAP, Optuna, deployment) + explicit regularization (γ per leaf, L2 λ on weights); LightGBM = histograms + leaf-wise + GOSS + EFB, best on CPU; CatBoost = ordered target statistics + symmetric trees (fast inference, best on GPU).
  - When to use: see Reference Tables (Table 12.1/12.2).
  - How: decide based on high-cardinality categoricals, CPU vs GPU, inference latency, leakage risk.
  - Why it works / failure mode: after tuning, the 4 libs fall within a narrow IC band (~0.002 IC); the gap between good and bad hyperparameter configurations is larger than the gap between libraries. Failure: leaf-wise LightGBM overfits on small/noisy data without num_leaves, max_depth, min_data_in_leaf.
- **Learning to Rank (LambdaMART)**: objective='lambdarank' (LightGBM), rank:ndcg (XGBoost), YetiRank/YetiRankPairwise (CatBoost); optimizes NDCG (e.g. ndcg_eval_at=[10]) with groups by date (group parameter) and a label_gain that maps intra-date return quintiles to relevance grades.
  - When to use: quantile/long-short strategies that only trade the tails with high breadth (e.g. 500 stocks, top/bottom decile = 50 names).
  - How: group by date; label relevance by quintile; train; compare NDCG@10 and IC against an MSE baseline.
  - Failure mode: scores are ordered but not calibrated to magnitudes; calibrate post-hoc or use pointwise prediction if the portfolio (e.g. mean-variance, Ch17) needs return forecasts. The advantage is diluted with full cross-section portfolios, low breadth, or very low SNR.
- **Monotonic constraints**: +1 (prediction does not decrease as the feature rises) / -1; name->direction mapping supported in all 3 libraries.
  - When to use: relationships with a known economic sign in low-SNR settings (e.g. value, P/E).
  - How: assign the sign, train, verify with a SHAP dependence plot (strictly monotonic curve), and compare IC with the unconstrained version.
  - Benefit: theory-guided regularization, auditability for the risk committee, robustness to regime changes. If IC is comparable, the constraint acts as regularization, not as information loss.
- **Optuna HPO (TPE + pruning)**: TPE maintains densities l(x) (good configs) and g(x) (bad ones) and maximizes l(x)/g(x); define-by-run API (trial.suggest_int/float) with conditional parameters.
  - When to use: GBM space with discrete/conditional parameters.
  - How: fixed low learning rate (0.01-0.05) + early stopping for the number of rounds; prioritize reg_alpha/reg_lambda; 100-200 trials maximum (50-100 if SNR is very low); MedianPruner or HyperbandPruner (saves >=50% of compute); shared storage (PostgreSQL/MySQL) for parallel runs.
  - Failure mode (Box 12.3): validation overfitting; 500 TPE trials exploit validation noise, and a 0.02 IC improvement vanishes on truly held-out data; defenses: few trials, coarse grids, evaluate final candidates on untouched data.
- **Multi-objective optimization (NSGAIISampler)**: Pareto frontier, e.g. maximize rank IC and minimize turnover.
  - How: two objectives; read the frontier's curvature (flat = both without sacrifice; steep = marginal IC at the cost of rising turnover).
  - Other pairs: return vs drawdown, Sharpe vs capacity.
- **TreeSHAP**: exact Shapley values in O(TLD^2) per prediction; decomposes into main effects + pairwise interactions.
  - When to use: dependence, interactions, feature selection, drift monitoring.
  - How: mean |SHAP| per fold/window; flag features with an importance change >50%; validate with rolling IC before acting.
  - Failure mode: Rashomon Effect, equally good models give different explanations; validate across model families.
- **Tabular DL (TabPFN, TabM, TabR)**: TabPFN = foundation model with in-context learning (X_train,y_train,X_test -> predictions without training); TabM = parameter-efficient ensemble with rank-1 adapters on a shared MLP backbone; TabR = encoder + k-NN retrieval.
  - When to use: see Table 12.2.
  - Failure mode: TabR without temporal isolation = lookahead bias (Box 12.1); TabPFN degrades faster than GBMs under distribution shift and on purely categorical data.

## Key Concepts
- **Bagging vs boosting**: bagging averages independent trees (reduces variance); boosting adds sequential corrections (reduces bias).
- **Random Forest**: bagging with a random subset of features per split; a baseline that GBMs must beat materially.
- **Pseudo-residuals**: the negative gradient of the loss with respect to the current prediction; what each tree fits.
- **Histogram-based learning**: discretizes features into bins to evaluate splits per bin rather than per distinct value.
- **GOSS**: keeps large-gradient samples and subsamples small-gradient ones (reweighted).
- **EFB**: bundles mutually exclusive features (typical after one-hot) into a single one.
- **Ordered target statistics (CatBoost)**: target statistic computed only from preceding samples in a random permutation, avoiding target-encoding leakage.
- **Oblivious/symmetric trees**: same split per level; prediction as a bitwise operation, very fast inference.
- **NDCG**: ranking metric that rewards getting the top of the list right exponentially more.
- **Rashomon Effect**: models with equal performance and contradictory explanations.
- **Conformal variants (Ch11 applied)**: split conformal, quantile regression, CQR, and ACI for intervals on GBM residuals.

## Mental Models
- Use boosting (not Random Forest) when the ensemble's error is systematic bias, not variance.
- Use LightGBM as the research default on CPU with massive walk-forward CV; XGBoost if you have a CUDA GPU or need the ecosystem; CatBoost if there are high-cardinality categoricals (tickers, analyst IDs), a large GPU, or inference latency matters.
- Use LTR when only the tails of a broad cross-section are traded; pointwise when the portfolio consumes magnitudes.
- Use SHAP (not gain/split importance) for any decision; native importance only for screening (gain is biased toward high cardinality, split-count toward continuous features, unstable across seeds).
- Use a SHAP monitoring score as an early warning, not as proof of concept drift; confirm with rolling IC.
- Use AutoGluon as a "ceiling sanity check": if your LightGBM/TabM is close, you have already extracted most of the signal.

## Anti-patterns
- **Choosing a library by accuracy benchmark**: the gap between libs (~0.002 IC) is smaller than the hyperparameter gap; decide on operational criteria.
- **Tuning tree structure and ignoring regularization**: reg_alpha/reg_lambda usually have the largest OOS impact in low-SNR settings.
- **Hundreds of HPO trials (500)**: validation overfitting, the main source of backtest-to-live degradation.
- **Using LTR scores as return forecasts**: they are not calibrated.
- **TabR with a retrieval index over the whole train set**: lookahead; treat results as an upper bound.
- **Trusting IID tabular benchmarks (TabArena, TALENT)**: they exclude temporal dependence; unequal tuning budgets, post-hoc ensembles, and dataset leakage (TabArena kept 51 of 1,053).
- **A regression model to predict sign**: AUC against direction stays within ±0.04 of 0.5.
- **GPU for LightGBM on small datasets**: its CUDA backend is FP64 only; on ETF (227K rows) heavy: ~9.5s CPU vs ~48s GPU.
- **TabPFN 2.5 weights in production**: non-commercial license by default.

## Code Examples
There are no code blocks in the extracted text; the chapter points to notebooks (01_ensemble_foundations, 02_gbm_comparison, 03_dl_vs_gbm, 04-07 Optuna, 08_shap_analysis, 11_conformal_gbm, 12_case_study_insights). Key configuration described (minimal parameter reconstruction):

```
pointwise: objective='regression'
ranking:   objective='lambdarank', label_gain=<intra-date quintile -> relevance>,
           ndcg_eval_at=[10], group=<cross-section size per date>
GPU:       XGBoost device='cuda' | CatBoost task_type='GPU' | LightGBM device='cuda'
```
What it demonstrates: the pointwise -> ranking switch is only the objective, the relevance labels, and the group by date.

## Reference Tables
Table 12.1 (4.9M rows, 72 features, 8 threads):

| Criterion | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| CPU train | ~100s | ~50s (fastest) | ~135s |
| GPU train | ~14s (~7x) | ~14s (~3.5x, FP64) | ~10s (~14x) |
| Prediction | 0.5M rows/s CPU; 1.7M GPU | 0.07M/s CPU | 6.5M/s CPU (~12x XGBoost) |
| High-cardinality categoricals | native | native | ordered TS without leakage |
| Tuning sensitivity | moderate | moderate-high | lower, data-dependent |
| Memory | moderate | best | moderate |

Table 12.2 (selection by data regime):

| Regime | Recommended |
|---|---|
| <10K effective samples, quick testing | TabPFN (v2.5; distilled) |
| 10K-100K, production | LightGBM (or XGBoost) |
| 10K-100K, DL justified | TabM, validate walk-forward |
| >100K with many categoricals | CatBoost (or LightGBM) |
| >1M with GPU | LightGBM and TabM (or TabR); no reliable crossover |
| Multimodal (text+tabular) | End-to-end DL |
| Local similarity/regime matters | TabR with temporal isolation |

TabPFN limits: original up to 10,000 samples / 500 features; v2.5 ~50,000 rows / 2,000 features.

Table 12.3 (average daily IC, primary label; Linear / GBM / TabM): US Firms 1m -0.005/+0.080/+0.031; ETFs 21d +0.054/+0.037/+0.041; CME Futures 5d -0.000/+0.032/+0.004; US Equities 1d +0.016/+0.032/+0.017; SP500 Options (straddle) +0.007/+0.018/+0.002; Crypto 8h +0.009/+0.011/+0.003; NASDAQ-100 15m +0.005/+0.006/-; SP500 Eq+Opt 5d -0.006/+0.006/+0.011; FX 1d +0.005/+0.003/+0.007.

Table 12.4 (GBM classifier: score->IC / own AUC / regression score->AUC): Crypto 8h +0.020/0.513/0.502; US Firms 1m +0.084/0.543/0.540; SP500 Eq+Opt 5d +0.004/0.507/0.489; 10d -0.010/0.505/0.498.

## Worked Example
Cross case study (Sec 12.6): a linear baseline, GBM, and TabM are fit to nine case studies with their primary label; the measure is the average daily IC with a HAC 95% CI (corrects for autocorrelation); the threshold is that the interval excludes zero.
- Result: GBM excludes zero in 5 of 9 (linear in 3; TabM in 3 of 8). GBM reaches significance where linear does not: US Firms (+0.080), CME Futures (+0.032), SP500 Options (+0.018). In US Equities it doubles the IC (+0.016 -> +0.032).
- ETFs: linear wins (+0.054 vs GBM +0.037, TabM +0.041); cross-asset momentum is nearly linear. FX and SP500 Eq+Opt: everything is within noise; more capacity does not manufacture signal.
- Configuration: grid of 3 losses (MSE, MAE, Huber) x leaves 7-127. MAE wins in 8 of 9 (MSE on ETFs). US Firms: MAE ~+0.080 vs Huber/MSE ~+0.030-0.035 (MAE does not chase extreme heavy-tailed returns). Depth matters least; early stopping is what pays off: peak at ~50 trees (SP500 Options, ETFs), 100 (CME, Crypto), 150 (FX), ~350 (US Firms); US Equities/NASDAQ-100 keep rising up to the 500 cap.
- Persistence: 3-month rolling IC; in US Equities 2000-2015 GBM beats linear over almost the entire period (including 2008-2009); in ETFs 2015-2023 the curves cross, with no lasting advantage.
- Decision: GBM adds ranking value in 5 case studies (US Firms, US Equities, CME, SP500 Options, NASDAQ-100); an interval that excludes zero certifies ranking, not return; Ch16-19 evaluate whether it survives costs and risk.

Multi-objective ETF example (nb 06): NSGA-II finds 2 non-dominated solutions (IC~0.029/norm turnover ~0.004 and IC~0.030/~0.005); single-objective TPE on IC converges to IC~0.010.

## Key Takeaways
- Start with shallow-tree GBM, MAE loss, and early stopping as the tuning outcome; LightGBM by default on CPU.
- Fix the learning rate at 0.01-0.05; tune regularization; limit HPO to 50-200 trials and reserve untouched data.
- Use LTR if you trade the tails of a broad cross-section; calibrate before feeding into portfolio optimization.
- Apply monotonic constraints where there is an economic sign; verify with SHAP dependence.
- Use TreeSHAP in every fold; monitor mean |SHAP| (alert if change >50%) and validate against rolling IC.
- Always validate against the linear baseline (ETFs: linear won); check persistence in rolling IC before deploying.
- TabPFN only for signal prototyping; TabM as the practical DL baseline; everything under walk-forward with purging/embargo.

## Connects To
- Ch11 (linear models, SHAP sanity checks, conformal prediction, IC/AUC baseline); Ch13 (sequences: LSTM/transformers/TCN); Ch16 (signal stage ranks by IC); Ch17 (mean-variance needs calibrated scores); Ch19 (position sizing with conformal intervals, risk monitoring); Ch26 (production alerts); Ch10 (NLP features, token-level SHAP with FinBERT); Ch7 (AUC).

## Code Reference

Official repository: [12_gradient_boosting/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/12_gradient_boosting). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_ensemble_foundations.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/01_ensemble_foundations.ipynb) + `.py` | 12.1 | Benchmarks RF vs XGBoost/LightGBM/CatBoost on the Chen-Pelger-Zhu academic firm characteristics panel; quantifies the GBM-over-bagging advantage and the small spread acr… |
| [`02_gbm_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/02_gbm_comparison.ipynb) + `.py` | 12.2 | Systematic GBM benchmark: 4 libraries × 3 presets × CPU/GPU on the ETF panel, plus a 5M-row scale benchmark, learning-to-rank and monotone-constraint demos. |
| [`03_dl_vs_gbm.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/03_dl_vs_gbm.ipynb) + `.py` | 12.3 | Walk-forward IC, training time, and early-stopping behavior for LightGBM, TabM, TabPFN, and a minimal torch MLP on the ETF panel. |
| [`04_optuna_tuning.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/04_optuna_tuning.ipynb) + `.py` | 12.4 | Full Optuna HPO workflow for LightGBM on ETFs — single-fold vs walk-forward HPO, pruning, fANOVA importance. |
| [`05_cross_library_hpo.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/05_cross_library_hpo.ipynb) + `.py` | 12.4 | Cross-library GBM HPO (XGBoost, LightGBM, CatBoost) on Chen-Pelger-Zhu firm characteristics at identical search budgets. |
| [`06_optuna_multi_asset.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/06_optuna_multi_asset.ipynb) + `.py` | 12.4 | Multi-objective HPO (IC vs turnover) with Pareto front, plus cross-asset hyperparameter transfer between ETFs and CME futures. |
| [`07_hpo_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/07_hpo_comparison.ipynb) + `.py` | 12.4 | Grid search vs Optuna TPE on identical budget, both on a discrete grid and continuous search space. |
| [`08_shap_analysis.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/08_shap_analysis.ipynb) + `.py` | 12.5 | TreeSHAP global + local explanation, MDI/PFI/SHAP consensus, interaction decomposition, walk-forward drift, and SHAP-based feature selection on ETFs. |
| [`09_xai_limitations.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/09_xai_limitations.ipynb) + `.py` | 12.5 | Demonstrates XAI instability: similar predictions can have divergent SHAP explanations (Rashomon set), and SHAP differs systematically across nominally equivalent fits. |
| [`10_shap_nlp_sentiment.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/10_shap_nlp_sentiment.ipynb) + `.py` | 12.5 | SHAP token-level attribution on FinBERT sentiment classification; GPU recommended for transformer inference. |
| [`11_conformal_gbm.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/11_conformal_gbm.ipynb) + `.py` | 12.5 | Conformal prediction intervals for GBM regression (split conformal, QR-conformal, CQR) with empirical coverage diagnostics. |
| [`12_case_study_insights.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/12_gradient_boosting/12_case_study_insights.ipynb) + `.py` | 12.6 | Cross-case-study insight notebook for the GBM family: daily-pooled IC with HAC inference, loss/depth/leaf grids, checkpoint-trajectory peak distribution, holdout decay,… |
