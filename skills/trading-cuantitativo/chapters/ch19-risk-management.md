# Chapter 19: Risk Management

## Core Idea
Risk management is system design, not a reporting layer: it turns a winning backtest into an operable system with constraints, adaptive controls and governance artifacts that (1) are implemented without lookahead, (2) are auditable and (3) exist before deployment.

## Frameworks Introduced
- **Risk taxonomy + control matrix**: each risk maps to an observable proxy, a control and a monitoring frequency (market, factor, leverage, concentration, liquidity, model, operational).
  - When to use: specify the risk term sheet before trading.
  - How: see Table 19.2 below; distinguish intended vs unintended exposures.
- **VaR / CVaR (Expected Shortfall)**: VaR_α = α quantile of the loss distribution; CVaR_α = E[L | L ≥ VaR_α]. Methods: historical simulation (robust, sample-dependent), parametric (fast, biased under fat tails), Extreme Value Theory (deep tails, requires choosing a threshold); Cornish-Fisher.
  - When to use: report both; VaR for familiarity, CVaR for severity; historical as the base and parametric as a sensitivity check.
  - Why it works / failure mode: CVaR is subadditive (Rockafellar-Uryasev) and VaR is not; VaR ignores the magnitude beyond the threshold; both underestimate under stress (liquidity, tail correlations, costs that multiply).
- **Regime-conditional risk metrics**: VaR/CVaR by regime (low-vol, normal, high-vol via realized vol or VIX); regimes persist for months; the most dangerous moments are the transitions. Liquidity haircuts: double or triple spreads, raise impact by the crisis/normal ratio, reduce assumed ADV; can raise effective VaR by 50-100%.
- **Cantelli bound (distribution-free)**: P(shortfall > kσ) ≤ 1/(1+k²); with Sharpe SR, P(losing period) ≤ 1/(1+SR²): 50% at SR=1, 20% at SR=2 (Gaussian: 16% and 2.3%). Hard floor when fat tails invalidate parametric VaR.
- **Volatility-model evaluation (Hansen-Lunde, Patton)**: use loss functions robust to the proxy: MSE of variance (not of volatility) and QLIKE (L = σ²_proxy/σ̂² − ln(σ²_proxy/σ̂²) − 1, penalizes under-prediction more). QLIKE for risk, MSE for alpha research; compute both.
- **Path-risk metrics**: Maximum Drawdown, drawdown duration, recovery time, Ulcer Index (root mean square of drawdowns), Calmar. Conditional drawdown analysis by regime.
- **Factor / sector / macro decomposition**: r_p = Σβ_k F_k + ε; Fama-French models (3/5), Barra/MSCI (10-20 factors), macro (rates, credit, VIX, dollar, oil). Extras: HAC (Newey-West) standard errors in attribution (report "+2.3% ± 0.8%"); correlation thresholding of residuals (|ρ|>0.15); variance diagnostic of the Mahalanobis distance (d²_t = r'Σ̂⁻¹r, expected = dimension); maximal attribution (hierarchical factor rotation); trade-level SHAP (TradeShapAnalyzer).
- **Stress testing**: (a) historical replay (2008 GFC, 2010 Flash Crash, 2015 CNY, 2020 COVID, 2022 Rate Shock) as regime exemplars; (b) cost stress (spreads 3-10x in severe crises); (c) combined scenario matrix (Table 19.6); (d) Student-t Monte Carlo; (e) factor stress (momentum reversal, value compression, rate shock, credit); (f) reverse stress testing (working backward from a 30% drawdown: which combination of factors, how plausible, what would be observed beforehand).
  - Rule: every stress that reveals a vulnerability maps to an accepted risk, a hedge, a reduced exposure or a kill switch; a one-page quarterly memo (scenarios, vulnerabilities, changes made). If the third section is empty, the stress test failed.
- **Adaptive risk controls without leakage**: the control at t uses only information available at t−1 or earlier. Components: GARCH(1,1) (σ²_t = ω + αε²_{t-1} + βσ²_{t-1}) and EWMA (combine them); volatility targeting (w = w*·σ_target/σ̂_{t-1}, one-step lag, cap on the factor in low vol); Short-Term Volatility Updating (STVU: slow correlation + fast vol multiplier); regime-triggered exposure caps (gross ~200% calm, ~150% elevated, ~100% severe, predefined thresholds using VIX/credit spreads); concentration and turnover caps that tighten under stress; position-level: fixed, trailing, ATR-scaled and MAE/MFE-calibrated stops.
  - Failure mode: regimes inferred by hidden-state models fitted on the full sample or smoothed probabilities = leakage.
- **Kill switches and governance**: three trigger families (performance, behavior, market-state); every trigger needs a metric, threshold, action, escalation path and reinstatement condition. Graduated escalation (Table 19.7). Drift detection with PSI. Model risk documentation (Federal Reserve 2011).
- **Deep hedging (Buehler et al. 2019)**: hedging as direct optimization of a risk objective (CVaR via Rockafellar-Uryasev); produces no-transaction regions; safe deployment = learned policy inside deterministic envelopes.

## Key Concepts
- **Four stylized facts (Cont, 2001)**: daily returns with almost no autocorrelation; |r| and r² strongly autocorrelated; fat tails; more Gaussian at long horizons.
- **Gamma vs delta events**: fast, sharp drops (1987, COVID, respond to options) vs slow, deep ones (2008, dot-com, require duration defenses).
- **Exceedance correlation**: correlation conditioned on extremes, higher in the downside tail (Ang-Timmermann).
- **Intended vs unintended exposure**: exposure sought by the thesis vs exposure acquired mechanically through construction.
- **Exogenous vs endogenous liquidity**: market conditions vs the strategy's own impact.
- **Drawdown asymmetry**: -30% requires +43%; -50% requires +100%.
- **PSI (Population Stability Index)**: <0.10 minimal drift; 0.10-0.25 moderate; >0.25 significant (retrain).
- **Precision matrix quality**: temporal stability of d² indicates the reliability of the inverse covariance.
- **Drawdown rule paradox (Varma, 2025)**: mechanical drawdown rules tend to lock in losses that a recovery would have erased.

## Mental Models
- Use CVaR when tail severity matters, VaR for communication, and both by regime when the strategy has an asymmetric tail.
- Use Cantelli when parametric VaR is unreliable (fat tails, regime change).
- Use adaptive controls only with observable, lagged series (lagged realized vol, published VIX), never with ex-post regime labels.
- Use graduated escalation with context (VIX, cross-asset confirmation) instead of a single binary cutoff.
- Use reverse stress testing when you want to uncover hidden vulnerabilities (e.g., a "market-neutral" book sensitive to a simultaneous value+momentum crash).
- Use learned exits to size winners, not as a substitute for explicit stops on losers.

## Anti-patterns
- **Thresholds mined in-sample, control parameters tuned on the evaluation window, kill switches calibrated with hindsight**: they make a weak control look strong; a control that cannot be specified ex ante or reconstructed afterward is not valid.
- **Using same-day realized volatility or smoothed regimes**: lookahead.
- **Gaussian VaR for tails**: structural bias in the part that matters.
- **Backtest with normal-period costs in the tail**: a historical -15% can be -25% with crisis spreads.
- **Static factor exposures**: a hedge calibrated with a fixed loading ends up mis-sized (IWM HML loading 0.1 → 0.3 in 2015-2024; positive stock-bond correlation in 2022).
- **Equity factor model applied to crypto/commodities**: attributes exposures to alpha.
- **PnL attribution without standard errors**: an attribution whose CI crosses zero does not support "driver".
- **Kill switch without escalation or reinstatement**: it is just a number on a dashboard.
- **Stress test that does not trigger action**: wasted effort.

## Code Examples
No code in this range (only references to notebooks and to the ml4t.backtest.risk module: StopLoss, TrailingStop, TighteningTrailingStop, ScaledExit composable with RuleChain/AllOf/AnyOf; MaxDrawdownLimit, DailyLossLimit, MaxPositionsLimit, GrossExposureLimit with warn/reduce/halt actions).

## Reference Tables
| Risk | Proxy | Control | Frequency |
|---|---|---|---|
| Market | Net beta, duration | Exposure limits, hedges | Daily |
| Factor | Factor loadings | Factor constraints | Weekly |
| Leverage | Gross/NAV | Hard caps, vol triggers | Intraday |
| Concentration | HHI, max weight | Position limits | Daily |
| Liquidity | ADV coverage, spreads | Capacity limits | Weekly |
| Model | OOS degradation | Re-research triggers | Monthly |
| Operational | Error rates | Automated checks | Continuous |

| Combined scenario | Vol | Spreads | Impact | Correlations |
|---|---|---|---|---|
| Moderate Stress | 1.5x | 2x | 1.5x | +0.2 |
| Severe Stress | 2x | 3x | 2x | +0.3 |
| Crisis | 3x | 5x | 3x | 0.7 universal |
| Liquidity Crisis | 1.5x | 10x | 5x | +0.2 |

| Kill switch level | Trigger | Action | Authority |
|---|---|---|---|
| 1 Watch | 5% DD | More monitoring | PM |
| 2 Caution | 10% DD | Reduce 50% | PM + Risk |
| 3 Review | 15% DD | Stop new positions | CIO |
| 4 Pause | 20% DD | Full unwind | CIO + Board |
| 5 Terminate | 30% DD | Shut down strategy | Board |

| Path metric | Measures | Use |
|---|---|---|
| Max Drawdown | Worst peak-to-trough drop | Mandates, communication to allocators |
| Drawdown Duration | Time in drawdown | Investor patience |
| Recovery Time | Time to recover the peak | Capital planning |
| Ulcer Index | Integrated pain | Holistic comparison |
| Calmar | Return/MDD | Path-adjusted return |

## Worked Example
VaR on SPY/AGG/GLD/EFA/EEM (equal-weight, 2006-2024, notebook 01_var_cvar): diversification reduces portfolio VaR by ~24% vs the weighted sum of individual VaRs. Regime: SPY 95% CVaR in high-vol = 4.33% vs 1.63% in low-vol (~2.7x). Kupiec backtest at 95%: historical exception ratio 1.11x (p=0.10) beats parametric 1.17x (p=0.012) and Cornish-Fisher 1.45x (p<0.001); Cornish-Fisher becomes unstable when excess kurtosis reaches double digits. Decision: historical base, regime-conditioned, with liquidity haircuts.

Path risk: SPY GFC 2007-2012 drawdown -55.2% over 355 trading days, recovery 869 days. 60/40 under stress: GFC -29.2%, COVID -21.5%; 2022: 60/40 -20%, defensive -20%, aggressive -25%, All Weather -26%. Student-t Monte Carlo 20d 95%: defensive VaR -4.5%/CVaR -5.9%; 60/40 -7.1%/-9.3%.

Attribution: SPY/QQQ/IWM/VTV/VUG portfolio (40/20/15/15/10), FF3 R² = 0.995, alpha 0.47% annual: the "alpha" is factor in disguise. Mahalanobis: mean d² 6.88 vs 5 expected, Var(d²)=1457 → flag HIGH → shrinkage before MVO. FF5 residuals: VTV-VUG -0.73, QQQ-VUG +0.54 (>0.15).

## Key Takeaways
- Design the risk term sheet before deployment: limits by name/sector/factor, leverage ceiling, triggers and authorities.
- Report VaR and CVaR by regime with liquidity haircuts; evaluate volatility models with QLIKE and variance MSE.
- Measure path risk (MDD, duration, recovery, Ulcer) and condition it by regime.
- Decompose exposures on a rolling basis, with HAC standard errors, and check residuals for latent sub-factors.
- Run historical stress + combined scenarios + reverse stress every quarter and require a memo with actions.
- Every adaptive control uses lagged information, parameters fixed ex ante and walk-forward evaluation; everything auditable.
- Monitor drift (PSI) and pre-commit to graduated kill switches with market context.

## Connects To
Ch6 (strategy definition), Ch7-9 (CV, regimes, GARCH), Ch11-14 (leakage, SHAP), Ch16 (backtest), Ch17 (allocators, conformal sizing Sec 17.4, covariance), Ch18 (cost stress, capacity as a risk limit), Ch20 (synthesis and risk overlays Sec 20.7).

## Code Reference

Official repository: [19_risk_management/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/19_risk_management). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_var_cvar.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/01_var_cvar.ipynb) + `.py` | 19.1, 19.3, 19.4 | This notebook demonstrates VaR and CVaR computation methods—historical, parametric, and Cornish-Fisher—along with backtesting to validate estimates against realized loss… |
| [`02_exit_strategies.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/02_exit_strategies.ipynb) + `.py` | 19.4, 19.7 | This notebook explores exit strategies that protect profits and limit losses. |
| [`03_position_sizing_mae_mfe.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/03_position_sizing_mae_mfe.ipynb) + `.py` | 19.4 | This notebook demonstrates position sizing methods and MAE/MFE analysis for stop calibration. |
| [`04_factor_exposure.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/04_factor_exposure.ipynb) + `.py` | 19.2, 19.5 | This notebook decomposes portfolio risk into factor components using regression-based methods. |
| [`05_trade_shap_diagnostics.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/05_trade_shap_diagnostics.ipynb) + `.py` | 19.5 | This notebook demonstrates TradeShapAnalyzer from ml4t.diagnostic.evaluation, which connects SHAP explanations to trade outcomes for systematic improvement. |
| [`06_stress_testing.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/06_stress_testing.ipynb) + `.py` | 19.1, 19.2, 19.6 | This notebook demonstrates stress testing methodologies including historical crisis replay, hypothetical scenarios, and Monte Carlo simulation. |
| [`07_drift_detection.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/07_drift_detection.ipynb) + `.py` | 19.7, 19.8 | This notebook demonstrates drift detection methods for production ML trading systems. |
| [`08_ml_exit_signals.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/08_ml_exit_signals.ipynb) + `.py` | 19.7 | This notebook demonstrates a two-model architecture for exit timing. |
| [`09_deep_hedging.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/09_deep_hedging.ipynb) + `.py` | 19.7 | This notebook demonstrates deep hedging (Buehler et al., 2019): a neural network learns hedging positions that minimize CVaR of terminal PnL under transaction costs. |
| [`10_ml4t_backtest_risk_demo.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/10_ml4t_backtest_risk_demo.ipynb) + `.py` | 19.4, 19.8 | Demonstrates the ml4t.backtest.risk module—the library implementation of risk management concepts discussed in Chapter 19. |
| [`11_systematic_risk_sweep.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/19_risk_management/11_systematic_risk_sweep.ipynb) + `.py` | 19.7 | Demonstrates how to systematically optimize position-level exit rules (StopLoss, TakeProfit, TrailingStop) through 1D sweeps, 2D grid searches, and MAE/MFE-calibrated st… |
