# Chapter 26: MLOps and Governance

## Core Idea
Every deployed model decays; the difference between a profitable operation and a capital-destroying one is how quickly decay is detected and how safely it is handled. The chapter builds a three-layer governance model (detection, response, automated safety) on top of a key taxonomy: technical failure (pipeline divergence, fix the bug) vs statistical failure (correct outputs that no longer predict, retrain, redesign or retire).

## Frameworks Introduced
- **Two-failure taxonomy (Table 26.1)**: technical (same input gives a different output: fix the bug) vs statistical (same input, same output, bad returns: retrain, redesign or retire).
  - When to use: first step of every investigation into poor live performance.
  - How: determine the category before acting; the statistical one has 4 mechanisms: overfitting, look-ahead bias (incl. survivorship), regime change, alpha decay (McLean and Pontiff 2016: after publication, anomalies lose profitability).
  - Why it works / failure mode: treating decay as a bug leads to futile debugging; treating bugs as decay leads to unnecessary model changes.
- **Three-layer governance model**: detection (data gates, rolling metrics, drift), response (shadow, staged rollout), automated safety (circuit breakers); right-sized MLOps.
- **Data integrity gates**: data contracts at ingest (Pandera, Great Expectations): types, nulls, monotonic timestamps, unique keys. Fail-closed policy for execution-critical feeds; fail-open with warnings for non-critical analytics.
- **Rolling metrics + realization ratio**: Sharpe over 30/60/90-day windows, IC, hit rate, win/loss, drawdown depth and duration, all on live returns net of costs; per metric: backtest value, live value, deviation, significance. Realization ratio = live Sharpe / backtest Sharpe (2.0 to 1.5 = 0.75).
- **Tiered alerts**: Watch (yellow, log only), Warning (orange, investigate, consider reducing exposure), Critical (red, protective/automated action). Calibrate from the backtest; measure the false-positive rate.
- **Drift detection by layer**: data drift (PSI, K-S), feature drift (SHAP: mean abs, variance, ranking), concept drift (ADWIN, DDM on prediction errors).
- **Drift x decay matrix (Table 26.3)**: see Reference Tables.
- **Safe model update workflow**: scheduled and/or triggered retraining; shadow mode (incumbent executes, candidate only predicts); A/B with limited capital; gradual ramp-up; explicit promotion criteria; tested rollback.
  - Why it works / failure mode: the replacement can be worse than the decayed original; A/B with real capital reveals clustering in illiquid names and impact that shadow does not see.
- **4-level circuit breakers (defense in depth)**: per-trade, per-strategy, portfolio, system; each level independent. Types: loss-based, position-based, anomaly-based; plus the software circuit breaker (CLOSED/OPEN/HALF_OPEN) for infrastructure resilience.
- **Recovery protocol**: acknowledge, assess, secure the position; diagnostic questions (technical vs model; unusual conditions; sequence); resumption criteria (healthy components, validated feeds, normalized market, root cause identified); gradual restart with reduced size or paper.
- **Override discipline**: log who/when/why, explicit risk acknowledgment, time limit (e.g., 1 hour), post-hoc review.
- **Right-sizing the MLOps stack**: minimal, intermediate, mature (see table).

## Key Concepts
- **PSI (Population Stability Index)**: sum over bins of (A_i - E_i) x ln(A_i/E_i), with A actual and E expected from training.
- **K-S test**: maximum distance between CDFs; no binning, preferred for continuous features.
- **SHAP monitoring**: mean absolute SHAP, SHAP variance and relative ranking.
- **ADWIN**: adaptive window that shrinks when drift is detected; **DDM**: running error rate and deviation with warning/drift states; simpler and less adaptive.
- **Concept drift types**: sudden, gradual, incremental, recurring (Lu et al. 2018).
- **Training-serving skew**: different features in training and inference; solved by feature stores (Feast: offline store Parquet/BigQuery/Redshift; online store Redis/DynamoDB; point-in-time joins).
- **Run manifest**: JSON with git commit, dataset snapshot (content hash), feature set version, registry version, configuration hash; DVC for data versioning.
- **Model registry (MLflow)**: experiment tracking, versioning with lineage, staging to Production.
- **SR 11-7 / CRISP-ML(Q)**: model risk guidance (effective challenge, continuous monitoring, documented governance); they require a model card, an independent validation memo and a monitoring report.
- **Alert fatigue**: thresholds that fire without changing decisions are noise.

## Mental Models
- Use execution monitoring alongside model monitoring when PnL drops: execution costs above what was assumed point to microstructure or infrastructure, not the model; retraining would be the wrong response.
- Use several rolling windows (30/60/90): a drop only in the short one = temporary noise; all dropping = persistent decay.
- Use hit rate and win/loss to decompose Sharpe: hit rate falling with stable win/loss = weak directionality; stable hit rate with worse win/loss = magnitude or execution.
- Use drift without decay as "robust model, just monitor"; decay without drift as incomplete monitoring coverage.
- Use automated rollback for low-level triggers (system errors, feeds) and manual rollback for performance degradation.
- Use simple infrastructure first: the monitoring (26.2-26.3) and safety (26.5) sections matter more than the tools.

## Anti-patterns
- **Monitoring performance without validating data first**: a large share of "drift incidents" are silent defects (stale prices, corporate actions, broken joins, timezones, missing bars, duplicates).
- **Point metrics (a single aggregate Sharpe)**: they hide trends.
- **Thresholds too tight or too loose**: alert fatigue or damage before detection.
- **Promoting on statistical significance alone**: a +0.05 Sharpe may not justify the risk; define a minimum effect size.
- **Short shadow windows**: insufficient statistical power for modest improvements.
- **Replacing the incumbent all at once** without A/B or a tested rollback.
- **Trying to "trade out of the hole"** or waiting for human review when hard limits are breached.
- **Overrides that are too easy or impossible**.
- **Adopting the full MLOps stack prematurely**.
- **Governance records only in chats or ad hoc dashboards**.

## Code Examples
Software circuit breaker state machine (reconstructed from the description):

```python
class CircuitBreaker:
    def call(self, fn, *args):
        if self.state == "OPEN":
            if time.monotonic() - self.opened_at < self.reset_timeout:
                raise CircuitOpen
            self.state = "HALF_OPEN"
        try:
            result = fn(*args)
        except Exception:
            self.failures += 1
            if self.state == "HALF_OPEN" or self.failures >= self.max_failures:
                self.state, self.opened_at = "OPEN", time.monotonic()
            raise
        self.state, self.failures = "CLOSED", 0
        return result
```
What it demonstrates: fail-fast after 5 consecutive failures, a 30 s timeout and a single probe call in HALF_OPEN (the chapter suggests pybreaker for production).

## Reference Tables
**PSI**: < 0.1 no significant shift; 0.1-0.25 moderate shift (investigate); > 0.25 significant shift (predictions likely unreliable).

**Monitoring methods (Table 26.2; extraction partially broken, triggers column reconstructed)**

| Method | Frequency | Typical trigger |
|---|---|---|
| PSI on key features | daily | PSI > 0.25 on high-importance inputs |
| K-S on continuous features | daily or weekly | p-value < 0.05 (recent vs reference) |
| SHAP importance | weekly | ranking reorder or large drop in mean abs SHAP on a core feature |
| ADWIN/DDM on errors | continuous | warning/drift state in the error stream |

**Drift vs decay (Table 26.3)**

| Drift | Decay | Interpretation | Response |
|---|---|---|---|
| Yes | No | robust model | monitor |
| No | Yes | incomplete coverage or gradual concept drift | expand monitoring, untracked features |
| Yes | Yes | feature drift explains the decay | retrain or per-regime model |
| No | No | healthy | normal monitoring |

**Example alert thresholds**: Watch: 30d Sharpe < 1.0 with 1.5 expected; Warning: Sharpe < 0.5 for two weeks; Critical: negative Sharpe for 30+ days or drawdown beyond the historical maximum. Persistent underperformance: live Sharpe 0.5 or more below expected.

**Circuit breakers (examples)**: loss-based: daily -2%, weekly -5%, from peak -10% to -15%; position-based: 5% per asset, 20% per sector, maximum correlation; anomaly-based: VIX > 40, spreads 3x normal, intraday move of 10%+; system: end-to-end latency > 100 ms.

**Rollout**: shadow ~60 days (monthly = at least 2 rebalance cycles); A/B with 5-10% of capital; steps 10 to 25, 25 to 50, 50 to 100 with a minimum period; full transition 60-90 days; promotion: candidate Sharpe >= incumbent + 0.2 (range 0.2-0.3), significance at 10%, no drawdown beyond the historical maximum.

**MLOps stack**: Minimal (pandas/numpy, versioned directories/Git LFS, cron/systemd, manual deploy with a checklist, basic Streamlit/Grafana); Intermediate (Feast, MLflow + DVC, Prefect/Dagster, GitHub Actions, Prometheus + Grafana with 15-30 s scrape and Alertmanager to Slack/PagerDuty, logs with correlation IDs); Mature (feature store with real-time serving, registry with approval workflows, Kubernetes, OpenTelemetry).

## Worked Example
Monitoring on us_equities_panel: reference = last pre-holdout year, current = most recent holdout window. PSI and K-S are computed per feature, plus the score distribution and rolling IC/hit rate/error; the dashboard separates inputs (PSI bars, score histogram) from outputs (rolling IC and hit rate) to tell whether the movement comes from the inputs, the relationship or both. Safe rollout (03_safe_model_rollout): incumbent chosen from the validation history, shadow over the following release year; realized correlation of 0.61 between both models' per-name scores; A/B with the candidate on a small fraction; staged allocation 10% to 25%. Decision: the candidate does not have to dominate every day, but disagreement must be observable, attributable and reviewable before scaling up. Circuit breakers (04): market rules tested on SPY's actual path in 2020 H1 (COVID crash); only the latency stream is synthetic, to inspect the rolling-average trip.

## Key Takeaways
- Classify first: technical (fix) or statistical (retrain/redesign/retire).
- Put fail-closed data contracts in place before monitoring; compare live vs backtest with the realization ratio and realized/simulated execution cost.
- Use PSI + K-S for inputs, SHAP for importance, ADWIN/DDM for errors; cross-check with performance (Table 26.3) before acting.
- Require shadow (~60 days), A/B with 5-10% of capital, promotion by explicit rules (+0.2 Sharpe, p<0.10, no new maximum drawdown) and verified rollback.
- Implement independent breakers at 4 levels with immediate halts, a recovery protocol and audited overrides.
- Version data, features and models (run manifest, Feast, DVC, MLflow) and produce a model card + independent validation + monitoring report.
- Size the infrastructure to the team's maturity.

## Connects To
Ch 25 (technical parity, SafeBroker, kill switches, training_metadata.json and failed_basket monitored here); Ch 22 (RAG governance, audit logging); Ch 24 (agent controls, observability); Ch 27 (system synthesis); validation and deflated Sharpe (Bailey and López de Prado), Harvey-Liu-Zhu, Ch 16 (overfitting).

## Code Reference

Official repository: [26_mlops_governance/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/26_mlops_governance). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

This chapter has no notebooks; only `README.md` in the folder.

Other notebooks in the folder, not linked from the README: `01_drift_monitoring.ipynb`, `02_online_drift_detection.ipynb`, `03_safe_model_rollout.ipynb`, `04_circuit_breakers.ipynb`, `05_feast_feature_store.ipynb`, `05b_feast_live.ipynb`, `06_mlflow_experiments.ipynb`.
