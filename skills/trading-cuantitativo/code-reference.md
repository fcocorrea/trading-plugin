# Code Reference — the book's repository

Official repository: https://github.com/stefan-jansen/machine-learning-for-trading (open source, continuously maintained; it is the primary source of the book's code).

**How it is organized**: each chapter has a folder `NN_name/` (the `NN` prefix is the chapter number). Inside, notebooks are numbered in reading order (`01_…`, `02_…`). Each `.ipynb` has a `.py` pair (jupytext) with the same name, and each folder has a `README.md` that maps every notebook to its section of the book. The book cites notebooks by name (pattern `\d{2}_[\w_]+`); the **Code Reference** table at the end of each `chapters/chNN-*.md` resolves those names to real files, with section and description.

## Folders per chapter

| Ch. | Folder | Notebooks | | Ch. | Folder | Notebooks |
|---|---|---|---|---|---|---|
| 1 | `01_process_is_edge` | 2 | | 15 | `15_causal_estimation` | 11 |
| 2 | `02_financial_data_universe` | 22 | | 16 | `16_strategy_simulation` | 18 |
| 3 | `03_market_microstructure` | 18 | | 17 | `17_portfolio_construction` | 12 |
| 4 | `04_fundamental_alternative_data` | 14 | | 18 | `18_transaction_costs` | 12 |
| 5 | `05_synthetic_data` | 8 | | 19 | `19_risk_management` | 11 |
| 6 | `06_strategy_definition` | 3 | | 20 | `20_strategy_synthesis` | 8 |
| 7 | `07_defining_the_learning_task` | 9 | | 21 | `21_rl_execution_hedging` | 7 |
| 8 | `08_financial_features` | 8 | | 22 | `22_rag_financial_research` | 8 |
| 9 | `09_model_based_features` | 15 | | 23 | `23_knowledge_graphs` | 10 |
| 10 | `10_text_feature_engineering` | 9 | | 24 | `24_autonomous_agents` | 11 |
| 11 | `11_ml_pipeline` | 8 | | 25 | `25_live_trading` | 13 |
| 12 | `12_gradient_boosting` | 12 | | 26 | `26_mlops_governance` | 7 |
| 13 | `13_dl_time_series` | 12 | | 27 | `27_systematic_edge` | 0 (README only) |
| 14 | `14_latent_factors` | 9 | | | | |

Note: chapter 12 of the book ("Advanced Models for Tabular Data") lives in `12_gradient_boosting/`, and chapter 6 ("Strategy Research Framework") in `06_strategy_definition/`.

## Other directories in the repository

- `case_studies/` — the **9 case studies** that run through chapters 6-20 (see below). References in the book to notebooks such as `14_costs.ipynb` or `18_costs.ipynb` point to these, not to the chapter folders.
- `utils/` — shared code that is **imported** (`from utils.x import y`; notebooks run from the repository root): `config.py` and `paths.py` (paths and `.env`), `style.py` (palette and figure defaults), `data_quality.py`, `downloading.py`, `artifact_specs.py`, `modeling.py` (dataset loading, model configs, folds), `cv_splits.py` (calendar-aware, leakage-safe walk-forward), `predictions_cache.py`, `reproducibility.py` (Python/NumPy/Torch seeds), `storage_benchmarks.py` (storage benchmarks for ch. 2).
- `scripts/` — command-line tools (e.g. `verify_installation.py`, `download_artifacts.py`, `create_experiment.py`, `sync_notebooks.py`).
- `data/` — data download scripts. `docs/` — `installation.md`, `running-notebooks.md`. `envs/` — Dockerfiles.

## The 9 case studies (`case_studies/<id>/`)

They apply the same ML4T workflow to different asset classes, frequencies and constraints (Ch 6-20).

| # | Folder | Asset class | Frequency | Universe | Primary label |
|---|---|---|---|---|---|
| 1 | `etfs` | Multi-asset ETFs | Daily | 100 ETFs | fwd_ret_21d |
| 2 | `crypto_perps_funding` | Crypto perpetuals | 8-hourly | 19 pairs | fwd_ret_8h |
| 3 | `nasdaq100_microstructure` | US equities, intraday | 15 min | 114 stocks | fwd_ret_15m |
| 4 | `sp500_equity_option_analytics` | S&P 500 equities | Daily | 634 stocks | fwd_ret_5d |
| 5 | `us_firm_characteristics` | US equities (fundamentals) | Monthly | ~2,500 stocks | fwd_ret_1m |
| 6 | `fx_pairs` | G10 pairs | Daily | 20 pairs | fwd_ret_1d |
| 7 | `cme_futures` | Multi-sector futures | Daily | 30 products | fwd_ret_5d |
| 8 | `sp500_options` | S&P 500 options | Daily | S&P 500 straddles | fwd_ret_dh_10d |
| 9 | `us_equities_panel` | Broad US equities | Daily | ~3,200 stocks | fwd_ret_1d |

**Phases (identical in all of them; stage numbering changes per market)** — example `case_studies/etfs/`:

| Phase | Ch. | Notebook (ETFs) |
|---|---|---|
| Feasibility | 6 | `01_feasibility_analysis` |
| Labels | 7 | `02_labels` |
| Financial features | 8 | `03_financial_features` |
| Model-based features | 9 | `04_model_based_features` |
| Evaluation (IC diagnostics) | 7-9 | `05_evaluation` |
| Linear (baseline to beat) | 11 | `06_linear` |
| Gradient boosting | 12 | `07_gbm` |
| Tabular DL | 12 | `08_tabular_dl` |
| Sequence DL | 13 | `09_dl_lstm`, `10_dl_tsmixer`, `10a_dl_nlinear` |
| Latent factors | 14 | `11_latent_factors`, `11a_pca` … `11e_supervised_autoencoder` |
| Causal DML | 15 | `12_causal_dml` |
| Model analysis | 11-15 | `13_model_analysis` |
| Backtest | 16 | `14_backtest` |
| Portfolio | 17 | `15_portfolio_management` |
| Risk | 19 | `16_risk_management` |
| Costs | 18 | `17_costs` |
| Holdout and synthesis | 20 | `18_holdout_predictions`, `19_holdout_backtest`, `20_strategy_analysis` |

- Layered configuration: `config/setup.yaml` (universe, cadence, costs, labels, walk-forward splits) → `config/training/{label}.yaml` (model menu) → `case_studies/config/{model_type}/{name}.yaml` (shared hyperparameters). Each run is stored with a configuration hash in `run_log/` (the run log of ch. 6.7). The plugin's experiment layout (`experiment-design` skill) mirrors these layers.
- Each stage checks the artifacts it needs and says which earlier stage to run, so a pipeline can resume halfway.
- Details per case study: `case_studies/<id>/README.md` and `case_studies/RUN_LOG.md`.

## Running the code

Commands run from the repository root, on macOS/Linux (on Windows, in the WSL2 Ubuntu terminal). The plugin does not run them: it only reads the code through `ml4t-ref`.

```bash
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading
cp .env.example .env                         # defaults are fine; edit only if a chapter asks for API keys

# Option A — Docker (recommended; covers all 27 chapters and 9 case studies)
docker compose pull ml4t
docker compose up ml4t                       # Jupyter Lab at http://localhost:8888
docker compose run --rm ml4t python 11_ml_pipeline/01_ols_inference.py
docker compose --profile gpu run --rm ml4t-gpu python 13_dl_time_series/01_core_architectures.py

# Option B — local with uv (~90% of the notebooks)
uv sync
uv run python 11_ml_pipeline/01_ols_inference.py
uv run python scripts/verify_installation.py
```

Notebooks that only run in Docker (image `py312` or another): ch. 5 `03_sigcwgan_signatures`; ch. 9 `06_path_signatures`; ch. 10 `01_word2vec`, `02_asset_embeddings`, `03_sentiment_evolution`; ch. 12 `10_shap_nlp_sentiment`; ch. 14 `06_conditional_autoencoder`; ch. 15 `06_fed_announcement_bsts`; ch. 2 `21_storage_benchmark_database` (image `benchmark`); ch. 12 `02_gbm_comparison` in its GPU section (image `rapids`). This list and the commands come from the repository's `docs/running-notebooks.md`; check it if the version changed.

## `ml4t-*` libraries of the ecosystem

The book and notebooks use companion libraries (documentation at https://ml4trading.io): `ml4t-data` (market data acquisition), `ml4t-engineer` (features, labels, alternative bars), `ml4t-models` (latent factors, SDF), `ml4t-diagnostic` (feature and strategy validation, e.g. `TradeShapAnalyzer`), `ml4t-backtest` (event-driven backtesting, `risk` module), `ml4t-live` (production with broker integration).

## How to use this when writing code

The local copy of the repository is managed by `ml4t-ref` (on the PATH while the plugin is enabled): `ml4t-ref sync` creates or updates it, `ml4t-ref path` gives its location and `ml4t-ref url <path>` the permalink at the copy's commit. The search is done by the `ml4t-code-scout` agent.

1. Identify the chapter (Topic Index in `SKILL.md`) and open its **Code Reference** table at the end of `chapters/chNN-*.md`.
2. Pick the notebook by name and section; read it in the local copy (the `.py` is easier to read and compare than the `.ipynb`).
3. Reuse the logic of `utils/cv_splits.py` and `utils/modeling.py` for walk-forward splits and data loading instead of reinventing it, adapted to the plugin's engine.
4. For a strategy's full flow, follow the stages of a case study in order.
5. Before citing a file name or command, check in the local copy that it still exists: it is a living project, and the listing in this skill reflects the repository as of 2026-10-28.
