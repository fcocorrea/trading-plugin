"""Estimators named in training/<label>.yaml (ML4T ch11-ch12).

Linear models are wrapped with a StandardScaler inside a Pipeline, so scaling is fitted on each
training window only (no standardization leakage). Regressors serve return labels; classifiers
serve direction labels and predict the probability of an up move.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .core.errors import TradingError


@dataclass(frozen=True)
class EstimatorSpec:
    target: str
    scaled: bool
    build: Callable[..., Any]


def _ols(**p):
    from sklearn.linear_model import LinearRegression
    return LinearRegression(**p)


def _ridge(**p):
    from sklearn.linear_model import Ridge
    return Ridge(**p)


def _lasso(**p):
    from sklearn.linear_model import Lasso
    return Lasso(**p)


def _elastic_net(**p):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(**p)


def _logistic(**p):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**{"max_iter": 1000, **p})


def _lgbm_regressor(**p):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**{"verbosity": -1, "random_state": 0, "deterministic": True, **p})


def _lgbm_classifier(**p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**{"verbosity": -1, "random_state": 0, "deterministic": True, **p})


ESTIMATORS: dict[str, EstimatorSpec] = {
    "ols": EstimatorSpec("return", True, _ols),
    "ridge": EstimatorSpec("return", True, _ridge),
    "lasso": EstimatorSpec("return", True, _lasso),
    "elastic_net": EstimatorSpec("return", True, _elastic_net),
    "lightgbm": EstimatorSpec("return", False, _lgbm_regressor),
    "logistic": EstimatorSpec("direction", True, _logistic),
    "lightgbm_classifier": EstimatorSpec("direction", False, _lgbm_classifier),
}
LINEAR_ESTIMATORS = ("ols", "ridge", "lasso", "elastic_net", "logistic")


def build_estimator(params: dict, target: str):
    params = dict(params)
    name = params.pop("estimator")
    spec = ESTIMATORS.get(name)
    if spec is None:
        raise TradingError(f"unknown estimator '{name}'; available: {sorted(ESTIMATORS)}")
    if spec.target != target:
        raise TradingError(f"estimator '{name}' serves {spec.target} labels, not {target}")
    try:
        model = spec.build(**params)
    except TypeError as e:
        raise TradingError(f"estimator '{name}': invalid parameters {params}: {e}") from e
    if not spec.scaled:
        return model
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), model)


def predict(model, X, target: str):
    if target == "direction":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
