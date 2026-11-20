"""Market rules registry. Every profile in markets.yaml points to an implementation here."""
from __future__ import annotations

from ..core.errors import CatalogError
from .crypto_perp import CryptoPerpRules
from .crypto_spot import CryptoSpotRules
from .equity import EquityRules
from .future import FutureRules

_IMPL = {
    "crypto_perp": CryptoPerpRules,
    "crypto_spot": CryptoSpotRules,
    "equity": EquityRules,
    "future": FutureRules,
}


def build_rules(market_cfg: dict, costs: dict):
    impl = market_cfg.get("implementation")
    if impl not in _IMPL:
        raise CatalogError(f"no market implementation '{impl}'")
    return _IMPL[impl](market_cfg, costs)
