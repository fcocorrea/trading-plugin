"""Adapter registry. Adding a source = class + entry here + entry in sources.yaml."""
from __future__ import annotations

from ..core.errors import CatalogError
from .binance_um import BinanceUMFutures
from .generic_csv import GenericCSV
from .oanda import Oanda
from .polygon import Polygon

_IMPL = {
    "binance_um_futures": BinanceUMFutures,
    "generic_csv": GenericCSV,
    "polygon": Polygon,
    "oanda": Oanda,
}


def build_source(source_id: str, config: dict):
    if source_id not in _IMPL:
        raise CatalogError(f"no adapter registered for '{source_id}'")
    return _IMPL[source_id](config)
