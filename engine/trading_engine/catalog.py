"""Catalog loading and resolution. Invalid experiments are stopped here, before any data is touched."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .core.errors import CatalogError, MissingCapabilityError, UnimplementedError
from .core.interfaces import Capabilities
from .core.paths import catalog_dirs


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_merged(name: str, *, allow_empty: bool = False) -> dict:
    """Plugin first (defaults); the workspace adds or overrides entries by id."""
    merged: dict = {}
    for d in reversed(catalog_dirs()):
        p = d / name
        if p.exists():
            merged.update(_read(p))
    if not merged and not allow_empty:
        raise CatalogError(f"{name} not found in {[str(d) for d in catalog_dirs()]}")
    return merged


def load_sources() -> dict:
    return _load_merged("sources.yaml")


def load_markets() -> dict:
    return _load_merged("markets.yaml")


def load_timeframes() -> dict:
    return _load_merged("timeframes.yaml")


def load_strategy_map() -> dict:
    return _load_merged("strategy_map.yaml")


def load_assets() -> dict:
    """Assets are declared only in the workspace: the plugin ships none."""
    return _load_merged("assets.yaml", allow_empty=True)


# requirement -> (check on Capabilities, stream the asset must declare)
REQUIREMENTS = {
    "trades": (lambda c: c.trades, "trades"),
    "quotes": (lambda c: c.quotes, "quotes"),
    "volume": (lambda c: c.has_volume, "trades"),
    "aggressor_side": (lambda c: c.aggressor_side, "trades"),
    "ts_recv": (lambda c: c.has_ts_recv, None),
}


def capabilities_of(source_cfg: dict) -> Capabilities:
    try:
        return Capabilities(**source_cfg["capabilities"])
    except (KeyError, TypeError) as e:
        raise CatalogError(f"malformed capabilities in sources.yaml: {e}") from e


@dataclass(frozen=True)
class Resolved:
    symbol: str
    asset: dict
    market_id: str
    market: dict
    source_id: str
    source: dict
    capabilities: Capabilities


def resolve(symbol: str, requires=()) -> Resolved:
    """Resolve asset -> market -> source and check that the source meets the requirements."""
    assets = load_assets()
    if symbol not in assets:
        raise CatalogError(f"asset {symbol} is not declared; add it to the workspace catalog/assets.yaml (/trading:add-asset)")
    asset = assets[symbol]
    streams = asset.get("streams", [])

    markets = load_markets()
    market_id = asset.get("market")
    if market_id not in markets:
        raise CatalogError(f"{symbol}: market '{market_id}' does not exist in markets.yaml")
    market = markets[market_id]
    if market.get("status") != "implemented":
        raise UnimplementedError(f"market {market_id} is not implemented: the engine will not run experiments on it")

    sources = load_sources()
    source_id = asset.get("source")
    if source_id not in sources:
        raise CatalogError(f"{symbol}: source '{source_id}' does not exist in sources.yaml")
    source = sources[source_id]
    if source.get("status") != "implemented":
        raise UnimplementedError(f"source {source_id} is not implemented")

    caps = capabilities_of(source)
    for req in requires:
        if req not in REQUIREMENTS:
            raise CatalogError(f"unknown requirement: '{req}'")
        check, needs_stream = REQUIREMENTS[req]
        if not check(caps):
            raise MissingCapabilityError(
                f"{symbol}: source {source_id} does not meet '{req}' "
                f"(trades={caps.trades}, quotes={caps.quotes}, volume_unit={caps.volume_unit})"
            )
        if needs_stream and needs_stream not in streams:
            raise MissingCapabilityError(
                f"{symbol}: the asset does not declare the '{needs_stream}' stream required by '{req}'"
            )

    return Resolved(symbol, asset, market_id, market, source_id, source, caps)
