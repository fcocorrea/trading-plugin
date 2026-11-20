"""USDT-margined perpetuals. v1 has no funding: a run that needs it fails instead of ignoring it."""
from __future__ import annotations

from ..core.errors import UnimplementedError


class CryptoPerpRules:
    def __init__(self, market_cfg: dict, costs: dict):
        self.cfg = market_cfg
        self.fee_maker_bps = float(costs["fee_maker_bps"])
        self.fee_taker_bps = float(costs["fee_taker_bps"])

    def is_tradable(self, ts_us: int) -> bool:
        return True  # 24/7 calendar

    def can_short(self, symbol: str, ts_us: int) -> bool:
        return bool(self.cfg.get("short_allowed", True))

    def fee(self, notional: float, side: str, liquidity: str) -> float:
        if liquidity not in ("maker", "taker"):
            raise ValueError(f"unknown liquidity: {liquidity}")
        bps = self.fee_maker_bps if liquidity == "maker" else self.fee_taker_bps
        return abs(notional) * bps / 10_000.0

    def settlement_events(self, start_us: int, end_us: int) -> list[dict]:
        raise UnimplementedError(
            "perpetual funding is not modeled in v1. A run that depends on it cannot execute."
        )
