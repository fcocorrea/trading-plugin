"""Skeleton: equities. Needs sessions, open/close auctions, short locates and the uptick rule."""
from ..core.errors import UnimplementedError


class EquityRules:
    def __init__(self, market_cfg: dict, costs: dict):
        raise UnimplementedError(
            "market equity_us: not implemented. Missing a source with L1 trades and session rules."
        )
