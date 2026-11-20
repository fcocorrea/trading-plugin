"""Skeleton: dated futures. Needs contract rolling and a margin model."""
from ..core.errors import UnimplementedError


class FutureRules:
    def __init__(self, market_cfg: dict, costs: dict):
        raise UnimplementedError(
            "market future_cme: not implemented. Missing source, rolling and margin model."
        )
