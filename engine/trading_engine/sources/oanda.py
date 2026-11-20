"""Skeleton: OTC forex via OANDA. Its volume is not traded size, so it fails the volume requirement."""
from ..core.errors import UnimplementedError


class Oanda:
    name = "oanda"

    def __init__(self, config: dict):
        raise UnimplementedError(
            "oanda: not implemented. OTC forex has no traded volume, so it cannot meet the requirement."
        )
