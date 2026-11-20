"""Skeleton: US equities via Polygon. Plan and cost must be checked before implementing it."""
from ..core.errors import UnimplementedError


class Polygon:
    name = "polygon"

    def __init__(self, config: dict):
        raise UnimplementedError(
            "polygon: adapter not implemented. Check the plan and cost of historical L1 trades/quotes "
            "before writing it (see catalog/sources.yaml)."
        )
