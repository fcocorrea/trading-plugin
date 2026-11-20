"""Explicit errors. An error is better than a silently wrong result."""


class TradingError(Exception):
    """Base of every engine error."""


class CatalogError(TradingError):
    """Malformed catalog or a reference to something that does not exist."""


class MissingCapabilityError(TradingError):
    """The source cannot meet a requirement of the experiment."""


class UnimplementedError(TradingError):
    """Component that exists as a skeleton and is not implemented yet."""


class DataValidationError(TradingError):
    """Data that fails validation; the day is rejected."""
