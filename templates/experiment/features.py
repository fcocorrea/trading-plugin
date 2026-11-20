"""Feature families of this experiment (ML4T ch08).

One function per family declared in setup.yaml (features.families[].name), with the same name.

Contract, checked by the runner:
- Input: the bars of ONE symbol, sorted by ts. Columns: symbol, ts (bar close, UTC), open, high,
  low, close, vwap, volume, quote_volume, buy_volume, sell_volume, n_trades.
- Output: a polars DataFrame with the same number of rows and one column per feature. The runner
  prefixes each column with "<family>__", applies the family's lag and drops warm-up rows.
- Causal: the value at row t may only use rows <= t. The runner recomputes every family on
  truncated histories and refuses a family whose past values change when the future is removed.
  Use rolling windows and shift(+n); never shift(-n), centered windows or full-sample statistics.

Skeleton:

    def <family_name>(bars: pl.DataFrame) -> pl.DataFrame:
        return bars.select(
            <polars expression>.alias("<feature_name>"),
        )
"""
import polars as pl  # noqa: F401
