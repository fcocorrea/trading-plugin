---
name: add-source
description: How to add a new data adapter to the engine, declaring real capabilities and validating before marking it as implemented. Use when integrating a provider, a new file format, or the user's own CSV files.
---
# Adding a data source

## Checklist
1. **Real capabilities.** Declare in `catalog/sources.yaml` only what the file or API delivers: `trades`, `quotes`, `aggressor_side`, `volume_unit`, `has_ts_recv`. If the provider gives no size, use `volume_unit: none`; the engine will reject it for any volume requirement.
2. **Adapter class** in `engine/trading_engine/sources/<id>.py`, with `name`, `capabilities`, `list_days`, `fetch_raw`, `to_trades` and `to_quotes`. Anything unsupported raises `UnimplementedError` with the reason.
3. **Normalization.** `to_trades` and `to_quotes` return DataFrames with the schema of `core/schema.py` (use `conform` and `quote_volume_expr`). Time units are detected with `detect_unit` and converted with `to_us`.
4. **Registration** in `sources/__init__.py`.
5. **Tests** for the adapter's parsing rules (side mapping, timestamp units, header handling), like `engine/tests/test_sources.py`.
6. **Try one small day.** Download, validate, check the manifest and the report before loading months.
7. **Status.** Mark it `implemented` only after steps 5 and 6.

Vendor due diligence comes first (ML4T ch02): what the data really contains, how it is timestamped, how it handles corrections, and whether it is point-in-time.

## Own CSV format (generic_csv)
Path: `data/raw/generic_csv/{SYMBOL}/{stream}/{YYYY-MM-DD}.csv`
- trades: `ts, price, size` required; `trade_id` and `side` (buy/sell) optional. If you declare `aggressor_side: true`, `side` becomes required.
- `size` is in the unit of `volume_unit` (`base` or `quote`).
- `ts` in UTC: each file may only contain rows of its own UTC day.
- quotes: `ts, bid_px, bid_sz, ask_px, ask_sz`.
- `ts` can be in ns, µs, ms or s; it is detected by magnitude.
