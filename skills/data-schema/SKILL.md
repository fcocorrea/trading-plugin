---
name: data-schema
description: The engine's normalized trades and quotes schema, volume units, timestamp normalization and validation rules. Use when writing or reviewing a data adapter, interpreting check-data, or designing bars, labels and features that depend on volume or spread.
---
# Data schema

## Trades (`TRADES_SCHEMA` in core/schema.py)
`trade_id, price, size, quote_volume, aggressor_side, ts_exchange_us, ts_recv_us`

- `size` is in the unit the source declares (`volume_unit`). `quote_volume` is the notional in quote currency: `price × size` when the unit is `base`, `size` when it is `quote`. `contracts` fails until the contract multiplier exists.
- `aggressor_side` is `buy`, `sell` or null. If the source declares `aggressor_side: true`, a null rejects the day.
- `trade_id` is null when the source has no identifiers. Then duplicates and gaps cannot be detected, and a warning is recorded.
- `ts_recv_us` may be null in historical files. That is a documented limitation, not an error.

## Quotes
`ts_exchange_us, ts_recv_us, bid_px, bid_sz, ask_px, ask_sz`.
Spread = `ask_px − bid_px`. Midprice = `(bid_px + ask_px) / 2`.

## Validation rules (data/validate.py)
1. **Volume is mandatory.** The source must declare `trades: true` and `volume_unit` in {base, quote, contracts}. Counting updates is not volume. Experiments must list `volume` in `universe.requires`.
2. Sizes and prices must be greater than zero. Otherwise the day is rejected.
3. **Timestamps.** The unit (ns, µs, ms or s) is detected by magnitude, once per file, then converted to microseconds. A timestamp going backwards rejects the day, and so does a row outside the file's UTC day (wrong time zone or wrong file).
4. A duplicated `trade_id` rejects the day. A gap in `trade_id` is a warning, because it can be real or a source failure.
5. Crossed quotes (bid ≥ ask) are a warning. They happen in very active moments.

Each day is written to a `.part` file and renamed after the manifest entry is recorded: a Parquet file at its final path is always complete and validated.

## What is not volume
The number of ticks, the number of quote updates, or the "volume" of OTC forex feeds. If a source reports it that way, declare `volume_unit: none`; it does not meet the requirement.

## Point-in-time
Every feature and label is built from data available at its timestamp (ML4T ch02, ch04). `ts_exchange_us` orders events; when a source provides `ts_recv_us`, it is the closer proxy for when the data was actually known.
