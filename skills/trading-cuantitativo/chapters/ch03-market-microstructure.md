# Chapter 3: Market Microstructure

## Core Idea
Market data is the output of a matching engine with explicit rules; understanding spreads, depth, order types, feeds, LOB reconstruction, and sampling methods is what lets backtests and cost models survive contact with real execution. Sampling ticks into bars changes the statistical and economic meaning of each observation.

## Frameworks Introduced
- **Three dimensions of liquidity**: Spread (cost of immediacy), Depth (quantity available at and beyond the best price), Resiliency (speed of replenishment after consumption).
  - **When to use**: assessing whether a small signal can overcome frictions or whether a large position is executable.
- **Liquidity provider frictions**: adverse selection (Kyle lambda: impact proportional to net order flow; Glosten-Milgrom: the spread arises from adverse selection) and inventory risk. The spread compensates order processing, inventory, and adverse selection; impact has temporary and permanent components and is nonlinear in size (Madhavan 2002).
- **Data hierarchy L1 / L2 / L3 (+ TAQ)**: L1 = top-of-book (venue-specific BBO, NBBO via SIP); L2 = market-by-price (aggregate size per level, no queue); L3 = market-by-order (submit/modify/cancel/execute events, deterministic replay).
  - **When to use**: choose the feed by strategy. TAQ suffices for most medium-horizon strategies; very short-term depth/queue/order-flow strategies require L2/L3.
  - **Trade-off**: L1 signals are more crowded; L2/L3 raise cost and engineering.
- **Price conventions rule of thumb**: midprice `(bid+ask)/2` for short-term price response (reduces bid-ask bounce); executable prices (bid to sell, ask to buy) for PnL markouts; last trade/close for standard OHLC.
- **LOB reconstruction pipeline**: parse -> normalize -> apply updates -> enforce invariants -> snapshot -> validate against independent references (BBO/NBBO).
  - **How**: maintain (1) an order registry keyed by order reference number (size, price, side) and (2) a price-level view. Add (A,F) inserts; X partially cancels; D deletes; U ends the order and creates a new reference; E executes at the order's price; C carries an explicit execution price. P and Q (trade/cross) are not book-update events.
  - **Why it works / failure mode**: it is state accounting; small inconsistencies accumulate. In production, quarantine and count violations instead of failing. Fails if a single venue's book is interpreted as the market.
- **Bar sampling**: time bars, tick, volume, dollar, and information-driven (tick imbalance TIB, volume imbalance VIB, run bars).
  - **When to use**: dollar bars as the ML default; time bars if the horizon/label is in clock time; information bars if the object of study is order flow (toxicity, informed trading), not "more normal returns."
  - **Why it works / failure mode**: the volume clock (Easley, López de Prado, O'Hara) yields returns closer to normal. Information-driven bars fail due to trade direction misclassification, circular initialization, parameter sensitivity, and threshold spiral.
- **Lee-Ready**: (1) quote test: price vs prevailing midpoint (above = buy, below = sell); (2) tick test as fallback at the midpoint (uptick = buy, downtick = sell, unchanged = carry forward the last nonzero direction). Align quotes with an as-of rule or a small lag.
- **Jump detection**: realized variance RV (total quadratic variance) vs bipower variation BV (robust to jumps); jump variance = max(RV - BV, 0). Lee-Mykland: standardized statistic with bipower local volatility over a trailing window K; rejection via Gumbel maxima at family-wise level α.
- **Sessionization workflow (5 steps)**: current exchange calendar (early closes), tag pre/regular/post in local time, validate closed days and early closes, define an explicit session date, treat halts/auctions as separate states.

## Key Concepts
- **Bid-ask spread**: lowest ask minus highest bid; cost of immediacy and primary measure of liquidity.
- **Marketable vs limit vs stop orders**: cross and execute / specify the worst acceptable price / activate on crossing a threshold (stop-market -> marketable, stop-limit -> limit that may not fill).
- **Iceberg / hidden / pegged / post-only**: diagnostic signatures: repeated prints at the same price with a refreshing quote; prints inside the spread with no visible quote; liquidity provision only.
- **Intraday U-shape**: volume, vol, and spreads peak at the open and close; in the 13-day ITCH sample the first and last 30 min = ~15-17% of daily volume each vs 2.6% at midday (~6x).
- **Bid-ask bounce**: negative first-order autocorrelation in tick returns from alternating hits at the bid/ask.
- **Aggressor side**: the side that initiated the trade (FIX tag 5797 at CME; DataBento provides it; ITCH P does not).
- **Threshold spiral**: positive feedback loop in imbalance bars with a high EWMA α.
- **Jarque-Bera**: statistic for deviation from normality (skew + kurtosis); lower = more normal.
- **Lock/cross**: best bid >= best ask, only valid in modeled auction/transition states.
- **Order flow imbalance (OFI)**: event-based (queue changes), distinct from level imbalance `(bid_size - ask_size)/(bid_size + ask_size)`.

## Mental Models
- Use **"price-formation process"** when evaluating a backtest: spread, depth, and queue are not just cost, they are part of the price formation process.
- Use **midprice for signals and executable prices for PnL** when measuring markouts.
- Use **dollar bars** when you only have trades and want an ML default robust to price-level changes.
- Use **the venue-local book as a microstructure lens**, not as total executable liquidity (hidden and internalized flow are invisible).
- Use **invariants and cross-field consistency** before statistical filters when validating microstructure.

## Anti-patterns
- **Assuming a fill at the close at the bar price**: ignores spread, depth, and queue (misspecification of the process).
- **Interpolating missing ticks/quotes**: fabricates activity; flag and exclude.
- **Treating exchange timestamps as decision time**: lookahead from observation/processing latency.
- **EWMA with high α in imbalance bars**: threshold spiral (17.5x inflation in NVDA).
- **Using the tick test alone when direction matters**: ~78-80% vs ~95% for Lee-Ready.
- **Flagging |r/σ_daily| > 4 as a jump**: under-detects on stressed days and over-detects on calm days.
- **Treating trade nulls in extended hours as missing data**: 20-23% of bars with no executions in AlgoSeek NASDAQ-100 bias aggregates.
- **Raw imbalance as a trading rule**: weak and noisy correlation; at 1s, spread and queue dominate.
- **Dollar bars without rescaling for splits**.

## Code Examples
Compact reconstruction of LOB state following the described model (registry + price-level view):
```python
orders = {}
levels = {}

def add(ref, side, price, size):
    assert ref not in orders
    orders[ref] = [side, price, size]
    levels[(side, price)] = levels.get((side, price), 0) + size

def reduce(ref, size):
    side, price, remaining = orders[ref]
    assert size <= remaining
    orders[ref][2] -= size
    levels[(side, price)] -= size
    if levels[(side, price)] == 0:
        del levels[(side, price)]
    if orders[ref][2] == 0:
        del orders[ref]

def replace(old_ref, new_ref, new_price, new_size):
    side = orders[old_ref][0]
    reduce(old_ref, orders[old_ref][2])
    add(new_ref, side, new_price, new_size)
```
What it demonstrates: the LOB as a state machine with a registry by order ID and per-level totals; X/E/C map to `reduce`, D to `reduce` for the remainder, U to `replace`, with invariants (existing reference, non-negative quantity, level removed at zero).

## Reference Tables
| Classification method (NVDA, 742,527 trades) | Coverage | Accuracy |
|---|---|---|
| Lee-Ready (quote + tick) | 100% | 94.7% |
| Continuous tick test | 100% | 80.0% |
| Nonzero-only tick test | 19% | 90.0% |

| Bar | Clock | Target | Typical use |
|---|---|---|---|
| Time | Calendar | Interval | Clock-time forecasting |
| Tick | Activity | Trade count | Equal trade frequency |
| Volume | Activity | Shares | Activity by size |
| Dollar | Activity | Traded value | ML default |
| Tick imbalance | Information | Signed count | Directional pressure |
| Volume imbalance | Information | Signed volume | Informed flow by size |
| Run | Information | Side dominance | Persistent one-sided flow |

| Bar (AAPL, ITCH, 14,184 trades) | N bars | Jarque-Bera | Autocorr lag-1 |
|---|---|---|---|
| Time 1 min | 390 | 609.0 | -0.023 |
| Volume 10K | 96 | 1.9 | 0.026 |
| Dollar $3M | 102 | 2.2 | -0.027 |
| Tick imbalance | 139 | 22.6 | 0.057 |
| Tick run | 455 | 12.5 | -0.007 |

| Bar (NVDA, direct aggressor, 195,420 trades) | N bars | JB | Autocorr |
|---|---|---|---|
| Time 1 min | 390 | 132.8 | 0.014 |
| Tick 500 | 390 | 58.8 | -0.061 |
| Volume 50K | 380 | 2.3 | -0.017 |
| Dollar $5M | 519 | 33.0 | 0.039 |
| Vol imbalance | 691 | 51.0 | 0.035 |

| ITCH parser (13 GB, 423M messages) | Python | Rust |
|---|---|---|
| Time | ~23 min | < 5 min |
| Peak memory | ~8 GB | < 500 MB |

Minimum LOB invariants: references exist before modify/execute/cancel; remaining quantity never negative (order and level); valid prices (tick size, bands); depth per level = sum of orders.

## Worked Example
ITCH Jan 30, 2020 (423M messages): 186.6M adds (A+F), 180.3M deletes, 5.0M partial cancels, 36.8M replaces, 8.6M executions; cancellation rate 96.6% (orders with delete / adds), execution rate 3.4%. 41% of cancellations occur in < 500 ms, 50% within 1s (median 0.99s), 80% within 10s; executions: 10% in < 1 ms, median 6.1 s, 1% wait > 40 min. Lesson: displayed liquidity is ephemeral (fleeting orders) and must be modeled carefully. Visible Nasdaq spread for NVDA: $0.01 at midday to $0.06 near the close. Imbalance vs returns across 50 stocks: weak average correlation (range -0.35 to 0.16): not tradable without conditioning.
Imbalance bar calibration: with a 47% buy fraction in NVDA, standard α inflated expected bar size 17.5x over the day; α=0.001 kept it at 0.98x. Rules: small α (e.g., 0.001), warmup of 100+ bars, monitor the expected_imbalance column; in production consider fixed thresholds.
Jumps (AlgoSeek, AMD/AMZN/FB 2020, 5-min bars, α=0.01, K=12, critical value 5.10): 0.37 jumps per symbol-day; 24%/31%/34% of days with at least one jump; jumps contribute ~6-10% of annual realized variance (6.8% AMD, 10.4% AMZN, 6.2% FB); 27% of jumps in the last 30 min; Lee-Mykland vs naive rule: 70 vs 55 (AMD), 103 vs 48 (AMZN), 109 vs 49 (FB). Features: jump_count, signed_jump_var, jump_variance, jump_share.

## Key Takeaways
- Model execution with spread, depth, and queue; use executable prices in markouts.
- Choose the feed by strategy: TAQ for medium horizon, L2/L3 only if it depends on depth/queue.
- Reconstruct the LOB as a state machine with invariants and validate against BBO/NBBO.
- Default: dollar bars; Lee-Ready (or direct aggressor) for information bars with a small alpha.
- Detect jumps with bipower local volatility (Lee-Mykland) and use them for label conditioning (skip 30 min after a jump) and as features.
- Validate sequence, qualifiers, and sessions; flag and exclude instead of interpolating; log the rules.

## Connects To
Ch 2 (data quality, corporate actions, bid-ask in OTC vs exchange), Ch 6 (order flow reversal strategy), Ch 7 (label conditioning, multiple testing), Ch 8 (OFI, trade imbalance, microstructure features, minute bars), Ch 9, Ch 18 (Kyle lambda, cost and impact models), Ch 11.

## Code Reference

Official repository: [03_market_microstructure/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/03_market_microstructure). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_itch_parser.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/01_itch_parser.ipynb) + `.py` | 3.3 | This notebook demonstrates how to parse NASDAQ's TotalView-ITCH binary protocol. |
| [`02_itch_lob_reconstruction.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/02_itch_lob_reconstruction.ipynb) + `.py` | 3.3 | This notebook reconstructs the limit order book (LOB) from ITCH messages. |
| [`03_itch_lob_analysis.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/03_itch_lob_analysis.ipynb) + `.py` | 3.3 | This notebook analyzes reconstructed order book data to demonstrate empirically-grounded patterns with predictive power. |
| [`04_itch_order_lifecycle_analysis.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/04_itch_order_lifecycle_analysis.ipynb) + `.py` | 3.3 | This notebook analyzes the complete lifecycle of limit orders using NASDAQ ITCH data, revealing critical insights about cancellation rates, time dynamics, and the role o… |
| [`05_itch_trading_activity.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/05_itch_trading_activity.ipynb) + `.py` | 3.3 | This notebook provides a high-level view of NASDAQ trading activity using TotalView-ITCH data. |
| [`06_itch_intraday_patterns.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/06_itch_intraday_patterns.ipynb) + `.py` | 3.3 | This notebook analyzes intraday volume patterns from NASDAQ ITCH data, demonstrating the well-documented U-shape. |
| [`07_itch_stylized_facts.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/07_itch_stylized_facts.ipynb) + `.py` | 3.3 | This notebook demonstrates fundamental microstructure patterns: order arrival/cancellation dynamics, the bid-ask bounce phenomenon, and the liquidity spectrum across sto… |
| [`08_databento_lob_reconstruction.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/08_databento_lob_reconstruction.ipynb) + `.py` | 3.3 | Part 1 of 3: LOB Reconstruction → Order Flow Analysis → Bar Calibration Uses mbo_data data. |
| [`09_databento_mbo_analysis.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/09_databento_mbo_analysis.ipynb) + `.py` | 3.3 | Part 2 of 3: LOB Reconstruction → Order Flow Analysis → Bar Calibration Uses databento_day, mbo_data data. |
| [`10_iex_lob_reconstruction.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/10_iex_lob_reconstruction.ipynb) + `.py` | 3.3 | This notebook demonstrates LOB reconstruction using IEX's free historical market data, providing a reader-runnable alternative to the NASDAQ ITCH examples in earlier not… |
| [`11_algoseek_taq_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/11_algoseek_taq_eda.ipynb) + `.py` | 3.3 | On March 16, 2020, the S&P 500 fell 12% - its worst single-day drop since 1987. |
| [`12_algoseek_taq_lob_reconstruction.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/12_algoseek_taq_lob_reconstruction.ipynb) + `.py` | 3.3 | Every trade has two sides: a buyer and a seller. |
| [`13_algoseek_minute_bars_eda.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/13_algoseek_minute_bars_eda.ipynb) + `.py` | 3.3 | This notebook provides a comprehensive exploration of the AlgoSeek TAQ minute bar dataset. |
| [`14_itch_bar_sampling.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/14_itch_bar_sampling.ipynb) + `.py` | 3.4 | Traditional OHLCV bars sample data at fixed time intervals. |
| [`15_itch_lee_ready.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/15_itch_lee_ready.ipynb) + `.py` | 3.4 | This notebook validates our Lee-Ready implementation against DataBento's ground truth aggressor labels. |
| [`16_itch_information_bars.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/16_itch_information_bars.ipynb) + `.py` | 3.4 | This notebook has two parts: Uses mbo_data, trades data. |
| [`17_databento_bar_sampling.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/17_databento_bar_sampling.ipynb) + `.py` | 3.4 | Part 3 of 3: LOB Reconstruction → Order Flow Analysis → Bar Calibration Uses mbo_data, multiday_trades data. |
| [`18_algoseek_jump_detection.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/03_market_microstructure/18_algoseek_jump_detection.ipynb) + `.py` | 3.5 | Decomposes daily realized variance into continuous (bipower) and jump components on AlgoSeek minute bars; applies the Lee–Mykland nonparametric test to time individual j… |
