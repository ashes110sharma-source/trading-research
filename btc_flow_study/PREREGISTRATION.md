# BTC flow study - rules fixed before any data is seen

Written 10 Nov 2026. Nothing below may be changed after the dev results are known. A changed rule is a new study.

## Question
Can futures-market data (funding, open interest, positioning, taker flow) produce 1-2 BTC trades a day that make
money after costs? Price patterns alone could not: about 750 rule combinations tested on 2020-23 were zero before costs.

## Data
BTCUSDT perpetual futures, Binance public archive (data.binance.vision): 1-min candles with taker-buy volume,
funding-rate history, and the 5-min "metrics" files (open interest, long/short ratios). No liquidation feed is used;
forced liquidations are approximated by a sharp move together with a drop in open interest.

## Windows
- DEV: from 45 days after the first available bar to 31 Dec 2024. The dev run cuts the data at 1 Jan 2025 before computing anything.
- TEST: 1 Jan 2025 onward. Opened once, for one setup, only if the pass mark below is met.

## Trading rules shared by every setup
- Decisions on 15-min candles; entry at the next candle's open; exits checked on 5-min candles.
- Stop 0.25 x ADR20. Target 2 x the stop distance. Time exit after 4 or 8 hours. Stop wins if both fall in one 5-min candle.
- At most 2 trades per UTC day, one open at a time.
- Costs: 0.10% round trip on every trade, plus 0.02% extra slippage on stop exits.
- An open-interest snapshot stamped T is treated as known only from T + 5 minutes.

## The six ideas (24 cells = 12 setups x 2 holding times)
| | Idea | Trade |
|---|---|---|
| H1 | Liquidation flush: 30-min move of 0.25 / 0.35 ADR while open interest drops 0.5% / 1.0% | fade the move |
| H2 | Funding rate in the top or bottom 10% / 5% of the last 90 funding events | fade the crowd paying |
| H3 | Break of the 4-hour range while open interest rose 0.5% / 1.0% in the last hour | follow the breakout |
| H3b | Break of the 4-hour range while open interest fell 0.5% / 1.0% | fade the breakout |
| H4 | Taker buying (selling) in the top 5% / 2% of the last 7 days but price did not rise (fall) | fade the aggressor |
| H5 | Retail long/short account ratio in the top or bottom 10% / 3% of the last 30 days | fade retail |

## Pass mark
Take the cell with the highest t-stat AFTER costs among cells with at least 150 dev trades. It qualifies for the test only if
- t-stat after costs is 2.5 or more, and
- at least 60% of its calendar quarters are positive.

If it does not qualify, no test is run and the study ends with "no edge found".
In the test, t-stat after costs of 2 or more = passed; positive but below 2 = not proven; negative = failed.

## Known limits
- 24 cells are tried, so the 2.5 bar is deliberately above the usual 2.
- Open-interest history in the archive starts later than price history; the dev window is as long as that data allows.
- Results are for Binance perpetual prices and assumed costs. They are research, not a recommendation to trade.
