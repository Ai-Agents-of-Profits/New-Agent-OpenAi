/**
 * Fine-Tuned Trading Strategy Prompts and Descriptions for the Trading Assistant
 */

export const strategyPrompts = {
  "precision_trade": `Execute a comprehensive precision trading analysis for BTC/USDT on the 1h timeframe, emphasizing high-confidence setups only.

Steps to follow:
1. Generate a detailed Token Dashboard (detail level="high") to aggregate Market Data, Technical Analysis, and Orderbook insights.
2. Identify and confirm the dominant short-term trend, key support/resistance levels, and recent price pivots.
3. Conduct thorough Technical Analysis:
   - Precisely evaluate momentum and trend indicators (RSI, MACD, EMA crossovers).
   - Identify clear chart patterns (breakouts, trend continuations, or reversals).
   - Measure volatility through Bollinger Bands and ATR.
4. Perform meticulous Orderbook Analysis:
   - Confirm liquidity depth at identified support/resistance levels.
   - Detect significant buy/sell walls, real-time order imbalances, and immediate supply/demand dynamics.
   - Assess bid-ask spread tightness for execution confidence.
5. Cross-validate insights:
   - Align market sentiment and macro-level news events from dashboard insights.
   - Confirm setup strength by identifying confluence across Technical, Orderbook, and Market Data.

Trade Setup:
- Clearly state if conditions for an ultra-high-confidence trade are met:
  - Provide exact entry, stop loss, and take profit levels, with concise and specific reasoning based on aggregated evidence.
  - Explicitly define the trade validity period with precision (e.g., "valid until next 2 hourly candle closes," "expires if price breaches [exact price level]," "must execute within next 1-hour candle").
- If no valid high-confidence setup exists, clearly list specific reasons and conditions preventing execution (e.g., ambiguous indicator alignment, insufficient liquidity confirmation, conflicting volume signals).`
};

export const strategyDescriptions = {
  "precision_trade": "Delivers exceptionally precise trade setups through exhaustive validation across market data, technical indicators, and orderbook liquidity. Focused exclusively on ultra-high-confidence signals, it provides explicitly defined entry, stop loss, take profit, and strict signal validity periods to ensure optimal timing and clarity."
};
