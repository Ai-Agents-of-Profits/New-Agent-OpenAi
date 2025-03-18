/**
 * Fine-Tuned Trading Strategy Prompts and Descriptions for the Trading Assistant
 */

export const strategyPrompts = {
  "breakout": `Perform a comprehensive analysis for BTC/USDT on the 1h timeframe.

Steps to follow:
1. First, generate a complete token dashboard using the Token Dashboard Agent to get comprehensive market insights.
2. Review current market price, volume patterns, and market structure from the dashboard.
3. Analyze the technical indicators, especially focusing on volatility measures, momentum, and price action patterns.
4. Examine the orderbook data to confirm liquidity at critical support/resistance levels.

Trade Setup:
- If a valid breakout trade setup exists, provide detailed entry, stop loss, and take profit levels with specific reasoning.
- CLEARLY SPECIFY THE SIGNAL VALIDITY PERIOD: Include exactly how long this signal remains valid (e.g., "valid for next 4 hours", "valid until next 4h candle close", "valid for the next 24 hours").
- If no valid setup exists, explain which conditions (e.g., weak technical signals, insufficient volume, or lack of liquidity) are unfavorable.`,
  
  "trend_following": `Conduct an in-depth trend analysis for BTC/USDT on the 1h timeframe.

Examine:
1. First, generate a complete token dashboard using the Token Dashboard Agent for a holistic market view.
2. Use the dashboard data to identify the dominant market trend (uptrend, downtrend, or sideways) across multiple timeframes.
3. Evaluate momentum strength with indicators from the dashboard such as RSI, MACD, and trend metrics.
4. Analyze moving average crossovers, especially between the 50 EMA and 200 EMA.
5. Review volume patterns and liquidity distribution to confirm trend strength.
6. Consider overall market sentiment and macro-level trends for further confirmation.

Trade Setup:
- If conditions indicate a high-probability trend trade, detail entry, stop loss, and take profit levels with a specified validity timeframe.
- CLEARLY SPECIFY THE SIGNAL VALIDITY PERIOD: Define the exact timeframe during which this setup remains valid (e.g., "valid for the next 8 hours", "valid until daily candle close", "expires after 3 consecutive 1h candles of counter-trend movement").
- Otherwise, provide a clear explanation of why the trade isn't advisable.`,
  
  "mean_reversion": `Analyze BTC/USDT on the 1h timeframe for mean reversion opportunities.

Focus on:
1. First, generate a complete token dashboard using the Token Dashboard Agent to assess overall market conditions.
2. Use the dashboard's technical indicators to detect overextended price moves (RSI, Stochastic, or CCI).
3. Measure deviations from key moving averages (e.g., 20 SMA) and volatility bands (Bollinger Bands).
4. Evaluate historical price behavior from the dashboard to validate potential reversals.
5. Examine the orderbook data to confirm reversal signals with volume trends.
6. Assess the risk/reward profile before taking a position.

Trade Setup:
- If favorable reversal conditions exist, provide a detailed trade setup including entry, stop loss, and take profit levels along with the timeframe validity.
- CLEARLY SPECIFY THE SIGNAL VALIDITY PERIOD: State the exact duration for which this reversal signal remains actionable (e.g., "valid for next 6 hours", "expires if price breaks beyond [specific level]", "requires entry within next 3 hourly candles").
- If not, explain the factors negating the trade opportunity.`,
  
  "comprehensive": `Perform an integrated market analysis for BTC/USDT on the 1h timeframe.

Include:
1. First, generate a complete token dashboard using the Token Dashboard Agent with a "high" detail level.
2. Review the dashboard for multi-timeframe trend analysis (short, medium, and long-term perspectives).
3. Identify key support/resistance levels and pivot points from historical price data.
4. Analyze abnormal volume patterns detected in the dashboard.
5. Study the confluence of technical indicators (RSI, MACD, Bollinger Bands, etc.).
6. Examine order book analysis to identify immediate supply and demand imbalances.
7. Consider market sentiment and any relevant news events.

Trade Setup:
- If a viable trade setup is found, detail the entry, stop loss, and take profit levels and specify the timeframe for which the setup holds.
- CLEARLY SPECIFY THE SIGNAL VALIDITY PERIOD: Provide a precise timeframe for signal validity (e.g., "valid for next 12 hours", "reassess after 4h candle close", "expires if volume drops below [specific level]").
- Otherwise, clearly explain which conditions are not met.`,
  
  "swing_trade": `Evaluate BTC/USDT on the 1h timeframe with a swing trading approach.

Examine:
1. First, generate a complete token dashboard using the Token Dashboard Agent, looking back at least 14 days.
2. Use the dashboard data to identify key swing highs and swing lows relative to the current trend.
3. Confirm the prevailing trend using medium-term moving averages from the dashboard.
4. Study momentum divergences at swing points via RSI, MACD, or similar indicators.
5. Analyze volume patterns to validate swing moves and potential reversals.
6. Determine optimal entry and exit points with robust risk management parameters.

Trade Setup:
- Provide a detailed trade setup including entry, stop loss, and take profit levels, along with the timeframe in which these levels are valid.
- CLEARLY SPECIFY THE SIGNAL VALIDITY PERIOD: Define the exact duration for which this swing setup remains valid (e.g., "valid for next 2-3 days", "reassess if price breaks [specific level]", "requires confirmation within next 6 hours").
- If conditions are not favorable, clearly outline the shortcomings.`,
  
  "scalping": `Perform a rapid scalping analysis for BTC/USDT on the 1m and 5m timeframes.

Focus on:
1. First, generate a token dashboard using the Token Dashboard Agent with timeframe="5m" and days_back=1.
2. Focus on real-time order book dynamics and market microstructure from the dashboard.
3. Identify high-probability support and resistance zones with swift price action validation.
4. Employ fast-paced momentum indicators (RSI, Stochastic) optimized for short intervals.
5. Analyze bid-ask spread behavior and liquidity concentrations from the orderbook data.
6. Confirm signals with minimal latency using high-frequency volume and order flow data.

Trade Setup:
- If a rapid trade setup is detected, specify detailed entry, stop loss, and take profit levels with the applicable timeframe.
- CLEARLY SPECIFY THE SIGNAL VALIDITY PERIOD: Provide exact timing for signal expiration (e.g., "valid for next 15-30 minutes", "expires after 3 consecutive 5m candles", "requires execution within 5 minutes").
- Otherwise, explain why the conditions do not support a scalping trade.`
};

export const strategyDescriptions = {
  "breakout": "Detects significant price breakouts by confirming strong volume, volatility, and order flow imbalances at key levels. Includes detailed trade setups with entry, stop loss, and take profit levels, along with timeframe validity or explanations when conditions are unfavorable.",
  "trend_following": "Identifies sustained market trends using robust momentum indicators and moving average confirmations. Provides precise trade setups when high-probability trends are detected, with detailed levels and timeframe specifications.",
  "mean_reversion": "Targets extreme price deviations from established averages, indicating potential reversals with favorable risk/reward ratios. Offers trade setups with clear entry, stop loss, and take profit details, or explains why conditions are unsuitable.",
  "comprehensive": "Integrates multi-dimensional analysis—including trend, volume, technical indicators, and order book insights—for a holistic market view. Details complete trade setups with timeframe validity or outlines factors that negate a trade opportunity.",
  "swing_trade": "Focuses on capturing medium-term price moves by pinpointing optimal swing highs/lows and entry/exit points using validated signals. Provides comprehensive trade setups including entry, stop loss, and take profit levels within specified timeframes.",
  "scalping": "Executes rapid trades based on minute-level price action, leveraging precise order book analysis and high-frequency technical indicators. Outlines quick trade setups with clear entry, stop loss, and take profit points, and specifies the timeframe for the setup."
};
