"""
Trading strategy prompts for generating market signals.
"""

# Predefined sophisticated prompts for different trading strategies
TRADING_PROMPTS = {
    "breakout": """
    Analyze the price action for {symbol} on the {timeframe} timeframe and identify potential breakout opportunities.
    
    Focus on:
    1. Key resistance and support levels that price might break through
    2. Volume patterns confirming breakout potential
    3. Recent consolidation or range-bound patterns
    4. Increasing volatility that might precede a breakout
    
    Provide a comprehensive breakout analysis with:
    - Price targets if a breakout occurs (both upside and downside scenarios)
    - Key confirmation indicators to validate a true breakout
    - Potential entry points, stop loss levels, and take profit targets
    - Risk/reward ratio assessment for the identified opportunities
    
    Be specific about price levels and use concrete data in your analysis.
    Conclude with a summary of whether a high-probability breakout setup exists.

    ## Trade Execution Guide (if valid signal exists)
    
    If a high-probability breakout setup exists, provide specific execution instructions:
    
    1. **Order Execution Details:**
       - Symbol: {symbol}
       - Order Type: Recommend using a limit order at a specific level or a market order on confirmation
       - Entry Price: Specific price level where the trade should be executed
       - Position Size: Recommend 50-100% of account balance
       - Side: Clearly specify long (buy) or short (sell) direction
    
    2. **Risk Management Parameters:**
       - Stop Loss
       - Take Profit
       - Leverage: Recommended leverage level (10-50x)
    
    
    Include the exact parameters for the Execution Agent to implement the trade.
    """,
    
    "trend_following": """
    Conduct a detailed trend analysis for {symbol} on the {timeframe} timeframe.
    
    Examine:
    1. The current market structure and overall trend direction (uptrend, downtrend, or ranging)
    2. Strength of the trend using momentum indicators (RSI, MACD, ADX)
    3. Moving average alignments and crossovers (50 EMA, 200 EMA)
    4. Recent pullbacks and their relation to key support/resistance levels
    
    Provide a trend-following analysis with:
    - Clear trend identification and strength assessment
    - Potential entry zones aligned with the prevailing trend
    - Recommended stop loss placement to manage risk
    - Take profit targets based on trend projection and key levels
    - Risk/reward assessment for the identified opportunities
    
    Be data-driven in your analysis and include specific price levels.
    Conclude with a high-conviction trend-following trade setup if one exists.

    ## Trade Execution Guide (if valid signal exists)
    
    If a high-conviction trend-following setup exists, provide specific execution instructions:
    
    1. **Order Execution Details:**
       - Symbol: {symbol}
       - Order Type: Specify market or limit (with exact price) based on entry urgency
       - Entry Price or Zone: Specific price level or zone for entry (e.g., pullback to support)
       - Position Size: Recommend position size based on distance to stop loss (1-2% account risk)
       - Side: Clearly indicate long (buy) for uptrend or short (sell) for downtrend
    
    2. **Risk Management Parameters:**
       - Stop Loss: Place below recent swing low (for longs) or above recent swing high (for shorts)
       - Take Profit: Multiple targets based on previous swing extensions or Fibonacci projections
       - Leverage: Conservative leverage recommendation based on trend strength and volatility
    
    
    Include the exact parameters for the Execution Agent to implement the trade efficiently.
    """,
    
    "mean_reversion": """
    Analyze {symbol} on the {timeframe} timeframe for mean reversion opportunities.
    
    Focus on:
    1. Overbought/oversold conditions using oscillators (RSI, Stochastic, CCI)
    2. Deviation from key moving averages or bands (Bollinger Bands, Keltner Channels)
    3. Historical volatility patterns and current volatility regime
    4. Key support/resistance levels that could serve as reversal points
    
    Provide a mean reversion analysis with:
    - Identification of potential price exhaustion points
    - Statistical perspective on the extent of the deviation from mean
    - Recommended entry points, stop levels, and profit targets
    - Risk/reward assessment for the counter-trend opportunity
    
    Include specific price levels and probabilities in your analysis.
    Conclude with a clear mean reversion trade recommendation if conditions are favorable.

    ## Trade Execution Guide (if valid signal exists)
    
    If a favorable mean reversion setup exists, provide specific execution instructions:
    
    1. **Order Execution Details:**
       - Symbol: {symbol}
       - Order Type: Limit order preferred (specify exact price) for entry at extremes
       - Entry Price: Precise price level at or near extreme deviation points
       - Position Size: Recommend size based on volatility and distance to stop (max 1-2% account risk)
       - Side: Specify counter-trend direction (buy during oversold, sell during overbought)
    
    2. **Risk Management Parameters:**
       - Stop Loss: Precise level beyond which the mean reversion thesis is invalidated
       - Take Profit: Target the mean (moving average) or previous equilibrium point
       - Leverage: Conservative recommendation (1-3x) due to counter-trend nature of these trades
    
    3. **Advanced Execution Options:**
       - Consider scaling in at multiple extreme levels rather than single entry
       - Implement tight trailing stop once price begins reverting to the mean
       - Set specific time-based invalidation if reversion doesn't occur within expected timeframe
    
    Include exact parameters required for the Execution Agent to implement this mean reversion trade.
    """,
    
    "comprehensive": """
    Perform a comprehensive market analysis for {symbol} on the {timeframe} timeframe.
    
    Include:
    1. Current market structure and trend analysis (short, medium, and long-term perspectives)
    2. Key support/resistance levels and price action patterns
    3. Volume analysis and unusual volume patterns
    4. Technical indicator confluence or divergence (RSI, MACD, Bollinger Bands, etc.)
    5. Volatility assessment and potential price ranges
    6. Order book analysis for immediate supply/demand imbalances if available
    
    Provide a detailed trading analysis with:
    - Multiple scenario planning (bullish, bearish, and neutral cases)
    - Potential entry points with clear invalidation levels
    - Risk-optimized position sizing recommendations
    - Multiple take profit targets with associated probabilities
    - Overall risk/reward assessment
    
    Be data-driven and specific about price levels, percentages, and probabilities.
    Conclude with your highest conviction trading opportunity considering all factors.

    ## Trade Execution Guide (if valid signal exists)
    
    If a high-conviction trading opportunity is identified, provide detailed execution instructions:
    
    1. **Order Execution Details:**
       - Symbol: {symbol}
       - Order Type: Recommend optimal order type based on market conditions
       - Entry Price or Zone: Specific entry level with acceptable range
       - Position Size: Precise calculation based on account size and risk percentage (1-2%)
       - Side: Clearly specify direction (long/buy or short/sell)
    
    2. **Risk Management Parameters:**
       - Stop Loss: Exact price level with explanation of why this level invalidates the thesis
       - Take Profit: Multiple targets with specific price levels and suggested allocation percentages
       - Leverage: Specific recommendation based on volatility and trade confidence
    
    3. **Advanced Execution Options:**
       - Detailed entry method (single entry vs. scaled entries at specific levels)
       - Progressive stop-loss management strategy with trigger points
       - Trailing stop implementation details once specific profit thresholds are reached
       - Position management rules for adding or reducing position size based on price action
    
    Include comprehensive parameters for the Execution Agent to implement this trade with precision.
    """,
    
    "swing_trade": """
    Analyze {symbol} on the {timeframe} timeframe specifically for swing trading opportunities.
    
    Examine:
    1. Current market structure in the context of larger timeframe trends
    2. Key swing high/low points and their relation to support/resistance
    3. Momentum indicators and potential divergences at swing points
    4. Volume patterns confirming or contradicting price action
    5. Optimal entry timing based on short-term price action
    
    Provide a swing trading analysis with:
    - Clear identification of the current swing pattern and its stage
    - Recommended entry zones with timing considerations
    - Protective stop placement to define maximum risk
    - Take profit targets based on previous swing magnitudes
    - Projected holding period and expected market behavior
    
    Include specific price levels and time projections in your analysis.
    Conclude with a well-defined swing trading setup if one is present.

    ## Trade Execution Guide (if valid signal exists)
    
    If a well-defined swing trading setup is identified, provide specific execution instructions:
    
    1. **Order Execution Details:**
       - Symbol: {symbol}
       - Order Type: Recommend limit orders near support/resistance levels or market on confirmation
       - Entry Price or Zone: Specific price level, potentially with scaled entry points
       - Position Size: Recommend 1-2% account risk adjusted for swing trade duration
       - Side: Clearly specify swing long (buy) or swing short (sell) direction
    
    2. **Risk Management Parameters:**
       - Stop Loss: Place below key swing low (for longs) or above key swing high (for shorts)
       - Take Profit: Multiple targets based on measured swing movements or Fibonacci extensions
       - Leverage: Conservative recommendation (1-3x) appropriate for multi-day positions
    
    3. **Advanced Execution Options:**
       - Specify holding period expectations (days to weeks)
       - Recommend position adjustment strategy at key price levels or time points
       - Suggest partial profit taking at specific technical levels
       - Provide criteria for early exit if swing pattern fails
    
    Include exact parameters for the Execution Agent to implement this swing trade effectively.
    """,

    "scalping": """
    Perform a detailed scalping analysis for {symbol} on the 1m and 5m timeframes.
    
    Focus on:
    1. Immediate order book dynamics and market microstructure
    2. High-precision support/resistance levels with price action validation
    3. Volume profile and tape reading to identify buying/selling pressure
    4. Short-term momentum indicators (RSI, Stochastic) calibrated for scalping
    5. Bid-ask spread patterns and liquidity concentration zones
    6. Real-time trading flow imbalances and potential stop-hunting zones
    
    Coordinate with the Orderbook Analysis Agent to examine:
    - Current bid/ask imbalances and potential short-term price direction
    - Limit order clustering at specific price points
    - Evidence of large player positioning
    - Spoofing or manipulation patterns
    
    Provide a scalping-focused analysis with:
    - Ultra-precise entry and exit points with price-level specificity
    - Immediate invalidation levels for quick risk management
    - Expected price movement magnitudes with time constraints
    - Optimal order types (market, limit, or conditional) for entry and exit
    - Multiple micro-target levels for partial position taking
    
    Include considerations for spread costs, execution slippage, and platform latency.
    Conclude with 2-3 specific scalping setups with exact price levels, expected duration (in minutes), 
    and quantified risk-reward metrics for each micro-movement opportunity.

    ## Trade Execution Guide (if valid signal exists)
    
    If viable scalping setups are identified, provide ultra-precise execution instructions:
    
    1. **Order Execution Details:**
       - Symbol: {symbol}
       - Order Type: Specify exact order type (usually limit for better fills, market only when necessary)
       - Entry Price: Precise to the pip/tick level based on order book structure
       - Position Size: Recommend larger percentage (2-5%) due to tight stops but still maintaining overall risk management
       - Side: Clearly state long/buy or short/sell with immediate execution timing
    
    2. **Risk Management Parameters:**
       - Stop Loss: Ultra-tight stop loss placed precisely beyond local support/resistance (specific price)
       - Take Profit: Multiple targets at specific price levels for rapid partial exits (recommend 25%, 50%, 25% splits)
       - Leverage: Specific recommendation (5-10x) appropriate for scalping, adjusted based on volatility
    
    3. **Advanced Execution Options:**
       - Implement time-based stop loss (exit if profit target not reached within X minutes)
       - Specific instructions for partial exits at exact price levels for quick base hits
       - Immediate trailing stop implementation once minimum profit threshold reached
       - Hard rules for position abandonment if price action shifts rapidly
    
    4. **Rapid Execution Workflow:**
       - Exact sequence of orders to place (entry, stop, take profits)
       - Specific instructions for order modifications based on price action in first 30-60 seconds
       - Clear criteria for adding to position if initial direction confirmed
    
    Include comprehensive but concise parameters for the Execution Agent to implement these rapid scalping trades with precision and speed.
    """
}

# Descriptions for the different trading strategies
STRATEGY_DESCRIPTIONS = {
    "breakout": """
    Breakout Trading Strategy
    
    The breakout trading strategy focuses on identifying and capitalizing on price movements that breach established support or resistance levels. This strategy is based on the principle that once price breaks through a significant level, it often continues to move in that direction with momentum.
    
    Key Components:
    
    1. Support/Resistance Identification: Identifying key price levels where the market has previously reversed or consolidated.
    
    2. Consolidation Recognition: Looking for periods where price trades in a narrow range, forming patterns like triangles, rectangles, or flags before a breakout occurs.
    
    3. Volume Confirmation: Using volume analysis to validate breakouts, as genuine breakouts typically occur with increased trading volume.
    
    4. False Breakout Awareness: Strategies to avoid or manage false breakouts, including waiting for candle closes beyond the breakout level.
    
    5. Momentum Assessment: Using indicators like RSI or MACD to confirm the strength of the breakout.
    
    Implementation:
    
    - Entry: When price conclusively breaks through a significant support or resistance level
    - Stop Loss: Placed below the breakout level for upside breakouts, or above for downside breakouts
    - Take Profit: Based on the measured move of the pattern or previous market swings
    
    This strategy works particularly well in volatile markets with clear trading ranges and during periods of news or fundamental catalysts that might trigger significant price movements.
    """,
    
    "trend_following": """
    Trend Following Strategy
    
    The trend following strategy aims to identify and follow established market trends, operating on the principle that prices tend to move in persistent directional trends over time. This approach focuses on "following the path of least resistance" rather than predicting reversals.
    
    Key Components:
    
    1. Trend Identification: Using tools like moving averages, trendlines, and higher timeframe analysis to determine the prevailing trend direction.
    
    2. Momentum Confirmation: Analyzing momentum indicators like RSI, MACD, or ADX to confirm trend strength.
    
    3. Pullback Recognition: Identifying temporary retracements within the trend that offer more favorable entry points.
    
    4. Moving Average Alignments: Using multiple moving averages (e.g., 50-day and 200-day) to confirm trend direction and strength.
    
    5. Risk Management: Implementing trailing stops to protect profits as the trend extends.
    
    Implementation:
    
    - Entry: When price shows continuation in the direction of the established trend, often after a pullback
    - Stop Loss: Below recent swing lows for uptrends, above recent swing highs for downtrends
    - Take Profit: Multiple targets based on extension of previous trend moves, or trailing stops
    
    This strategy typically performs best in markets with strong directional bias and extended trends, such as those driven by significant fundamental shifts or during major bull or bear market phases.
    """,
    
    "mean_reversion": """
    Mean Reversion Strategy
    
    The mean reversion strategy is based on the statistical concept that prices tend to return to their average or mean value over time. This approach looks for extreme price movements away from established averages as potential reversal opportunities.
    
    Key Components:
    
    1. Overbought/Oversold Identification: Using oscillators like RSI, Stochastic, or CCI to identify extreme market conditions.
    
    2. Deviation Measurement: Analyzing how far price has moved from moving averages or bands (Bollinger Bands, Keltner Channels).
    
    3. Statistical Analysis: Assessing standard deviations and historical volatility patterns to gauge the extremity of current price movements.
    
    4. Support/Resistance Confluence: Identifying when price reaches extreme levels that also coincide with historical support or resistance.
    
    5. Divergence Recognition: Looking for divergences between price and momentum indicators as potential reversal signals.
    
    Implementation:
    
    - Entry: When price reaches statistically extreme levels with confirmation of slowing momentum
    - Stop Loss: Beyond the extreme price point to limit risk if the deviation continues
    - Take Profit: At or near the mean value (such as a moving average) or previous support/resistance
    
    This strategy typically works best in range-bound markets or during periods of price consolidation, and may perform poorly during strong trending markets or when fundamental shifts occur.
    """,
    
    "comprehensive": """
    Comprehensive Market Analysis Strategy
    
    The comprehensive market analysis strategy integrates multiple analytical approaches and timeframes to develop a holistic view of market conditions. This strategy aims to identify high-probability trading opportunities by finding confluence among different analytical methods.
    
    Key Components:
    
    1. Multi-Timeframe Analysis: Examining short, medium, and long-term timeframes to understand the complete market context.
    
    2. Technical Integration: Combining price action, chart patterns, indicators, and volume analysis to form a comprehensive technical view.
    
    3. Market Structure Assessment: Analyzing the sequence of highs and lows to determine the market's structural phase (accumulation, markup, distribution, markdown).
    
    4. Scenario Planning: Developing multiple potential scenarios (bullish, bearish, and neutral) with associated probabilities and triggers.
    
    5. Order Flow Analysis: Incorporating order book data and liquidity analysis when available to identify significant support/resistance zones.
    
    Implementation:
    
    - Entry: Based on multiple confirmations across different analytical methods
    - Stop Loss: Placed at levels that invalidate the primary trading thesis
    - Take Profit: Multiple targets with partial position management at each level
    
    This strategy is adaptable to various market conditions but requires more sophisticated analysis and experience. It works particularly well during periods of market transition or when trading higher timeframes with significant capital at risk.
    """,
    
    "swing_trade": """
    Swing Trading Strategy
    
    The swing trading strategy aims to capture "swings" or medium-term price movements within larger trends. This approach focuses on holding positions for several days to weeks to profit from expected price movements between support and resistance levels.
    
    Key Components:
    
    1. Swing Point Identification: Locating key swing highs and lows within the overall market structure.
    
    2. Trend Within Range Recognition: Identifying the short-term trend within the context of larger market movements.
    
    3. Momentum Analysis: Using momentum indicators to confirm the strength of swing movements and identify potential exhaustion points.
    
    4. Entry Timing Optimization: Developing precise entry criteria based on price action patterns at support/resistance levels.
    
    5. Cycle Analysis: Understanding the typical duration and magnitude of market swings in the traded instrument.
    
    Implementation:
    
    - Entry: At or near potential reversal points with confirmation of price action
    - Stop Loss: Beyond the counter-swing point to define clear invalidation
    - Take Profit: Based on previous swing magnitudes or significant support/resistance levels
    
    This strategy works well in markets with clearly defined trading ranges or in trending markets with regular retracements. It's particularly suitable for traders who cannot monitor markets continuously and prefer less frequent but higher-probability trades.
    """,

    "scalping": """
    Scalping Strategy
    
    The scalping strategy focuses on capturing numerous small price movements within very short timeframes, typically minutes or even seconds. This high-frequency approach requires precision timing, ultra-fast execution, and deep understanding of market microstructure to profit from minimal price differentials.
    
    Key Components:
    
    1. Order Book Analysis: Reading the order book to identify immediate support/resistance levels, liquidity pools, and order imbalances that may indicate short-term price direction.
    
    2. Market Microstructure Understanding: Analyzing the bid-ask spread dynamics, market depth, and tape reading to interpret real-time trading flow.
    
    3. Ultra-Short-Term Technical Indicators: Utilizing fast-responding technical indicators like 1-minute RSI, Stochastic oscillators calibrated for scalping, and momentum indicators with shortened parameters.
    
    4. Price Action Patterns: Identifying micro patterns such as short-term double tops/bottoms, 1-minute engulfing patterns, and quick momentum bursts.
    
    5. Volume Profile Analysis: Monitoring real-time volume distribution to spot potential short-term exhaustion points or continuation signals.
    
    6. Algorithmic Trading Pattern Recognition: Understanding common algorithmic trading patterns and leveraging these patterns for entry and exit timing.
    
    Implementation:
    
    - Entry: Precision entries at key micro-support/resistance levels with confirmation from order flow
    - Stop Loss: Extremely tight stops, often just a few ticks or points away from entry
    - Take Profit: Multiple small targets, typically taking partial profits at predefined minimal favorable price movements
    - Position Sizing: Usually larger position sizes to compensate for smaller price movements
    - Trade Management: Immediate exit on any sign of unfavorable price action or failure to move quickly in the anticipated direction
    
    This strategy works best in highly liquid markets with tight spreads and low transaction costs. It requires intense focus, sophisticated tooling for market data visualization, and often benefits from specialized order types like iceberg orders, fill-or-kill, and immediate-or-cancel orders.
    
    Scalping is particularly effective during periods of range-bound activity or when volatility is low but consistent. It's also frequently employed around high-impact news events to capitalize on immediate price dislocations before larger moves establish themselves.
    """
}
