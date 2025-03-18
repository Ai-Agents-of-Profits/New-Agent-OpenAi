"""
Trading Strategy Agent - Uses sophisticated predefined prompts to interact with the 
main crypto market analysis agent and generate trading signals that can be handed off 
to the Binance Executor for trade execution.
"""
import os
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import ccxt.async_support as ccxt

from agents import Agent, Runner, handoff, Tool, RunContextWrapper
from src.exchange.connector import ExchangeConnector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trading Signal model
class TradingSignal(BaseModel):
    """Model for a trading signal with execution parameters"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    direction: str = Field(..., description="Trade direction ('buy' or 'sell')")
    entry_type: str = Field("market", description="Entry type ('market' or 'limit')")
    entry_price: Optional[float] = Field(None, description="Suggested entry price (None for market orders)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price level")
    take_profit: Optional[float] = Field(None, description="Take profit price level")
    position_size: float = Field(..., description="Trade quantity in base currency")
    leverage: Optional[int] = Field(1, description="Leverage to use (1-125)")
    confidence: float = Field(..., description="Signal confidence score (0-1)")
    timeframe: str = Field(..., description="Analysis timeframe used")
    reasoning: str = Field(..., description="Trading logic and reasoning")
    expiration: Optional[datetime] = Field(None, description="Signal expiration time")

# Predefined sophisticated prompts for different trading strategies
TRADING_PROMPTS = {
    "breakout": """
    Analyze {symbol} for breakout trading opportunities on the {timeframe} timeframe. 
    
    I need you to:
    1. Check if price is approaching or testing a significant resistance or support level
    2. Analyze volume patterns to confirm potential breakout validity
    3. Evaluate momentum indicators (RSI, MACD) to confirm directional strength
    4. Assess market structure for higher highs/higher lows (bullish) or lower highs/lower lows (bearish)
    5. Identify key levels for stop loss and take profit based on recent swing points
    
    First get the market price, then perform technical analysis, and finally analyze the orderbook to confirm liquidity at key levels.
    
    If a valid breakout trade setup exists, provide detailed entry, stop loss, and take profit levels with specific reasoning.
    If no valid setup exists, clearly explain why conditions aren't favorable.
    """,
    
    "trend_following": """
    Conduct a comprehensive trend analysis for {symbol} on the {timeframe} timeframe.
    
    Please provide:
    1. Clear identification of the current trend direction (bullish, bearish, or ranging)
    2. Trend strength assessment using ADX or similar indicators if available
    3. Key moving averages positions (are fast MAs above or below slow MAs?)
    4. Analysis of recent price action relative to these moving averages
    5. Volume analysis to confirm trend validity
    6. Identification of the most suitable entry point for a trend-following trade
    7. Logical stop loss placement (below recent swing low for uptrend, above recent swing high for downtrend)
    8. Multiple take profit targets with risk-reward ratios
    
    First get a market summary, then perform detailed technical analysis with a focus on trend indicators.
    
    Be specific with price levels for entry, stop loss, and take profit targets. If no clear trend exists, state this clearly.
    """,
    
    "mean_reversion": """
    Analyze {symbol} for mean reversion trading opportunities on the {timeframe} timeframe.
    
    Specifically evaluate:
    1. Current price position relative to key moving averages (20, 50, 200)
    2. Bollinger Band positioning - is price near or outside the bands?
    3. RSI readings to identify overbought (>70) or oversold (<30) conditions
    4. Historical volatility to assess normal price range behavior
    5. Recent price action to confirm potential reversal signals (candlestick patterns)
    6. Volume patterns to confirm exhaustion or reversal
    
    First get a market summary, then perform technical analysis with focus on oscillators and mean reversion indicators.
    
    If a strong mean reversion setup exists, provide specific entry price (limit order level), stop loss above/below recent swing, 
    and take profit at a realistic mean value. If conditions aren't right, explain why the setup is invalid.
    """,
    
    "comprehensive": """
    Generate a complete trading recommendation for {symbol} on the {timeframe} timeframe. 
    
    I need an extensive analysis including:
    1. Overall market context and sentiment
    2. Current price action and structure
    3. Key support and resistance levels
    4. Multiple technical indicator readings (trend, momentum, volatility)
    5. Order book analysis for liquidity and potential barriers
    6. Volume profile and unusual patterns
    7. Risk assessment based on volatility
    
    Start by generating a comprehensive token dashboard, then analyze specific entry, stop loss, and take profit levels.
    
    Provide a detailed directional bias with confidence level, optimal trade entry methods (market or limit), 
    precise stop loss placement, multiple take profit targets with risk-reward ratios, and suggested position sizing.
    If no high-probability trade exists, clearly state this with supporting evidence.
    """,
    
    "swing_trade": """
    Analyze {symbol} for swing trading opportunities on the {timeframe} timeframe.
    
    Evaluate the following:
    1. Market structure - clear higher highs/higher lows for uptrend or lower highs/lower lows for downtrend
    2. Key support and resistance levels, especially those tested multiple times
    3. Recent price action - is price at a logical level to enter a swing trade?
    4. Volume confirmation for potential moves
    5. Multiple indicator confluence (e.g., RSI, Stochastic, MACD) supporting the direction
    6. Risk-reward ratio of at least 1:2 for potential setups
    
    First get the market price, then perform technical analysis, and analyze the orderbook to confirm liquidity at key levels.
    
    If a valid swing trade exists, provide specific entry, stop loss, and take profit levels with reasoning for each.
    If no valid swing trade opportunity exists, explain the current market conditions and why it's better to wait.
    """
}

class TradingStrategyAgent:
    """
    Agent that uses predefined prompts to query the market analysis system
    and generate trading signals that can be handed off to the execution agent.
    """
    
    def __init__(self, market_agent: Agent, execution_agent: Optional[Agent] = None):
        """
        Initialize the trading strategy agent.
        
        Args:
            market_agent: The main market analysis orchestration agent
            execution_agent: Optional execution agent for trade execution
        """
        self.market_agent = market_agent
        self.execution_agent = execution_agent
        self.prompt_templates = TRADING_PROMPTS
        self.exchange = None
        self.active_trades = {}  # Store active trades for monitoring
        
        # Create the agent
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the trading strategy agent with tools"""
        
        tools = [
            Tool(
                name="generate_trading_signal",
                description="Generate a trading signal based on predefined strategy prompts",
                function=self._generate_signal,
                parameters={
                    "strategy": {"type": "string", "enum": list(TRADING_PROMPTS.keys()), "description": "Trading strategy type"},
                    "symbol": {"type": "string", "description": "Trading pair symbol (e.g., 'BTC/USDT')"},
                    "timeframe": {"type": "string", "description": "Timeframe for analysis (e.g., '15m', '1h', '4h', '1d')"},
                }
            ),
            Tool(
                name="list_strategies",
                description="List available trading strategy types",
                function=self._list_strategies,
                parameters={}
            ),
            Tool(
                name="get_strategy_description",
                description="Get detailed description of a trading strategy",
                function=self._get_strategy_description,
                parameters={
                    "strategy": {"type": "string", "enum": list(TRADING_PROMPTS.keys()), "description": "Trading strategy type"},
                }
            ),
            Tool(
                name="execute_trading_signal",
                description="Execute a trading signal by placing orders on Binance Futures",
                function=self._execute_trading_signal,
                parameters={
                    "symbol": {"type": "string", "description": "Trading pair symbol (e.g., 'BTC/USDT')"},
                    "direction": {"type": "string", "enum": ["buy", "sell"], "description": "Trade direction"},
                    "entry_type": {"type": "string", "enum": ["market", "limit"], "description": "Entry type (market or limit)"},
                    "entry_price": {"type": "number", "description": "Entry price (for limit orders, optional for market)"},
                    "position_size": {"type": "number", "description": "Position size in base currency"},
                    "stop_loss": {"type": "number", "description": "Stop loss price level (optional)"},
                    "take_profit": {"type": "number", "description": "Take profit price level (optional)"},
                    "leverage": {"type": "integer", "description": "Leverage to use (1-125, optional)"}
                }
            ),
            Tool(
                name="get_account_balance",
                description="Get account balance from Binance Futures",
                function=self._get_account_balance,
                parameters={}
            ),
            Tool(
                name="get_open_positions",
                description="Get all open positions on Binance Futures",
                function=self._get_open_positions,
                parameters={}
            ),
            Tool(
                name="get_open_orders",
                description="Get all open orders on Binance Futures",
                function=self._get_open_orders,
                parameters={
                    "symbol": {"type": "string", "description": "Trading pair symbol (e.g., 'BTC/USDT')", "required": False}
                }
            ),
            Tool(
                name="monitor_trade",
                description="Monitor a specific trading position",
                function=self._monitor_trade,
                parameters={
                    "symbol": {"type": "string", "description": "Trading pair symbol (e.g., 'BTC/USDT')"}
                }
            ),
            Tool(
                name="close_position",
                description="Close an open position on Binance Futures",
                function=self._close_position,
                parameters={
                    "symbol": {"type": "string", "description": "Trading pair symbol (e.g., 'BTC/USDT')"}
                }
            )
        ]
        
        instructions = """
        You are the Trading Strategy Agent, specializing in generating high-quality trading signals
        using sophisticated predefined prompts and comprehensive market analysis.
        
        Your primary goal is to identify optimal trading opportunities using various strategies:
        
        1. Breakout: Identify potential breakouts from key levels with volume confirmation
        2. Trend Following: Find high-probability entries in established trends
        3. Mean Reversion: Look for overbought/oversold conditions for counter-trend moves
        4. Comprehensive: Complete multi-factor analysis for highest-confidence signals
        5. Swing Trade: Identify swing trading opportunities at key market structure points
        
        For each strategy:
        1. Use the generate_trading_signal tool with the appropriate strategy type
        2. Review the analysis and determine if a valid setup exists
        3. If a trade should be executed, you can use the execute_trading_signal tool
        
        Present trading signals with:
        - Clear directional bias (buy/sell)
        - Specific entry price or price range
        - Precise stop loss level with reasoning
        - Take profit targets with risk-reward ratios
        - Confidence level and supporting evidence
        
        When there's no clear setup, honestly communicate that no trade is recommended.
        
        Always prioritize risk management and only suggest high-probability trades with 
        favorable risk-reward profiles.
        
        For signal execution:
        1. Validate the signal has all required parameters
        2. Use the execute_trading_signal tool to place orders
        3. Confirm execution and provide order IDs
        4. Set stop losses and take profits as appropriate
        
        For account monitoring:
        1. Use get_account_balance to check available funds
        2. Use get_open_positions to monitor current positions
        3. Use monitor_trade to track a specific position's performance
        4. Use close_position to exit a trade when appropriate
        """
        
        # Create the agent
        agent = Agent(
            name="Trading Strategy Agent",
            instructions=instructions,
            tools=tools
        )
        
        # Add execution agent as a handoff if provided
        if self.execution_agent:
            agent.handoffs = [
                handoff(
                    agent=self.execution_agent,
                    tool_description_override="Execute a trading signal by placing orders on Binance Futures"
                )
            ]
            
        return agent
    
    async def initialize_exchange(self):
        """Initialize the CCXT exchange connection to Binance Futures"""
        try:
            # Get API credentials from environment variables
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                logger.error("Binance API credentials not found in environment variables")
                return False
                
            # Initialize Binance Futures connection
            self.exchange = ccxt.binanceusdm({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'hedgeMode': False  # Set to True if using hedge mode
                }
            })
            
            # Load markets
            await self.exchange.load_markets()
            logger.info("Binance Futures exchange initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            return False
    
    async def _generate_signal(self, strategy: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate a trading signal using the specified strategy.
        
        Args:
            strategy: Strategy type from the predefined templates
            symbol: Trading pair symbol
            timeframe: Analysis timeframe
            
        Returns:
            Trading signal or analysis results
        """
        try:
            # Get the prompt template
            prompt_template = self.prompt_templates.get(strategy)
            if not prompt_template:
                return {"error": f"Strategy '{strategy}' not found"}
                
            # Format the prompt
            prompt = prompt_template.format(symbol=symbol, timeframe=timeframe)
            
            # Run the prompt through the market analysis agent
            logger.info(f"Generating {strategy} trading signal for {symbol} on {timeframe} timeframe")
            result = await Runner.run(self.market_agent, prompt)
            
            # Extract and parse the response
            response_text = result.final_output
            
            # Attempt to structure the response as a trading signal
            signal = self._parse_trading_signal(response_text, symbol, strategy, timeframe)
            
            return {
                "strategy": strategy,
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis": response_text,
                "trading_signal": signal
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {"error": str(e), "strategy": strategy, "symbol": symbol}
    
    def _parse_trading_signal(self, text: str, symbol: str, strategy: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Parse the text response from the market analysis agent into a structured trading signal.
        Uses heuristics to extract key information.
        
        Args:
            text: Response text from the market analysis agent
            symbol: Trading pair symbol
            strategy: Strategy type used
            timeframe: Analysis timeframe
            
        Returns:
            Structured trading signal or None if no signal could be parsed
        """
        # Default values
        signal = {
            "symbol": symbol,
            "direction": None,
            "entry_type": "market",
            "entry_price": None,
            "stop_loss": None, 
            "take_profit": None,
            "position_size": 0.01,  # Default small position size
            "leverage": 1,
            "confidence": 0.5,
            "timeframe": timeframe,
            "strategy": strategy,
            "reasoning": text,
            "raw_analysis": text
        }
        
        # Attempt to determine trade direction
        if "buy" in text.lower() or "long" in text.lower() or "bullish" in text.lower():
            signal["direction"] = "buy"
        elif "sell" in text.lower() or "short" in text.lower() or "bearish" in text.lower():
            signal["direction"] = "sell"
            
        # This is a simplified heuristic approach - in production, you'd want more sophisticated parsing
        # For this demo, we'll return None if we couldn't determine direction
        if not signal["direction"]:
            return None
            
        # For now, return the partial signal - in a real implementation, you'd
        # use NLP or regex to extract precise price levels for entry, stop loss, etc.
        return signal
    
    async def _execute_trading_signal(self, symbol: str, direction: str, entry_type: str, 
                                     position_size: float, entry_price: Optional[float] = None,
                                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                     leverage: int = 1) -> Dict[str, Any]:
        """
        Execute a trading signal by placing orders on Binance Futures.
        
        Args:
            symbol: Trading pair symbol
            direction: Trade direction ('buy' or 'sell')
            entry_type: Entry type ('market' or 'limit')
            position_size: Position size in base currency
            entry_price: Entry price (required for limit orders)
            stop_loss: Stop loss price level (optional)
            take_profit: Take profit price level (optional)
            leverage: Leverage to use (1-125)
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Initialize exchange if not already done
            if not self.exchange:
                success = await self.initialize_exchange()
                if not success:
                    return {"error": "Failed to initialize exchange connection"}
            
            # Validate inputs
            if entry_type == "limit" and not entry_price:
                return {"error": "Entry price is required for limit orders"}
                
            # Set leverage
            if leverage > 1:
                try:
                    await self.exchange.set_leverage(leverage, symbol)
                    logger.info(f"Set leverage to {leverage}x for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to set leverage: {e}")
            
            # Place the main entry order
            order_params = {}
            
            # For more complex orders like reducing margin or setting stop orders in the same call
            # if needed in the future, we can add more parameters
            
            # Place the order based on type
            logger.info(f"Placing {direction} {entry_type} order for {position_size} {symbol}")
            
            if entry_type == "market":
                main_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=direction,
                    amount=position_size,
                    params=order_params
                )
            else:  # limit order
                main_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=direction,
                    amount=position_size,
                    price=entry_price,
                    params=order_params
                )
            
            logger.info(f"Main order placed: {main_order['id']}")
            
            # Results to return
            results = {
                "main_order": main_order,
                "stop_loss_order": None,
                "take_profit_order": None
            }
            
            # Place stop loss if provided
            if stop_loss:
                # Stop loss is opposite side of the entry
                stop_side = "sell" if direction == "buy" else "buy"
                
                try:
                    sl_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='stop_market',
                        side=stop_side,
                        amount=position_size,
                        params={
                            'stopPrice': stop_loss,
                            'reduceOnly': True
                        }
                    )
                    logger.info(f"Stop loss order placed: {sl_order['id']}")
                    results["stop_loss_order"] = sl_order
                except Exception as e:
                    logger.error(f"Error placing stop loss order: {e}")
                    results["stop_loss_error"] = str(e)
            
            # Place take profit if provided
            if take_profit:
                # Take profit is opposite side of the entry
                tp_side = "sell" if direction == "buy" else "buy"
                
                try:
                    tp_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='take_profit_market',
                        side=tp_side,
                        amount=position_size,
                        params={
                            'stopPrice': take_profit,
                            'reduceOnly': True
                        }
                    )
                    logger.info(f"Take profit order placed: {tp_order['id']}")
                    results["take_profit_order"] = tp_order
                except Exception as e:
                    logger.error(f"Error placing take profit order: {e}")
                    results["take_profit_error"] = str(e)
            
            return {
                "success": True,
                "message": f"Successfully executed {direction} {entry_type} order for {symbol}",
                "orders": results
            }
            
        except Exception as e:
            logger.error(f"Error executing trading signal: {e}")
            return {"error": str(e), "success": False}
    
    async def _list_strategies(self) -> Dict[str, str]:
        """
        List available trading strategies.
        
        Returns:
            Dictionary of strategy names and brief descriptions
        """
        return {
            "breakout": "Identify breakouts from key support/resistance levels with volume confirmation",
            "trend_following": "Find high-probability entries in established trends with defined risk levels",
            "mean_reversion": "Look for overbought/oversold conditions for counter-trend moves",
            "comprehensive": "Complete multi-factor analysis combining technical, orderbook, and market data",
            "swing_trade": "Identify swing trading opportunities at key market structure points"
        }
    
    async def _get_strategy_description(self, strategy: str) -> Dict[str, str]:
        """
        Get a detailed description of a trading strategy.
        
        Args:
            strategy: Strategy type
            
        Returns:
            Dictionary with strategy details
        """
        descriptions = {
            "breakout": """
            Breakout Trading Strategy:
            
            This strategy focuses on identifying when price breaks through significant support or resistance levels
            with enough momentum to continue in the breakout direction. Key components include:
            
            - Identification of clear, tested support/resistance levels
            - Volume confirmation (increased volume on breakout)
            - Momentum indicator confirmation (RSI, MACD)
            - Assessment of market structure
            - Orderbook analysis to confirm lack of resistance after breakout
            
            Entry is typically placed just beyond the breakout level, with stops placed on the opposite side
            of the broken level. Take profit is set at the next significant level or using a risk-reward ratio.
            """,
            
            "trend_following": """
            Trend Following Strategy:
            
            This strategy is based on the principle that markets tend to continue in their current direction.
            Key components include:
            
            - Clear identification of trend direction using moving averages and price structure
            - Trend strength assessment (ADX or similar)
            - Waiting for pullbacks or consolidations within the trend
            - Entry on resumption of trend movement
            - Stop loss placement at recent swing points against the trend
            - Multiple take profit targets at extensions of the trend
            
            This approach avoids trying to pick tops and bottoms, instead focusing on high-probability
            continuations of established trends with defined risk parameters.
            """,
            
            "mean_reversion": """
            Mean Reversion Strategy:
            
            This strategy is based on the principle that prices tend to revert to their mean or average over time.
            Key components include:
            
            - Identification of significant deviation from moving averages or bands
            - Oscillator readings in overbought/oversold territory (RSI, Stochastic)
            - Confirmation of exhaustion or reversal signals (candlestick patterns)
            - Volume analysis to confirm potential reversal
            - Careful stop loss placement to protect against trend continuation
            
            Mean reversion trades have higher win rates but smaller profit targets, typically aiming
            for a move back to the mean (moving average) rather than extended moves.
            """,
            
            "comprehensive": """
            Comprehensive Analysis Strategy:
            
            This strategy integrates multiple analysis methods to find the highest-probability setups.
            Key components include:
            
            - Market context and overall sentiment analysis
            - Technical indicator confluence across multiple timeframes
            - Support/resistance identification and validation
            - Volume and liquidity analysis
            - Order book structure assessment
            - Risk-adjusted position sizing
            - Multiple scenario planning
            
            This approach generates fewer but higher-quality signals by requiring confluence
            of multiple factors before suggesting a trade. It incorporates both technical and
            market microstructure factors for a complete picture.
            """,
            
            "swing_trade": """
            Swing Trading Strategy:
            
            This strategy aims to capture "swings" in price action over periods of days to weeks.
            Key components include:
            
            - Market structure analysis (higher highs/higher lows or lower highs/lower lows)
            - Key level identification (support/resistance, previous swing points)
            - Multiple timeframe analysis for alignment
            - Entry at optimal risk/reward points (after pullbacks in trends)
            - Wider stop losses to accommodate normal market noise
            - Take profit at previous swing points or key levels
            
            Swing trading balances the more volatile day trading approaches with position trading,
            aiming for significant moves while managing risk through careful entry selection.
            """
        }
        
        if strategy not in descriptions:
            return {"error": f"Strategy '{strategy}' not found"}
            
        return {
            "strategy": strategy,
            "description": descriptions[strategy]
        }
    
    async def _get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance from Binance Futures.
        
        Returns:
            Dictionary with account balance information
        """
        try:
            # Initialize exchange if not already done
            if not self.exchange:
                success = await self.initialize_exchange()
                if not success:
                    return {"error": "Failed to initialize exchange connection"}
            
            # Get account balance
            balance = await self.exchange.fetch_balance()
            logger.info("Account balance fetched successfully")
            
            return {
                "success": True,
                "balance": balance
            }
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {"error": str(e), "success": False}
    
    async def _get_open_positions(self) -> Dict[str, Any]:
        """
        Get all open positions on Binance Futures.
        
        Returns:
            Dictionary with open positions information
        """
        try:
            # Initialize exchange if not already done
            if not self.exchange:
                success = await self.initialize_exchange()
                if not success:
                    return {"error": "Failed to initialize exchange connection"}
            
            # Get open positions
            positions = await self.exchange.fetch_positions()
            logger.info("Open positions fetched successfully")
            
            return {
                "success": True,
                "positions": positions
            }
            
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return {"error": str(e), "success": False}
    
    async def _get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all open orders on Binance Futures.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dictionary with open orders information
        """
        try:
            # Initialize exchange if not already done
            if not self.exchange:
                success = await self.initialize_exchange()
                if not success:
                    return {"error": "Failed to initialize exchange connection"}
            
            # Get open orders
            if symbol:
                orders = await self.exchange.fetch_open_orders(symbol=symbol)
            else:
                orders = await self.exchange.fetch_open_orders()
            logger.info("Open orders fetched successfully")
            
            return {
                "success": True,
                "orders": orders
            }
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return {"error": str(e), "success": False}
    
    async def _monitor_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Monitor a specific trading position.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with trade monitoring information
        """
        try:
            # Initialize exchange if not already done
            if not self.exchange:
                success = await self.initialize_exchange()
                if not success:
                    return {"error": "Failed to initialize exchange connection"}
            
            # Get trade information
            trade_info = await self.exchange.fetch_trade(symbol=symbol)
            logger.info(f"Trade information for {symbol} fetched successfully")
            
            return {
                "success": True,
                "trade_info": trade_info
            }
            
        except Exception as e:
            logger.error(f"Error monitoring trade: {e}")
            return {"error": str(e), "success": False}
    
    async def _close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Close an open position on Binance Futures.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with position closing information
        """
        try:
            # Initialize exchange if not already done
            if not self.exchange:
                success = await self.initialize_exchange()
                if not success:
                    return {"error": "Failed to initialize exchange connection"}
            
            # Close position
            close_result = await self.exchange.close_position(symbol=symbol)
            logger.info(f"Position for {symbol} closed successfully")
            
            return {
                "success": True,
                "close_result": close_result
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"error": str(e), "success": False}
    
    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user request by running the trading strategy agent.
        
        Args:
            user_input: User's input string
            
        Returns:
            Dictionary with the response
        """
        try:
            logger.info(f"Processing request: '{user_input}'")
            
            # Run the agent
            result = await Runner.run(self.agent, user_input)
            
            # Extract the response
            response = {
                "response": result.final_output,
                "success": True
            }
            
            logger.info(f"Request processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "response": f"An error occurred while processing your request: {str(e)}",
                "success": False
            }
    
    async def close(self):
        """Close the exchange connection"""
        if self.exchange:
            await self.exchange.close()
            logger.info("Exchange connection closed")

async def setup_trading_strategy_agent(market_agent, execution_agent=None):
    """
    Set up the trading strategy agent with connections to market analysis and execution agents.
    
    Args:
        market_agent: Main market analysis orchestration agent
        execution_agent: Optional execution agent
        
    Returns:
        Configured trading strategy agent
    """
    try:
        logger.info("Setting up trading strategy agent...")
        
        # Create the trading strategy agent
        strategy_agent = TradingStrategyAgent(market_agent, execution_agent)
        
        # Initialize the exchange connection
        await strategy_agent.initialize_exchange()
        
        logger.info("Trading strategy agent set up successfully!")
        
        return strategy_agent
        
    except Exception as e:
        logger.error(f"Error setting up trading strategy agent: {e}")
        raise

async def main():
    """Main function to run the trading strategy agent independently."""
    try:
        # Import the market analysis system
        from enhanced_main import setup_crypto_agents
        
        # Setup market analysis agent
        market_agent = await setup_crypto_agents()
        
        # Setup trading strategy agent
        strategy_agent = await setup_trading_strategy_agent(market_agent)
        
        print("Trading Strategy Agent - Type 'exit' to quit")
        print("----------------------------------------------------------")
        print("Example commands:")
        print("- generate signal BTC/USDT breakout 4h")
        print("- list strategies")
        print("- describe strategy trend_following")
        print("----------------------------------------------------------")
        
        while True:
            user_input = input("\nEnter your request (or 'exit' to quit): ")
            
            if user_input.lower() == 'exit':
                print("Exiting the trading strategy agent. Goodbye!")
                # Cleanup
                await strategy_agent.close()
                break
                
            try:
                print("\nProcessing your request...")
                
                # Process the request
                result = await strategy_agent.process_request(user_input)
                
                # Display the response
                print("\n🤖 Trading Strategy Agent Response:")
                print("-----------------")
                print(result["response"])
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"\nAn error occurred while processing your request: {e}")
                
    except Exception as e:
        logger.error(f"Main function error: {e}")
        print(f"Error: {e}")
    finally:
        # Ensure the exchange connection is closed properly
        if 'strategy_agent' in locals() and hasattr(strategy_agent, 'close'):
            await strategy_agent.close()

if __name__ == "__main__":
    asyncio.run(main())
