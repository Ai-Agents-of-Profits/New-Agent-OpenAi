"""
Fixed Trading Strategy Agent - Compatible with Python 3.13
Works around the "Cannot instantiate typing.Union" error while maintaining the original architecture.
"""
import os
import asyncio
import logging
import json
import platform
from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import ccxt

# Configure Windows-specific event loop policy for CCXT compatibility
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from agents import Agent, Runner, handoff, Tool, function_tool, RunContextWrapper
from src.exchange.connector import ExchangeConnector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trading Signal model - avoid using Optional types directly
class TradingSignal(BaseModel):
    """Model for a trading signal with execution parameters"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    direction: str = Field(..., description="Trade direction ('buy' or 'sell')")
    entry_type: str = Field("market", description="Entry type ('market' or 'limit')")
    entry_price: float = Field(None, description="Suggested entry price (None for market orders)")
    stop_loss: float = Field(None, description="Stop loss price level")
    take_profit: float = Field(None, description="Take profit price level") 
    position_size: float = Field(..., description="Trade quantity in base currency")
    leverage: int = Field(1, description="Leverage to use (1-125)")
    confidence: float = Field(..., description="Signal confidence score (0-1)")
    timeframe: str = Field(..., description="Analysis timeframe used")
    reasoning: str = Field(..., description="Trading logic and reasoning")
    expiration: datetime = Field(None, description="Signal expiration time")

# Parameter models for function_tool decorator
class GenerateSignalParams(BaseModel):
    """Parameters for generating a trading signal"""
    strategy: str = Field(..., description="Trading strategy type")
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '15m', '1h', '4h', '1d')")

class ExecuteSignalParams(BaseModel):
    """Parameters for executing a trading signal"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    direction: str = Field(..., description="Trade direction") 
    entry_type: str = Field("market", description="Entry type (market or limit)")
    position_size: float = Field(..., description="Position size in base currency")
    entry_price: float = Field(None, description="Entry price (for limit orders)")
    stop_loss: float = Field(None, description="Stop loss price level")
    take_profit: float = Field(None, description="Take profit price level")
    leverage: int = Field(1, description="Leverage to use (1-125)")

class GetOrdersParams(BaseModel):
    """Parameters for getting open orders"""
    symbol: str = Field(None, description="Trading pair symbol (e.g., 'BTC/USDT')")

class MonitorTradeParams(BaseModel):
    """Parameters for monitoring a trade"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")

class ClosePositionParams(BaseModel):
    """Parameters for closing a position"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")

class StrategyDescriptionParams(BaseModel):
    """Parameters for getting strategy description"""
    strategy: str = Field(..., description="Trading strategy type")

class TrailingStopParams(BaseModel):
    """Parameters for setting a trailing stop"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    activation_price: float = Field(None, description="Price at which the trailing stop is activated")
    callback_rate: float = Field(..., description="Callback rate in percentage (e.g., 1.0 for 1%)")
    position_size: float = Field(..., description="Position size to close with trailing stop")
    side: str = Field(..., description="Side of trailing stop ('buy' or 'sell')")

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
    Fixed Trading Strategy Agent for crypto trading.
    Uses the function_tool decorator approach to avoid typing.Union instantiation issues.
    """
    
    def __init__(self, market_agent=None, execution_agent=None):
        """
        Initialize the Trading Strategy Agent with optional market and execution agents.
        """
        self.market_agent = market_agent
        self.execution_agent = execution_agent
        self.exchange = None
        self.exchange_id = "binanceusdm"  # Default exchange
        self.agent = None
        self.trading_prompts = TRADING_PROMPTS
        
        # Initialize CCXT exchange - using synchronous version to avoid event loop issues
        self._init_exchange()
        
    def _init_exchange(self):
        """Initialize the exchange connection"""
        try:
            binance_api_key = os.getenv('BINANCE_API_KEY')
            binance_api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not binance_api_key or not binance_api_secret:
                logger.warning("BINANCE_API_KEY or BINANCE_API_SECRET not found in environment variables.")
                return None
            
            # Use synchronous CCXT instead of async version
            self.exchange = ccxt.binanceusdm({
                'apiKey': binance_api_key,
                'secret': binance_api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
            
            # Test connection
            self.exchange.load_markets()
            logger.info(f"Successfully connected to {self.exchange_id}")
            
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            self.exchange = None
        
    def create_agent(self):
        """Create the Trading Strategy Agent using function_tool decorators"""
        
        # Create agent with tools that use @function_tool decorator
        self.agent = Agent(
            name="Trading Strategy Agent",
            instructions="""
            You are the Trading Strategy Agent, specialized in generating and executing cryptocurrency trading signals.
            
            Your capabilities include:
            1. Generating trading signals based on various strategies
            2. Executing trades on Binance Futures
            3. Monitoring open positions and orders
            4. Closing positions when needed
            
            When generating signals, analyze the market thoroughly using technical analysis, order book data, and 
            market conditions. Always provide clear reasoning behind your recommendations.
            
            When executing trades, ensure proper risk management with appropriate stop-loss and take-profit levels.
            
            Provide clear and concise responses about the actions you've taken and always prioritize 
            capital preservation and risk management.
            """,
            tools=[
                Tool(
                    name="generate_signal",
                    description="Generate a trading signal based on the specified strategy and market conditions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "description": "Trading strategy type"
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Trading pair symbol (e.g., 'BTC/USDT')"
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Timeframe for analysis (e.g., '15m', '1h', '4h', '1d')"
                            }
                        },
                        "required": ["strategy", "symbol", "timeframe"]
                    },
                    function=self._generate_signal_wrapper
                ),
                Tool(
                    name="execute_signal",
                    description="Execute a trading signal by placing orders on the exchange",
                    parameters={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading pair symbol (e.g., 'BTC/USDT')"
                            },
                            "direction": {
                                "type": "string",
                                "description": "Trade direction"
                            },
                            "entry_type": {
                                "type": "string",
                                "description": "Entry type (market or limit)",
                                "default": "market"
                            },
                            "position_size": {
                                "type": "number",
                                "description": "Position size in base currency"
                            },
                            "entry_price": {
                                "type": "number",
                                "description": "Entry price (for limit orders)"
                            },
                            "stop_loss": {
                                "type": "number",
                                "description": "Stop loss price level"
                            },
                            "take_profit": {
                                "type": "number",
                                "description": "Take profit price level"
                            },
                            "leverage": {
                                "type": "integer",
                                "description": "Leverage to use (1-125)",
                                "default": 1
                            }
                        },
                        "required": ["symbol", "direction", "position_size"]
                    },
                    function=self._execute_signal_wrapper
                ),
                Tool(
                    name="get_open_orders",
                    description="Get open orders for a symbol or all symbols",
                    parameters={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading pair symbol (e.g., 'BTC/USDT')"
                            }
                        },
                        "required": []
                    },
                    function=self._get_open_orders_wrapper
                ),
                Tool(
                    name="monitor_trade",
                    description="Monitor an open trade for a symbol",
                    parameters={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading pair symbol (e.g., 'BTC/USDT')"
                            }
                        },
                        "required": ["symbol"]
                    },
                    function=self._monitor_trade_wrapper
                ),
                Tool(
                    name="close_position",
                    description="Close an open position for a symbol",
                    parameters={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading pair symbol (e.g., 'BTC/USDT')"
                            }
                        },
                        "required": ["symbol"]
                    },
                    function=self._close_position_wrapper
                ),
                Tool(
                    name="get_strategy_description",
                    description="Get a description of a trading strategy",
                    parameters={
                        "type": "object",
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "description": "Trading strategy type"
                            }
                        },
                        "required": ["strategy"]
                    },
                    function=self._get_strategy_description_wrapper
                ),
                Tool(
                    name="set_trailing_stop",
                    description="Set a trailing stop for an open position",
                    parameters={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading pair symbol (e.g., 'BTC/USDT')"
                            },
                            "activation_price": {
                                "type": "number",
                                "description": "Price at which the trailing stop is activated"
                            },
                            "callback_rate": {
                                "type": "number",
                                "description": "Callback rate in percentage (e.g., 1.0 for 1%)"
                            },
                            "position_size": {
                                "type": "number",
                                "description": "Position size to close with trailing stop"
                            },
                            "side": {
                                "type": "string",
                                "description": "Side of trailing stop ('buy' or 'sell')"
                            }
                        },
                        "required": ["symbol", "callback_rate", "position_size", "side"]
                    },
                    function=self._set_trailing_stop_wrapper
                )
            ]
        )
        
        return self.agent
    
    async def _generate_signal_wrapper(self, strategy: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Wrapper for the generate_signal method to work with Tool"""
        params = GenerateSignalParams(strategy=strategy, symbol=symbol, timeframe=timeframe)
        return await self._generate_signal_impl(params)
    
    async def _execute_signal_wrapper(self, symbol: str, direction: str, position_size: float, 
                                     entry_type: str = "market", entry_price: float = None, 
                                     stop_loss: float = None, take_profit: float = None, 
                                     leverage: int = 1) -> Dict[str, Any]:
        """Wrapper for execute_signal method"""
        params = ExecuteSignalParams(
            symbol=symbol,
            direction=direction,
            entry_type=entry_type,
            position_size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage
        )
        return await self._execute_signal_impl(params)
    
    async def _get_open_orders_wrapper(self, symbol: str = None) -> Dict[str, Any]:
        """Wrapper for get_open_orders method"""
        params = GetOrdersParams(symbol=symbol)
        return await self._get_open_orders_impl(params)
    
    async def _monitor_trade_wrapper(self, symbol: str) -> Dict[str, Any]:
        """Wrapper for monitor_trade method"""
        params = MonitorTradeParams(symbol=symbol)
        return await self._monitor_trade_impl(params)
    
    async def _close_position_wrapper(self, symbol: str) -> Dict[str, Any]:
        """Wrapper for close_position method"""
        params = ClosePositionParams(symbol=symbol)
        return await self._close_position_impl(params)
    
    async def _get_strategy_description_wrapper(self, strategy: str) -> Dict[str, Any]:
        """Wrapper for get_strategy_description method"""
        params = StrategyDescriptionParams(strategy=strategy)
        return await self._get_strategy_description_impl(params)
    
    async def _set_trailing_stop_wrapper(self, symbol: str, callback_rate: float, position_size: float,
                                        side: str, activation_price: float = None) -> Dict[str, Any]:
        """Wrapper for set_trailing_stop method"""
        params = TrailingStopParams(
            symbol=symbol,
            activation_price=activation_price,
            callback_rate=callback_rate,
            position_size=position_size,
            side=side
        )
        return await self._set_trailing_stop_impl(params)
    
    async def _generate_signal_impl(self, params: GenerateSignalParams) -> Dict[str, Any]:
        """
        Generate a trading signal based on the specified strategy and market conditions.
        
        Args:
            params: Parameters for generating a trading signal
        
        Returns:
            A dictionary containing the trading signal details
        """
        try:
            strategy = params.strategy
            symbol = params.symbol
            timeframe = params.timeframe
            
            # Check if we have a prompt for this strategy
            if strategy not in self.trading_prompts:
                return {
                    "error": f"Strategy {strategy} not found. Available strategies: {', '.join(self.trading_prompts.keys())}"
                }
            
            prompt = self.trading_prompts[strategy].format(symbol=symbol, timeframe=timeframe)
            
            if not self.market_agent:
                return {
                    "error": "No market analysis agent available to generate signals."
                }
                
            # Direct implementation - bypassing problematic handoff/Runner mechanisms
            try:
                # Using the market agent's as_tool functionality directly
                from agents import RunContext
                
                # Create a simple run context for the agent call
                context = RunContext(user_input=prompt)
                
                # Call the agent directly
                response = await self.market_agent.run_async(context)
                final_output = response.output
                
                # Parse the result to extract a structured trading signal
                signal = self._parse_signal_from_text(final_output, symbol, timeframe)
            except Exception as e:
                logger.error(f"Error in market analysis: {e}")
                return {
                    "success": False,
                    "error": f"Market analysis failed: {str(e)}"
                }
            
            if signal:
                return {
                    "success": True,
                    "signal": signal.dict(),
                    "message": "Trading signal generated successfully."
                }
            else:
                return {
                    "success": False,
                    "message": "No valid trading signal could be extracted from the analysis.",
                    "raw_analysis": final_output
                }
                
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_signal_from_text(self, text: str, symbol: str, timeframe: str) -> TradingSignal:
        """
        Parse a trading signal from the text response of the market agent.
        """
        try:
            # Try to extract key elements
            direction = None
            if "buy" in text.lower() or "long" in text.lower():
                direction = "buy"
            elif "sell" in text.lower() or "short" in text.lower():
                direction = "sell"
            
            if not direction:
                return None
            
            # Extract entry, stop loss, and take profit values using simple pattern matching
            import re
            
            # Entry price
            entry_type = "market"  # Default
            entry_price = None
            entry_match = re.search(r"entry\s*(?:price|level)?\s*(?::|at|\:|\-|=)\s*(\d+\.?\d*)", text, re.IGNORECASE)
            if entry_match:
                entry_price = float(entry_match.group(1))
                entry_type = "limit"
            
            # Stop loss
            stop_loss = None
            sl_match = re.search(r"stop[\-\s]loss\s*(?::|at|\:|\-|=)\s*(\d+\.?\d*)", text, re.IGNORECASE)
            if sl_match:
                stop_loss = float(sl_match.group(1))
            
            # Take profit
            take_profit = None
            tp_match = re.search(r"take[\-\s]profit\s*(?::|at|\:|\-|=)\s*(\d+\.?\d*)", text, re.IGNORECASE)
            if tp_match:
                take_profit = float(tp_match.group(1))
            
            # Position size (default to a small value if not specified)
            position_size = 0.01
            size_match = re.search(r"position\s*(?:size|amount)?\s*(?::|at|\:|\-|=)\s*(\d+\.?\d*)", text, re.IGNORECASE)
            if size_match:
                position_size = float(size_match.group(1))
            
            # Leverage (default to 1x if not specified)
            leverage = 1
            leverage_match = re.search(r"leverage\s*(?::|at|\:|\-|=)\s*(\d+)x?", text, re.IGNORECASE)
            if leverage_match:
                leverage = min(int(leverage_match.group(1)), 125)  # Cap at 125x (Binance limit)
            
            # Confidence
            confidence = 0.7  # Default medium-high confidence
            conf_match = re.search(r"confidence\s*(?::|at|\:|\-|=)\s*(\d+\.?\d*)", text, re.IGNORECASE)
            if conf_match:
                confidence = min(float(conf_match.group(1)), 1.0)
                if confidence > 1.0:  # If provided as percentage
                    confidence = confidence / 100.0
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                direction=direction,
                entry_type=entry_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                leverage=leverage,
                confidence=confidence,
                timeframe=timeframe,
                reasoning=text,
                expiration=None  # No expiration set
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error parsing signal from text: {e}")
            return None
    
    async def _execute_signal_impl(self, params: ExecuteSignalParams) -> Dict[str, Any]:
        """
        Execute a trading signal by placing orders on the exchange.
        
        Args:
            params: Parameters for executing a trading signal
            
        Returns:
            A dictionary containing the execution results
        """
        try:
            symbol = params.symbol
            direction = params.direction
            entry_type = params.entry_type
            position_size = params.position_size
            entry_price = params.entry_price
            stop_loss = params.stop_loss
            take_profit = params.take_profit
            leverage = params.leverage
            
            if not self.exchange:
                return {
                    "error": "Exchange connection not available."
                }
            
            logger.info(f"Executing {direction} {entry_type} order for {symbol} with size {position_size}...")
            
            # Set leverage
            try:
                self.exchange.set_leverage(leverage, symbol)
                logger.info(f"Leverage set to {leverage}x for {symbol}")
            except Exception as e:
                logger.warning(f"Error setting leverage: {e}")
            
            # Place main order
            order_result = None
            if entry_type == "market":
                order_result = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=direction,
                    amount=position_size
                )
            elif entry_type == "limit" and entry_price:
                order_result = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=direction,
                    amount=position_size,
                    price=entry_price
                )
            
            if not order_result:
                return {
                    "error": "Failed to place order. Check parameters and exchange connection."
                }
            
            # Place stop loss if provided
            sl_order = None
            if stop_loss:
                sl_side = "buy" if direction == "sell" else "sell"
                try:
                    sl_order = self.exchange.create_order(
                        symbol=symbol,
                        type='stop_market',
                        side=sl_side,
                        amount=position_size,
                        params={
                            'stopPrice': stop_loss,
                            'reduceOnly': True
                        }
                    )
                except Exception as e:
                    logger.error(f"Error placing stop loss: {e}")
            
            # Place take profit if provided
            tp_order = None
            if take_profit:
                tp_side = "buy" if direction == "sell" else "sell"
                try:
                    tp_order = self.exchange.create_order(
                        symbol=symbol,
                        type='take_profit_market',
                        side=tp_side,
                        amount=position_size,
                        params={
                            'stopPrice': take_profit,
                            'reduceOnly': True
                        }
                    )
                except Exception as e:
                    logger.error(f"Error placing take profit: {e}")
            
            return {
                "success": True,
                "message": f"{direction.capitalize()} {entry_type} order executed successfully.",
                "main_order": order_result,
                "stop_loss_order": sl_order,
                "take_profit_order": tp_order
            }
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_open_orders_impl(self, params: GetOrdersParams) -> Dict[str, Any]:
        """
        Get open orders for a symbol or all symbols.
        
        Args:
            params: Parameters for getting open orders
            
        Returns:
            A dictionary containing the open orders
        """
        try:
            symbol = params.symbol
            
            if not self.exchange:
                return {
                    "error": "Exchange connection not available."
                }
            
            if symbol:
                logger.info(f"Fetching open orders for {symbol}...")
                orders = self.exchange.fetch_open_orders(symbol)
            else:
                logger.info("Fetching all open orders...")
                orders = self.exchange.fetch_open_orders()
            
            # Format orders for readability
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    "id": order.get("id"),
                    "symbol": order.get("symbol"),
                    "type": order.get("type"),
                    "side": order.get("side"),
                    "price": order.get("price"),
                    "amount": order.get("amount"),
                    "status": order.get("status"),
                    "datetime": order.get("datetime")
                })
            
            return {
                "success": True,
                "open_orders": formatted_orders,
                "count": len(formatted_orders)
            }
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _monitor_trade_impl(self, params: MonitorTradeParams) -> Dict[str, Any]:
        """
        Monitor an open trade for a symbol.
        
        Args:
            params: Parameters for monitoring a trade
            
        Returns:
            A dictionary containing the trade status
        """
        try:
            symbol = params.symbol
            
            if not self.exchange:
                return {
                    "error": "Exchange connection not available."
                }
            
            logger.info(f"Monitoring position for {symbol}...")
            
            # Get position
            positions = self.exchange.fetch_positions([symbol])
            position = next((p for p in positions if p.get("symbol") == symbol), None)
            
            # Get open orders
            orders = self.exchange.fetch_open_orders(symbol)
            
            # Get current market price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker.get("last")
            
            # Calculate profit/loss if position exists
            pnl = None
            pnl_percentage = None
            if position and float(position.get("contracts", 0)) > 0:
                entry_price = float(position.get("entryPrice", 0))
                side = position.get("side")
                size = float(position.get("contracts", 0))
                notional = size * entry_price
                
                if side == "long":
                    pnl = (current_price - entry_price) * size
                    pnl_percentage = (current_price - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - current_price) * size
                    pnl_percentage = (entry_price - current_price) / entry_price * 100
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": current_price,
                "position": position,
                "open_orders": orders,
                "pnl": pnl,
                "pnl_percentage": pnl_percentage
            }
            
        except Exception as e:
            logger.error(f"Error monitoring trade: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _close_position_impl(self, params: ClosePositionParams) -> Dict[str, Any]:
        """
        Close an open position for a symbol.
        
        Args:
            params: Parameters for closing a position
            
        Returns:
            A dictionary containing the closure results
        """
        try:
            symbol = params.symbol
            
            if not self.exchange:
                return {
                    "error": "Exchange connection not available."
                }
            
            logger.info(f"Closing position for {symbol}...")
            
            # Get position
            positions = self.exchange.fetch_positions([symbol])
            position = next((p for p in positions if p.get("symbol") == symbol), None)
            
            if not position or float(position.get("contracts", 0)) == 0:
                return {
                    "success": False,
                    "message": f"No open position found for {symbol}."
                }
            
            # Get position details
            side = position.get("side")
            contracts = float(position.get("contracts", 0))
            close_side = "sell" if side == "long" else "buy"
            
            # Close position with market order
            order_result = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=contracts,
                params={
                    'reduceOnly': True
                }
            )
            
            # Cancel all related orders
            try:
                self.exchange.cancel_all_orders(symbol)
                logger.info(f"Cancelled all orders for {symbol}")
            except Exception as e:
                logger.warning(f"Error cancelling orders: {e}")
            
            return {
                "success": True,
                "message": f"Position for {symbol} closed successfully.",
                "order": order_result
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_strategy_description_impl(self, params: StrategyDescriptionParams) -> Dict[str, Any]:
        """
        Get a description of a trading strategy.
        
        Args:
            params: Parameters for getting a strategy description
            
        Returns:
            A dictionary containing the strategy description
        """
        try:
            strategy = params.strategy
            
            if strategy not in self.trading_prompts:
                return {
                    "success": False,
                    "message": f"Strategy {strategy} not found. Available strategies: {', '.join(self.trading_prompts.keys())}"
                }
            
            # Clean up the trading prompt for display
            prompt = self.trading_prompts[strategy]
            description = "\n".join(line.strip() for line in prompt.split("\n")).strip()
            
            strategies_info = {
                "breakout": "Identifies potential breakouts from key support/resistance levels and enters trades with momentum.",
                "trend_following": "Follows established trends, entering on pullbacks or trend continuation signals.",
                "mean_reversion": "Identifies overbought/oversold conditions and looks for price to revert to the mean.",
                "comprehensive": "Combines multiple strategies for a thorough market analysis and high-probability trades.",
                "swing_trade": "Identifies medium-term swing trading opportunities over multiple days to weeks."
            }
            
            strategy_info = strategies_info.get(strategy, "No additional information available.")
            
            return {
                "success": True,
                "strategy": strategy,
                "description": strategy_info,
                "prompt_template": description
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy description: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _set_trailing_stop_impl(self, params: TrailingStopParams) -> Dict[str, Any]:
        """
        Set a trailing stop for an open position.
        
        Args:
            params: Parameters for setting a trailing stop
            
        Returns:
            A dictionary containing the trailing stop order details
        """
        try:
            symbol = params.symbol
            activation_price = params.activation_price
            callback_rate = params.callback_rate
            position_size = params.position_size
            side = params.side.lower()
            
            if not self.exchange:
                return {
                    "success": False,
                    "error": "Exchange connection not available."
                }
            
            logger.info(f"Setting trailing stop for {symbol} with {callback_rate}% callback...")
            
            # Check if side is valid
            if side not in ["buy", "sell"]:
                return {
                    "success": False,
                    "error": f"Invalid side: {side}. Must be 'buy' or 'sell'."
                }
            
            # Prepare parameters for trailing stop
            order_params = {
                'callbackRate': callback_rate,
                'reduceOnly': True,
            }
            
            # Add activation price if provided
            if activation_price:
                order_params['activationPrice'] = activation_price
            
            # Create trailing stop order
            order_result = self.exchange.create_order(
                symbol=symbol,
                type='TRAILING_STOP_MARKET',
                side=side,
                amount=position_size,
                params=order_params
            )
            
            return {
                "success": True,
                "message": f"Trailing stop order placed successfully with {callback_rate}% callback rate.",
                "order": order_result
            }
            
        except Exception as e:
            logger.error(f"Error setting trailing stop: {e}")
            return {
                "success": False,
                "error": str(e)
            }

async def setup_trading_strategy_agent(market_agent, execution_agent=None):
    """
    Set up the Trading Strategy Agent.
    
    Args:
        market_agent: Market analysis agent to use for generating signals
        execution_agent: Optional execution agent for placing trades
        
    Returns:
        The configured Trading Strategy Agent
    """
    try:
        logger.info("Setting up Trading Strategy Agent...")
        
        strategy_agent = TradingStrategyAgent(market_agent, execution_agent)
        agent = strategy_agent.create_agent()
        
        logger.info("Trading Strategy Agent has been set up successfully!")
        
        return agent
        
    except Exception as e:
        logger.error(f"Error setting up Trading Strategy Agent: {e}")
        raise

async def main():
    """
    Main function to run the Trading Strategy Agent as a standalone system.
    """
    try:
        from enhanced_main import setup_crypto_agents
        
        # Set up market agent
        market_agent = await setup_crypto_agents()
        
        # Set up trading strategy agent
        trading_agent = TradingStrategyAgent(market_agent)
        
        # Direct method access mode - avoiding the Agent wrapper which causes the Union issue
        print("Trading Strategy Agent - Type 'exit' to quit")
        print("---------------------------------------------")
        
        while True:
            user_input = input("\nEnter your command (or 'exit' to quit): ")
            
            if user_input.lower() == 'exit':
                print("Exiting the agent system. Goodbye!")
                break
                
            try:
                print("\nProcessing your request...")
                
                # Parse the command manually instead of using the Runner
                parts = user_input.lower().split()
                
                if not parts:
                    print("Please enter a valid command.")
                    continue
                
                command = parts[0]
                
                if command == "generate" and len(parts) >= 4:
                    # Format: generate <strategy> <symbol> <timeframe>
                    strategy = parts[1]
                    symbol = parts[2].upper() + "/USDT"  # Add USDT pair automatically
                    timeframe = parts[3]
                    
                    result = await trading_agent._generate_signal_wrapper(
                        strategy=strategy,
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    
                elif command == "execute" and len(parts) >= 4:
                    # Format: execute <symbol> <direction> <size> [leverage]
                    symbol = parts[1].upper() + "/USDT"  # Add USDT pair automatically
                    direction = parts[2].lower()
                    size = float(parts[3])
                    leverage = int(parts[4]) if len(parts) > 4 else 1
                    
                    result = await trading_agent._execute_signal_wrapper(
                        symbol=symbol,
                        direction=direction,
                        position_size=size,
                        leverage=leverage
                    )
                    
                elif command == "monitor" and len(parts) >= 2:
                    # Format: monitor <symbol>
                    symbol = parts[1].upper() + "/USDT"  # Add USDT pair automatically
                    
                    result = await trading_agent._monitor_trade_wrapper(symbol=symbol)
                    
                elif command == "close" and len(parts) >= 2:
                    # Format: close <symbol>
                    symbol = parts[1].upper() + "/USDT"  # Add USDT pair automatically
                    
                    result = await trading_agent._close_position_wrapper(symbol=symbol)
                    
                elif command == "orders":
                    # Format: orders [symbol]
                    symbol = parts[1].upper() + "/USDT" if len(parts) > 1 else None
                    
                    result = await trading_agent._get_open_orders_wrapper(symbol=symbol)
                    
                elif command == "balance":
                    # Get account balance
                    try:
                        if not trading_agent.exchange:
                            result = {
                                "success": False,
                                "message": "Exchange connection not available."
                            }
                        else:
                            balance = trading_agent.exchange.fetch_balance()
                            
                            # Format the balance for display
                            total_balance = balance.get('total', {})
                            free_balance = balance.get('free', {})
                            used_balance = balance.get('used', {})
                            
                            # Filter out zero balances
                            assets = []
                            for currency, amount in total_balance.items():
                                if amount > 0:
                                    assets.append({
                                        "currency": currency,
                                        "total": amount,
                                        "free": free_balance.get(currency, 0),
                                        "used": used_balance.get(currency, 0)
                                    })
                            
                            result = {
                                "success": True,
                                "account_type": "futures",
                                "assets": assets
                            }
                    except Exception as e:
                        result = {
                            "success": False,
                            "error": f"Error fetching balance: {str(e)}"
                        }
                    
                elif command == "strategy" and len(parts) >= 2:
                    # Format: strategy <strategy_name>
                    strategy = parts[1]
                    
                    result = await trading_agent._get_strategy_description_wrapper(strategy=strategy)
                    
                elif command == "trailing" and len(parts) >= 5:
                    # Format: trailing <symbol> <side> <size> <callback_rate> [activation_price]
                    symbol = parts[1].upper() + "/USDT"  # Add USDT pair automatically
                    side = parts[2].lower()
                    size = float(parts[3])
                    callback_rate = float(parts[4])
                    activation_price = float(parts[5]) if len(parts) > 5 else None
                    
                    result = await trading_agent._set_trailing_stop_wrapper(
                        symbol=symbol,
                        side=side,
                        position_size=size,
                        callback_rate=callback_rate,
                        activation_price=activation_price
                    )
                    
                elif command == "help":
                    # Show available commands
                    result = {
                        "success": True,
                        "message": """
Available commands:
- generate <strategy> <symbol> <timeframe>: Generate a trading signal
  Example: generate breakout btc 1h
  
- execute <symbol> <direction> <size> [leverage]: Execute a trade
  Example: execute btc buy 0.01 5
  
- monitor <symbol>: Monitor an open position
  Example: monitor btc
  
- close <symbol>: Close an open position
  Example: close btc
  
- orders [symbol]: Get open orders (optional: for specific symbol)
  Example: orders btc
  
- balance: Check account balance
  
- trailing <symbol> <side> <size> <callback_rate> [activation_price]: Set a trailing stop
  Example: trailing btc sell 0.01 1.0 50000
  
- strategy <strategy_name>: Get description of a trading strategy
  Example: strategy breakout
  
- help: Show this help message

Available strategies: breakout, trend_following, mean_reversion, comprehensive, swing_trade
"""
                    }
                    
                else:
                    result = {
                        "success": False,
                        "message": "Unknown command. Type 'help' to see available commands."
                    }
                
                # Display the response
                print("\n🤖 Agent Response:")
                print("-----------------")
                
                if isinstance(result, dict):
                    if result.get("success", False):
                        print("✅ Success!")
                        
                        # Format the result based on its type
                        if "signal" in result:
                            print("\nTrading Signal:")
                            signal = result["signal"]
                            print(f"Symbol: {signal.get('symbol')}")
                            print(f"Direction: {signal.get('direction')}")
                            print(f"Entry Type: {signal.get('entry_type')}")
                            print(f"Position Size: {signal.get('position_size')}")
                            print(f"Leverage: {signal.get('leverage', 1)}x")
                            
                            if signal.get('entry_price'):
                                print(f"Entry Price: {signal.get('entry_price')}")
                            if signal.get('stop_loss'):
                                print(f"Stop Loss: {signal.get('stop_loss')}")
                            if signal.get('take_profit'):
                                print(f"Take Profit: {signal.get('take_profit')}")
                            
                            print(f"\nConfidence: {signal.get('confidence', 0) * 100:.2f}%")
                            print(f"Timeframe: {signal.get('timeframe')}")
                            
                            if signal.get('reasoning'):
                                print(f"\nReasoning:\n{signal.get('reasoning')}")
                                
                        elif "open_orders" in result:
                            print("\nOpen Orders:")
                            orders = result["open_orders"]
                            if orders:
                                for i, order in enumerate(orders, 1):
                                    print(f"\nOrder {i}:")
                                    print(f"Symbol: {order.get('symbol')}")
                                    print(f"Type: {order.get('type')} {order.get('side')}")
                                    print(f"Price: {order.get('price')}")
                                    print(f"Amount: {order.get('amount')}")
                                    print(f"Status: {order.get('status')}")
                                    print(f"Created: {order.get('datetime')}")
                            else:
                                print("No open orders found.")
                                
                        elif "assets" in result:
                            print("\nAccount Balance:")
                            assets = result["assets"]
                            if assets:
                                print(f"Account Type: {result.get('account_type', 'futures')}")
                                print("\n{:<8} {:<12} {:<12} {:<12}".format("ASSET", "TOTAL", "FREE", "USED"))
                                print("-" * 50)
                                for asset in assets:
                                    print("{:<8} {:<12.8f} {:<12.8f} {:<12.8f}".format(
                                        asset["currency"],
                                        asset["total"],
                                        asset["free"],
                                        asset["used"]
                                    ))
                            else:
                                print("No assets found or zero balances.")
                                
                        elif "position" in result:
                            print("\nPosition Status:")
                            position = result["position"]
                            print(f"Symbol: {result.get('symbol')}")
                            print(f"Current Price: {result.get('current_price')}")
                            
                            if position and float(position.get("contracts", 0)) > 0:
                                print(f"Side: {position.get('side')}")
                                print(f"Size: {position.get('contracts')} contracts")
                                print(f"Entry Price: {position.get('entryPrice')}")
                                print(f"Liquidation Price: {position.get('liquidationPrice')}")
                                
                                if result.get('pnl') is not None:
                                    print(f"Unrealized PnL: {result.get('pnl'):.8f}")
                                    print(f"PnL %: {result.get('pnl_percentage'):.2f}%")
                            else:
                                print("No open position found.")
                                
                            if result.get('open_orders'):
                                print("\nRelated Orders:")
                                for i, order in enumerate(result['open_orders'], 1):
                                    print(f"{i}. {order.get('type')} {order.get('side')} @ {order.get('price')}")
                                    
                        elif "strategy" in result:
                            print(f"\nStrategy: {result.get('strategy')}")
                            print(f"\nDescription: {result.get('description')}")
                            
                        elif "message" in result:
                            print(result["message"])
                            
                        # Print any additional information
                        for key, value in result.items():
                            if key not in ["success", "message", "signal", "open_orders", "position", 
                                          "strategy", "description", "pnl", "pnl_percentage"]:
                                print(f"{key}: {value}")
                                
                    else:
                        print("❌ Error:")
                        if "error" in result:
                            print(result["error"])
                        elif "message" in result:
                            print(result["message"])
                        else:
                            print("Unknown error occurred.")
                else:
                    print(str(result))
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"\nAn error occurred while processing your request: {e}")
                
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"\nFailed to initialize the agent system: {e}")

if __name__ == "__main__":
    asyncio.run(main())
