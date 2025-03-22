"""
Execution agent for placing and managing trades on cryptocurrency exchanges.
"""
import logging
from typing import Dict, Any, Optional
import ccxt
import ccxt.async_support as ccxt_async
from pydantic import BaseModel, Field

from agents import Agent, Tool
from .base_agent import BaseMarketAgent
from ..exchange.connector import ExchangeConnector

logger = logging.getLogger(__name__)


class ExecuteTradeRequest(BaseModel):
    """Model for trade execution requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    side: str = Field(..., description="Trade direction ('buy' or 'sell')")
    amount: float = Field(..., description="Position size to trade")
    order_type: str = Field(..., description="Order type ('market', 'limit', or 'stop_limit')")
    price: Optional[float] = Field(None, description="Price for limit orders")
    stop_price: Optional[float] = Field(None, description="Stop trigger price for stop-limit orders")
    leverage: Optional[int] = Field(None, description="Leverage level (1-125)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price level")
    take_profit: Optional[float] = Field(None, description="Take profit price level")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")


class ClosePositionRequest(BaseModel):
    """Model for closing position requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")


class GetPositionsRequest(BaseModel):
    """Model for getting positions requests."""
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")


class GetOpenOrdersRequest(BaseModel):
    """Model for getting open orders requests."""
    symbol: Optional[str] = Field(None, description="Trading pair symbol (optional)")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")


class SetLeverageRequest(BaseModel):
    """Model for setting leverage requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    leverage: int = Field(..., description="Leverage level (1-125)")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")


class TrailingStopRequest(BaseModel):
    """Model for setting trailing stop requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    activation_price: Optional[float] = Field(None, description="Price at which the trailing stop is activated")
    callback_rate: float = Field(..., description="Callback rate in percentage (e.g., 1.0 for 1%)")
    amount: float = Field(..., description="Position size to close with trailing stop")
    side: str = Field(..., description="Side of trailing stop ('buy' or 'sell')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")


class GetBalanceRequest(BaseModel):
    """Model for getting account balance."""
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")


class ExecutionAgent(BaseMarketAgent):
    """
    Specialized agent for executing trades and managing positions on cryptocurrency exchanges.
    Provides functionality for placing orders, monitoring positions, and implementing risk management.
    """
    
    def __init__(self):
        """Initialize the execution agent."""
        super().__init__(
            name="Execution Agent",
            description="A specialized agent for executing trades and managing positions on cryptocurrency exchanges."
        )
        
        # Initialize tools list
        self.tools = []
        
        # Register tools
        self._register_tools()
    
    def _get_instructions(self) -> str:
        """
        Get specific instructions for the execution agent.
        
        Returns:
            String with agent instructions
        """
        return """
        You are the Execution Agent, a specialized agent for executing trades and managing positions on 
        cryptocurrency exchanges.
        
        Your primary responsibilities are:
        
        1. **Execute Trades**: Place market and limit orders based on trading signals
        2. **Manage Positions**: Monitor, adjust, and close existing positions
        3. **Implement Risk Management**: Set stop losses, take profits, and trailing stops
        4. **Track Portfolio Performance**: Monitor account balance and position P/L
        
        ### Your Capabilities:
        
        - Execute precise buy/sell orders on futures exchanges
        - Set and adjust leverage for trades
        - Implement various order types (market, limit, stop, trailing stop)
        - Monitor open positions and orders in real-time
        - Integrate technical analysis into execution decisions
        
        ### Risk Management Guidelines:
        
        1. Always confirm order details before execution
        2. Recommend appropriate position sizing based on account size (1-2% risk per trade)
        3. Always suggest stop loss placement for new positions
        4. Monitor for proper risk-reward ratio (minimum 1:2)
        5. Suggest trailing stops to protect profits on winning trades
        
        Use your tools to execute trades precisely and manage risk effectively.
        When generating signals, be specific about the strategy used and the reasoning behind recommendations.
        """
    
    def _register_tools(self) -> None:
        """Register tools for the execution agent."""
        # Tool for executing trades
        self.add_tool(
            Tool(
                name="execute_trade",
                description="Execute a trade on a cryptocurrency exchange",
                function=self._execute_trade,
                parameters=[
                    ExecuteTradeRequest
                ]
            )
        )
        
        # Tool for closing positions
        self.add_tool(
            Tool(
                name="close_position",
                description="Close an open position on a cryptocurrency exchange",
                function=self._close_position,
                parameters=[
                    ClosePositionRequest
                ]
            )
        )
        
        # Tool for getting open positions
        self.add_tool(
            Tool(
                name="get_positions",
                description="Get all open positions on a cryptocurrency exchange",
                function=self._get_positions,
                parameters=[
                    GetPositionsRequest
                ]
            )
        )
        
        # Tool for getting open orders
        self.add_tool(
            Tool(
                name="get_open_orders",
                description="Get all open orders on a cryptocurrency exchange",
                function=self._get_open_orders,
                parameters=[
                    GetOpenOrdersRequest
                ]
            )
        )
        
        # Tool for setting leverage
        self.add_tool(
            Tool(
                name="set_leverage",
                description="Set leverage for a trading pair on a cryptocurrency exchange",
                function=self._set_leverage,
                parameters=[
                    SetLeverageRequest
                ]
            )
        )
        
        # Tool for setting trailing stop
        self.add_tool(
            Tool(
                name="set_trailing_stop",
                description="Set a trailing stop for an open position",
                function=self._set_trailing_stop,
                parameters=[
                    TrailingStopRequest
                ]
            )
        )
        
        # Tool for getting account balance
        self.add_tool(
            Tool(
                name="get_balance",
                description="Get account balance information",
                function=self._get_balance,
                parameters=[
                    GetBalanceRequest
                ]
            )
        )
    
    def add_tool(self, tool):
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool instance to add
        """
        self.tools.append(tool)
    
    async def _execute_trade(self, params: ExecuteTradeRequest) -> Dict[str, Any]:
        """
        Execute a trade on a cryptocurrency exchange with support for conditional (stop-limit) orders.
        
        Args:
            params: Parameters for the trade execution
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                # Set leverage if provided
                if params.leverage and params.leverage > 1:
                    try:
                        await connector.set_leverage(
                            symbol=params.symbol,
                            leverage=params.leverage
                        )
                        logger.info(f"Set leverage to {params.leverage}x for {params.symbol}")
                    except Exception as e:
                        logger.error(f"Error setting leverage: {e}")
                        return {
                            "success": False,
                            "error": f"Failed to set leverage: {str(e)}"
                        }
                
                # Execute the trade based on order type
                order = None
                if params.order_type.lower() == "market":
                    order = await connector.create_market_order(
                        symbol=params.symbol,
                        side=params.side.lower(),
                        amount=params.amount
                    )
                elif params.order_type.lower() == "limit" and params.price:
                    order = await connector.create_limit_order(
                        symbol=params.symbol,
                        side=params.side.lower(),
                        amount=params.amount,
                        price=params.price
                    )
                elif params.order_type.lower() == "stop_limit" and params.price and params.stop_price:
                    # For stop-limit orders, we need both the stop trigger price and the limit price
                    order_params = {'stopPrice': params.stop_price}
                    order = await connector.create_order(
                        symbol=params.symbol,
                        type='STOP_LIMIT',
                        side=params.side.lower(),
                        amount=params.amount,
                        price=params.price,
                        params=order_params
                    )
                    logger.info(f"Placed stop-limit order for {params.symbol}: Stop at {params.stop_price}, Limit at {params.price}")
                else:
                    return {
                        "success": False,
                        "error": "Invalid order type or missing required parameters. For stop_limit orders, both price and stop_price are required."
                    }
                
                # Set stop loss if provided (only for non-stop-limit entry orders)
                stop_loss_order = None
                if params.stop_loss and params.order_type.lower() != "stop_limit":
                    try:
                        # Determine the stop loss side (opposite of the entry order)
                        sl_side = "sell" if params.side.lower() == "buy" else "buy"
                        
                        stop_loss_order = await connector.create_stop_loss_order(
                            symbol=params.symbol,
                            side=sl_side,
                            amount=params.amount,
                            stop_price=params.stop_loss
                        )
                        
                        logger.info(f"Set stop loss at {params.stop_loss} for {params.symbol}")
                    except Exception as e:
                        logger.error(f"Error setting stop loss: {e}")
                
                # Set take profit if provided
                take_profit_order = None
                if params.take_profit:
                    try:
                        # Determine the take profit side (opposite of the entry order)
                        tp_side = "sell" if params.side.lower() == "buy" else "buy"
                        
                        take_profit_order = await connector.create_take_profit_order(
                            symbol=params.symbol,
                            side=tp_side,
                            amount=params.amount,
                            price=params.take_profit
                        )
                        
                        logger.info(f"Set take profit at {params.take_profit} for {params.symbol}")
                    except Exception as e:
                        logger.error(f"Error setting take profit: {e}")
                
                return {
                    "success": True,
                    "message": f"Successfully executed {params.side} {params.order_type} order for {params.amount} {params.symbol}",
                    "order": order,
                    "stop_loss": stop_loss_order,
                    "take_profit": take_profit_order
                }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _close_position(self, params: ClosePositionRequest) -> Dict[str, Any]:
        """
        Close an open position on a cryptocurrency exchange.
        
        Args:
            params: Parameters for closing the position
            
        Returns:
            Dictionary with closure results
        """
        try:
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                # Get position to determine size and direction
                positions = await connector.fetch_positions()
                
                # Find the position for the specified symbol
                position = None
                for pos in positions:
                    if pos['symbol'] == params.symbol:
                        position = pos
                        break
                
                if not position or float(position['contracts']) == 0:
                    return {
                        "success": False,
                        "error": f"No open position found for {params.symbol}"
                    }
                
                # Determine the side to close the position
                close_side = "sell" if position['side'].lower() == "long" else "buy"
                
                # Close the position with a market order
                close_order = await connector.create_market_order(
                    symbol=params.symbol,
                    side=close_side,
                    amount=abs(float(position['contracts'])),
                    params={"reduceOnly": True}
                )
                
                return {
                    "success": True,
                    "message": f"Successfully closed position for {params.symbol}",
                    "position": position,
                    "close_order": close_order
                }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_positions(self, params: GetPositionsRequest) -> Dict[str, Any]:
        """
        Get all open positions on a cryptocurrency exchange.
        
        Args:
            params: Parameters for getting positions
            
        Returns:
            Dictionary with position data
        """
        try:
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                positions = await connector.fetch_positions()
                
                # Filter out positions with zero contracts
                active_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]
                
                return {
                    "success": True,
                    "positions": active_positions,
                    "count": len(active_positions)
                }
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_open_orders(self, params: GetOpenOrdersRequest) -> Dict[str, Any]:
        """
        Get all open orders on a cryptocurrency exchange.
        
        Args:
            params: Parameters for getting open orders
            
        Returns:
            Dictionary with open order data
        """
        try:
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                if params.symbol:
                    orders = await connector.fetch_open_orders(symbol=params.symbol)
                else:
                    orders = await connector.fetch_open_orders()
                
                return {
                    "success": True,
                    "orders": orders,
                    "count": len(orders)
                }
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _set_leverage(self, params: SetLeverageRequest) -> Dict[str, Any]:
        """
        Set leverage for a trading pair on a cryptocurrency exchange.
        
        Args:
            params: Parameters for setting leverage
            
        Returns:
            Dictionary with leverage setting results
        """
        try:
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                result = await connector.set_leverage(
                    symbol=params.symbol,
                    leverage=params.leverage
                )
                
                return {
                    "success": True,
                    "message": f"Successfully set leverage to {params.leverage}x for {params.symbol}",
                    "result": result
                }
            
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _set_trailing_stop(self, params: TrailingStopRequest) -> Dict[str, Any]:
        """
        Set a trailing stop for an open position.
        
        Args:
            params: Parameters for setting the trailing stop
            
        Returns:
            Dictionary with trailing stop setting results
        """
        try:
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                # Prepare parameters for trailing stop
                order_params = {
                    'callbackRate': params.callback_rate,
                    'reduceOnly': True,
                }
                
                # Add activation price if provided
                if params.activation_price:
                    order_params['activationPrice'] = params.activation_price
                
                # Create trailing stop order
                order = await connector.create_order(
                    symbol=params.symbol,
                    type='TRAILING_STOP_MARKET',
                    side=params.side.lower(),
                    amount=params.amount,
                    params=order_params
                )
                
                return {
                    "success": True,
                    "message": f"Trailing stop order placed successfully with {params.callback_rate}% callback rate.",
                    "order": order
                }
            
        except Exception as e:
            logger.error(f"Error setting trailing stop: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_balance(self, params: GetBalanceRequest) -> Dict[str, Any]:
        """
        Get account balance information from the exchange.
        
        Args:
            params: Parameters for getting balance information
            
        Returns:
            Dictionary with account balance data
        """
        try:
            # Log the request for debugging
            logger.info(f"Retrieving balance information from exchange: {params.exchange or 'default'}")
            
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                # Check if connector is properly initialized
                if not connector or not hasattr(connector, 'exchange_instance'):
                    logger.error("Exchange connector not properly initialized. Check API credentials.")
                    return {
                        "success": False,
                        "error": "Exchange connector not properly initialized. Check API credentials."
                    }
                
                # Fetch balance information
                logger.info("Fetching balance from exchange API...")
                balance = await connector.fetch_balance()
                
                if not balance:
                    logger.error("Received empty balance response from exchange")
                    return {
                        "success": False,
                        "error": "Received empty balance response from exchange"
                    }
                
                logger.info(f"Successfully retrieved balance data: {balance.keys()}")
                
                # Calculate total balance in USDT
                total_usdt_value = 0
                free_balance = {}
                used_balance = {}
                total_balance = {}
                
                # Extract the relevant information
                if 'free' in balance:
                    free_balance = balance['free']
                
                if 'used' in balance:
                    used_balance = balance['used']
                
                if 'total' in balance:
                    total_balance = balance['total']
                    
                    # Calculate USDT value for major coins
                    for coin, amount in total_balance.items():
                        if amount > 0:
                            if coin == 'USDT':
                                total_usdt_value += amount
                            else:
                                try:
                                    # Try to get the current price of the coin in USDT
                                    symbol = f"{coin}/USDT"
                                    ticker = await connector.fetch_ticker(symbol)
                                    coin_price = ticker['last']
                                    coin_value = amount * coin_price
                                    total_usdt_value += coin_value
                                except Exception as e:
                                    logger.warning(f"Could not calculate USDT value for {coin}: {e}")
                    
                    logger.info(f"Estimated total portfolio value: {total_usdt_value} USDT")
                
                return {
                    "success": True,
                    "free": free_balance,
                    "used": used_balance,
                    "total": total_balance,
                    "total_usdt_value": total_usdt_value,
                    "timestamp": balance.get('timestamp', None),
                    "datetime": balance.get('datetime', None)
                }
            
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return {
                "success": False,
                "error": f"Error getting balance: {str(e)}"
            }
