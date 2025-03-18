"""
Enhanced version of the crypto market analysis system with specialized agents.
This version avoids problematic type annotations while implementing the core functionality.
"""
import os
import asyncio
import logging
import platform
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

# Configure Windows-specific event loop policy
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from agents import Agent, Runner, Tool, function_tool
from src.agents.market_data_agent import MarketDataAgent
from src.agents.orderbook_agent import OrderbookAnalysisAgent
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.token_dashboard_agent import TokenDashboardAgent
from src.agents.execution_agent import ExecutionAgent
from src.exchange.connector import ExchangeConnector
from src.data.processor import DataProcessor
from src.trading.order_parser import OrderParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Model definitions for agent parameters
class OrderbookAnalysisParams(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: binanceusdm)")
    depth: Optional[int] = Field(None, description="Depth of orderbook to analyze")

class TechnicalAnalysisParams(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: binanceusdm)")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '15m', '1h', '4h', '1d')")
    indicators: Optional[List[str]] = Field(None, description="List of indicators to calculate")

class PriceParams(BaseModel):
    """
    Parameters for price data retrieval.
    """
    symbol: str
    exchange: Optional[str] = Field(None, description="Exchange ID (default: binanceusdm)")

class MarketSummaryParams(BaseModel):
    """
    Parameters for market summary retrieval.
    """
    symbol: str
    exchange: Optional[str] = Field(None, description="Exchange ID (default: binanceusdm)")

class MarketDataParams(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: binanceusdm)")
    data_type: Optional[str] = Field(None, description="Type of market data to retrieve (price, summary)")
    limit: Optional[int] = Field(None, description="Number of results to return (for volume data)")

class TokenDashboardParams(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: binanceusdm)")
    timeframe: Optional[str] = Field(None, description="Timeframe for analysis (e.g., '1h', '4h', '1d')")
    days_back: Optional[int] = Field(None, description="Number of days of historical data to analyze (max 30)")
    detail_level: Optional[str] = Field(None, description="Level of detail ('low', 'medium', 'high')")

class ExecuteTradeParams(BaseModel):
    """Parameters for trade execution."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    side: str = Field(..., description="Trade direction ('buy' or 'sell')")
    amount: float = Field(..., description="Position size to trade")
    order_type: str = Field(..., description="Order type ('market' or 'limit')")
    price: Optional[float] = Field(None, description="Price for limit orders")
    leverage: Optional[int] = Field(None, description="Leverage level (1-125)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price level")
    take_profit: Optional[float] = Field(None, description="Take profit price level")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")

class ClosePositionParams(BaseModel):
    """Parameters for closing a position."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")

class GetPositionsParams(BaseModel):
    """Parameters for getting positions."""
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")

class GetOpenOrdersParams(BaseModel):
    """Parameters for getting open orders."""
    symbol: Optional[str] = Field(None, description="Trading pair symbol (optional)")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")

class SetLeverageParams(BaseModel):
    """Parameters for setting leverage."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    leverage: int = Field(..., description="Leverage level (1-125)")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")

class TrailingStopParams(BaseModel):
    """Parameters for setting a trailing stop."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    activation_price: Optional[float] = Field(None, description="Price at which the trailing stop is activated")
    callback_rate: float = Field(..., description="Callback rate in percentage (e.g., 1.0 for 1%)")
    amount: float = Field(..., description="Position size to close with trailing stop")
    side: str = Field(..., description="Side of trailing stop ('buy' or 'sell')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")

class GetBalanceParams(BaseModel):
    """Parameters for getting account balance."""
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")

# Real implementation functions
@function_tool
async def analyze_orderbook(params: OrderbookAnalysisParams):
    """
    Analyze the orderbook for a cryptocurrency pair.
    
    Args:
        params: Parameters for orderbook analysis
    """
    symbol = params.symbol
    exchange = params.exchange or "binanceusdm"
    depth = params.depth or 100  # Default to 100 levels if not specified
    
    try:
        # Use real exchange connector
        connector = ExchangeConnector(exchange)
        
        try:
            # Fetch real orderbook data
            orderbook = await connector.fetch_orderbook(symbol, depth)
            
            # Initialize the data processor
            processor = DataProcessor()
            
            # Perform actual orderbook analysis using the data processor
            analysis = processor.analyze_orderbook(orderbook)
            
            # Convert the analysis result to dictionary form
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "analysis": analysis.dict() if hasattr(analysis, 'dict') else analysis
            }
            
            return result
            
        finally:
            # Ensure proper cleanup of resources
            await connector.close()
            
    except Exception as e:
        logger.error(f"Error analyzing orderbook: {e}")
        return {"error": str(e), "symbol": symbol, "exchange": exchange}

@function_tool
async def perform_technical_analysis(params: TechnicalAnalysisParams):
    """
    Perform technical analysis on a cryptocurrency pair.
    
    Args:
        params: Parameters for technical analysis
    """
    symbol = params.symbol
    exchange = params.exchange or "binanceusdm"
    timeframe = params.timeframe or "1h"  # Default to 1h timeframe
    # Default indicators if not specified
    indicators = params.indicators or ["rsi", "macd", "bollinger", "sma", "ema"]
    
    try:
        # Use real exchange connector
        connector = ExchangeConnector(exchange)
        
        try:
            # Fetch real OHLCV data (100 candles)
            ohlcv_data = await connector.fetch_ohlcv(symbol, timeframe, 100)
            
            # Initialize data processor
            processor = DataProcessor()
            
            # Clean and prepare data
            df = processor.clean_ohlcv_data(ohlcv_data)
            
            # Get last candlestick for market data
            market_data = {
                "open": float(df['open'].iloc[-1]) if not df.empty else None,
                "high": float(df['high'].iloc[-1]) if not df.empty else None,
                "low": float(df['low'].iloc[-1]) if not df.empty else None,
                "close": float(df['close'].iloc[-1]) if not df.empty else None,
                "volume": float(df['volume'].iloc[-1]) if not df.empty else None
            }
            
            # Calculate requested indicators
            analysis_results = {}
            
            if "sma" in indicators:
                sma20 = processor.calculate_sma(df, 20)
                sma50 = processor.calculate_sma(df, 50)
                analysis_results["sma"] = {
                    "sma20": float(sma20.values["sma"][-1]) if sma20.values["sma"] else None,
                    "sma50": float(sma50.values["sma"][-1]) if sma50.values["sma"] else None
                }
                
            if "ema" in indicators:
                ema12 = processor.calculate_ema(df, 12)
                ema26 = processor.calculate_ema(df, 26)
                analysis_results["ema"] = {
                    "ema12": float(ema12.values["ema"][-1]) if ema12.values["ema"] else None,
                    "ema26": float(ema26.values["ema"][-1]) if ema26.values["ema"] else None
                }
                
            if "rsi" in indicators:
                rsi = processor.calculate_rsi(df)
                analysis_results["rsi"] = float(rsi.values["rsi"][-1]) if rsi.values["rsi"] else None
                
            if "macd" in indicators:
                macd = processor.calculate_macd(df)
                analysis_results["macd"] = {
                    "value": float(macd.values["macd"][-1]) if macd.values["macd"] else None,
                    "signal": float(macd.values["signal"][-1]) if macd.values["signal"] else None,
                    "histogram": float(macd.values["histogram"][-1]) if macd.values["histogram"] else None
                }
                
            if "bollinger" in indicators:
                bollinger = processor.calculate_bollinger_bands(df)
                analysis_results["bollinger_bands"] = {
                    "upper": float(bollinger.values["upper"][-1]) if bollinger.values["upper"] else None,
                    "middle": float(bollinger.values["middle"][-1]) if bollinger.values["middle"] else None,
                    "lower": float(bollinger.values["lower"][-1]) if bollinger.values["lower"] else None
                }
            
            # Determine trend based on EMAs
            trend = "neutral"
            if "ema" in analysis_results:
                ema12 = analysis_results["ema"]["ema12"]
                ema26 = analysis_results["ema"]["ema26"]
                if ema12 and ema26:
                    if ema12 > ema26:
                        trend = "bullish"
                    elif ema12 < ema26:
                        trend = "bearish"
            
            # Determine key levels using support/resistance detection
            key_levels = await connector.fetch_key_levels(symbol, timeframe) if hasattr(connector, 'fetch_key_levels') else None
            if not key_levels:
                # Fallback to calculating support/resistance from OHLCV data
                levels = processor.identify_support_resistance_levels(df) if hasattr(processor, 'identify_support_resistance_levels') else None
                key_levels = levels if levels else {
                    "support": [float(min(df['low'].tail(20))), float(min(df['low'].tail(50)))],
                    "resistance": [float(max(df['high'].tail(20))), float(max(df['high'].tail(50)))]
                }
            
            # Calculate momentum
            momentum = 0.5  # neutral by default
            if "rsi" in analysis_results and analysis_results["rsi"]:
                rsi_value = analysis_results["rsi"]
                if rsi_value > 70:
                    momentum = 0.8  # strong positive momentum
                elif rsi_value > 60:
                    momentum = 0.65  # positive momentum
                elif rsi_value < 30:
                    momentum = 0.2  # strong negative momentum
                elif rsi_value < 40:
                    momentum = 0.35  # negative momentum
            
            # Calculate volatility using ATR if available
            volatility = None
            try:
                atr = processor.calculate_atr(df, 14) if hasattr(processor, 'calculate_atr') else None
                if atr and atr.values.get("atr") and len(atr.values["atr"]) > 0:
                    # Normalize ATR as percentage of price
                    last_price = df['close'].iloc[-1] if not df.empty else None
                    if last_price:
                        volatility = float(atr.values["atr"][-1]) / last_price
                    else:
                        volatility = 0.34  # fallback
                else:
                    # Calculate simple volatility as standard deviation
                    volatility = float(df['close'].pct_change().std()) if not df.empty else 0.34
            except Exception:
                volatility = 0.34  # fallback
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "market_data": market_data,
                "analysis": {
                    "trend": trend,
                    "momentum": momentum,
                    "volatility": volatility,
                    "key_levels": key_levels,
                    "indicators": analysis_results
                }
            }
            
        finally:
            # Ensure proper cleanup of resources
            await connector.close()
            
    except Exception as e:
        logger.error(f"Error performing technical analysis: {e}")
        return {"error": str(e), "symbol": symbol, "exchange": exchange, "timeframe": timeframe}

@function_tool
async def get_price(params: PriceParams):
    """
    Get the current price for a cryptocurrency pair.
    
    Args:
        params: Parameters for price data retrieval
    """
    # Use default values if parameters are not provided
    symbol = params.symbol
    exchange = params.exchange or "binanceusdm"
    
    # Normalize exchange to lowercase
    exchange = exchange.lower()
    
    connector = None
    try:
        # Create a direct connection to the exchange
        connector = ExchangeConnector(exchange)
        
        try:
            # Fetch ticker data directly
            ticker = await connector.fetch_ticker(symbol)
            
            # Extract just the price information
            return {
                "symbol": symbol,
                "exchange": exchange,
                "price": float(ticker.get('last', 0.0) or 0.0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error retrieving price data: {e}")
            return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error retrieving price data: {e}")
        return {"error": str(e)}
    finally:
        # Ensure proper cleanup of resources
        if connector:
            await connector.close()

@function_tool
async def get_market_summary(params: MarketSummaryParams):
    """
    Get a detailed market summary for a cryptocurrency pair including price, volume, and 24h statistics.
    
    Args:
        params: Parameters for market summary retrieval
    """
    # Use default values if parameters are not provided
    symbol = params.symbol
    exchange = params.exchange or "binanceusdm"
    
    # Normalize exchange to lowercase
    exchange = exchange.lower()
    
    connector = None
    try:
        # Create a direct connection to the exchange
        connector = ExchangeConnector(exchange)
        
        try:
            # Fetch ticker data directly
            ticker = await connector.fetch_ticker(symbol)
            
            # Safely extract values with defaults for missing fields, handling None values
            last_price = float(ticker.get('last', 0.0) or 0.0)
            bid = float(ticker.get('bid', 0.0) or 0.0)
            ask = float(ticker.get('ask', 0.0) or 0.0)
            volume_24h = float(ticker.get('quoteVolume', ticker.get('volume', 0.0)) or 0.0)
            percent_change = float(ticker.get('percentage', ticker.get('change', 0.0)) or 0.0)
            high_24h = float(ticker.get('high', 0.0) or 0.0)
            low_24h = float(ticker.get('low', 0.0) or 0.0)
            timestamp = ticker.get('timestamp', int(datetime.now().timestamp() * 1000))
            
            # Format the response
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp / 1000).isoformat(),
                "price_data": {
                    "last_price": round(last_price, 8),
                    "bid": round(bid, 8),
                    "ask": round(ask, 8),
                    "spread": round(ask - bid, 8) if ask > 0 and bid > 0 else 0.0,
                    "spread_percentage": round((ask - bid) / bid * 100, 4) if ask > 0 and bid > 0 else 0.0,
                },
                "24h_stats": {
                    "high": round(high_24h, 8),
                    "low": round(low_24h, 8),
                    "volume": round(volume_24h, 2),
                    "percent_change": round(percent_change, 2),
                    "range_percentage": round((high_24h - low_24h) / low_24h * 100, 2) if low_24h > 0 else 0.0,
                }
            }
        except Exception as e:
            logger.error(f"Error retrieving market summary: {e}")
            return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error retrieving market summary: {e}")
        return {"error": str(e)}
    finally:
        # Ensure proper cleanup of resources
        if connector:
            await connector.close()

@function_tool
async def get_token_dashboard(params: TokenDashboardParams):
    """
    Generate a comprehensive token dashboard with market data, technical analysis, and orderbook analysis.
    
    Args:
        params: Parameters for token dashboard generation
    """
    symbol = params.symbol
    exchange = params.exchange or "binanceusdm"
    timeframe = params.timeframe or "1h"
    days_back = min(params.days_back or 7, 30)  # Cap at 30 days
    detail_level = params.detail_level or "medium"
    
    try:
        # Create exchange connector
        connector = ExchangeConnector(exchange)
        
        try:
            # Call dashboard controller directly
            from src.data.token_dashboard.dashboard_controller import get_token_dashboard as get_dashboard
            
            dashboard_data = await get_dashboard(
                exchange=connector.exchange_instance,
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
                days_back=days_back,
                detail_level=detail_level
            )
            
            # Structure the response
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "days_analyzed": days_back,
                "detail_level": detail_level,
                "dashboard": dashboard_data
            }
            
            return result
            
        finally:
            # Ensure proper cleanup of resources
            await connector.close()
            
    except Exception as e:
        logger.error(f"Error generating token dashboard: {e}")
        return {"error": str(e)}

@function_tool
async def execute_trade(params: ExecuteTradeParams) -> Dict[str, Any]:
    """
    Execute a trade on a cryptocurrency exchange.
    
    Args:
        params: Parameters for trade execution
    
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
            
            # Execute the trade
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
            else:
                return {
                    "success": False,
                    "error": "Invalid order type or missing price for limit order"
                }
            
            # Set stop loss if provided
            stop_loss_order = None
            if params.stop_loss:
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

@function_tool
async def close_position(params: ClosePositionParams) -> Dict[str, Any]:
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

@function_tool
async def get_positions(params: GetPositionsParams) -> Dict[str, Any]:
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

@function_tool
async def get_open_orders(params: GetOpenOrdersParams) -> Dict[str, Any]:
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

@function_tool
async def set_leverage(params: SetLeverageParams) -> Dict[str, Any]:
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

@function_tool
async def set_trailing_stop(params: TrailingStopParams) -> Dict[str, Any]:
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

@function_tool
async def get_balance(params: GetBalanceParams) -> Dict[str, Any]:
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

async def setup_crypto_agents():
    """Setup and initialize all crypto market analysis agents."""
    try:
        logger.info("Setting up specialized crypto market analysis agents...")
        
        # Initialize specialized agents
        orderbook_agent = Agent(
            name="Orderbook Analysis Agent",
            instructions="""
            You are the Orderbook Analysis Agent, specialized in analyzing market depth and liquidity.
            
            Analyze order book data to identify:
            - Buy/sell ratio and pressure
            - Liquidity distribution
            - Support and resistance levels
            - Order book imbalances
            - Large orders and walls
            - Depth distribution and cumulative volume
            - Market microstructure patterns
            
            Provide detailed insights about market structure and potential price action based on order book analysis.
            """,
            tools=[analyze_orderbook]
        )
        
        # Create the technical analysis agent
        technical_agent = Agent(
            name="Technical Analysis Agent",
            instructions="""
            You are the Technical Analysis Agent, specialized in analyzing price charts and indicators.
            
            Analyze price data to identify:
            - Trend direction and strength
            - Support and resistance levels
            - Key technical indicators (RSI, MACD, Bollinger Bands, etc.)
            - Advanced indicators (ATR, OBV, Stochastic, etc.)
            - Chart patterns and potential breakouts
            - Market structure and volatility
            
            Provide insights about potential price movements based on technical analysis.
            """,
            tools=[perform_technical_analysis]
        )
        
        # Create the market data agent
        market_data_agent = Agent(
            name="Market Data Agent",
            instructions="""
            You are the Market Data Agent, specialized in retrieving current market data.
            
            When retrieving market data, use the appropriate data_type parameter:
            - For simple price queries, use data_type="price"
            - For detailed market information, use data_type="summary"
            
            Provide accurate and up-to-date information about:
            - Current prices of cryptocurrency pairs (use data_type="price" for basic price only)
            - Detailed market summaries with bid/ask prices, 24h statistics, and volume (use data_type="summary")
            
            When users ask for comprehensive analysis or a "full picture" of a token, use the Token Dashboard Agent
            which provides integrated insights from multiple data sources.
            
            For trading-related requests, use the Execution Agent to:
            - Execute trades with proper risk management
            - Monitor and close positions
            - Set and adjust leverage
            - Implement trailing stops and other advanced order types
            - Generate trading signals based on various strategies
            
            Always maintain a helpful, informative tone and explain complex concepts in clear terms.
            """,
            tools=[get_price, get_market_summary]
        )
        
        # Create the token dashboard agent
        token_dashboard_agent = Agent(
            name="Token Dashboard Agent",
            instructions="""
            You are the Token Dashboard Agent, specialized in providing comprehensive token analysis.
            
            Generate detailed dashboards that include:
            - Current market data and price information
            - Historical price analysis and trends
            - Volume analysis and unusual volume patterns
            - Orderbook depth and liquidity analysis
            - Technical indicator analysis and trading signals
            - Futures-specific data (funding rates, open interest) when available
            
            Focus on presenting the most important insights from each section, highlighting significant patterns,
            anomalies, or potential trading opportunities. Explain technical concepts in an accessible way
            and provide context about why certain indicators or metrics are important.
            
            Deliver comprehensive, multi-faceted analysis that gives users a complete picture of a token's
            current market status and potential future movements.
            """,
            tools=[get_token_dashboard]
        )
        
        # Create the execution agent
        execution_agent = Agent(
            name="Execution Agent",
            instructions="""
            You are the Execution Agent, specialized in executing trades and managing positions.

            Actions you can perform:
            - Execute market and limit orders
            - Close open positions
            - Set leverage for trading
            - Get account balance information
            - Get all open positions
            - Get all open orders
            - Set trailing stops for open positions

            Provide clear and accurate information about trade execution and position management.
            """,
            tools=[
                execute_trade,
                close_position,
                get_positions,
                get_open_orders,
                set_leverage,
                set_trailing_stop,
                get_balance
            ]
        )
        
        # Create the main orchestration agent
        main_agent = Agent(
            name="Crypto Market Analysis Orchestrator",
            instructions="""
            You are the Crypto Market Analysis Orchestrator, responsible for coordinating specialized analysis agents.
            
            You can delegate tasks to the following specialized agents:
            
            1. Market Data Agent: For retrieving current prices, market summaries, and volume information
            2. Orderbook Analysis Agent: For analyzing market depth, liquidity, and order book patterns
            3. Technical Analysis Agent: For performing technical analysis using indicators and chart patterns
            4. Token Dashboard Agent: For comprehensive token analysis combining all data sources
            5. Execution Agent: For executing trades, managing positions, and implementing risk management
            
            Based on the user's question, determine which specialized agent(s) would be most appropriate to handle it,
            and delegate accordingly. Synthesize the information from multiple agents when necessary to provide
            comprehensive answers.
            
            When users ask for comprehensive analysis or a "full picture" of a token, use the Token Dashboard Agent
            which provides integrated insights from multiple data sources.
            
            For trading-related requests, use the Execution Agent to:
            - Execute trades with proper risk management
            - Monitor and close positions
            - Set and adjust leverage
            - Implement trailing stops and other advanced order types
            - Generate trading signals based on various strategies
            
            Always maintain a helpful, informative tone and explain complex concepts in clear terms.
            """,
            tools=[
                market_data_agent.as_tool(
                    tool_name="get_price",
                    tool_description="Get the current price of a cryptocurrency pair"
                ),
                market_data_agent.as_tool(
                    tool_name="get_market_summary",
                    tool_description="Get a detailed market summary with price, volume, and 24h statistics"
                ),
                orderbook_agent.as_tool(
                    tool_name="analyze_orderbook_data",
                    tool_description="Analyze market depth, liquidity, and order book patterns"
                ),
                technical_agent.as_tool(
                    tool_name="perform_technical_analysis",
                    tool_description="Perform technical analysis using indicators and chart patterns"
                ),
                token_dashboard_agent.as_tool(
                    tool_name="get_token_dashboard",
                    tool_description="Generate a comprehensive token dashboard with market data, technical analysis, and orderbook analysis"
                ),
                execution_agent.as_tool(
                    tool_name="execute_trade",
                    tool_description="Execute a trade on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="close_position",
                    tool_description="Close an open position on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="get_positions",
                    tool_description="Get all open positions on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="get_open_orders",
                    tool_description="Get all open orders on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="set_leverage",
                    tool_description="Set leverage for a trading pair on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="set_trailing_stop",
                    tool_description="Set a trailing stop for an open position"
                ),
                execution_agent.as_tool(
                    tool_name="get_balance",
                    tool_description="Get account balance information from the exchange"
                )
            ]
        )
        
        logger.info("All specialized crypto market analysis agents have been set up successfully!")
        
        return main_agent
    
    except Exception as e:
        logger.error(f"Error setting up crypto agents: {e}")
        raise

async def handle_user_message(message_text: str, user_id: str, session_id: str = None) -> str:
    """
    Process a user message and generate a response.
    
    Args:
        message_text: The message from the user
        user_id: Unique user identifier
        session_id: Optional session identifier
        
    Returns:
        Assistant response message
    """
    logger.info(f"Processing message: {message_text}")
    
    # Check if this is a trade execution request
    if message_text.lower().startswith("execute") or message_text.lower().startswith("excute"):
        return await process_execution_request(message_text)
    
    # Regular chat processing
    try:
        # Initialize or get existing conversation manager
        conversation_manager = get_conversation_manager(user_id)
        
        # Add the user message to the conversation
        conversation_manager.add_message("user", message_text)
        
        # Process with the main agent
        response = await main_agent.arun(prompt=message_text)
        
        # Add the assistant's response to the conversation
        conversation_manager.add_message("assistant", response)
        
        return response
    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        return f"I encountered an error while processing your message: {str(e)}"

async def process_execution_request(message_text: str) -> str:
    """
    Process a trade execution request.
    
    Args:
        message_text: The execution request message
        
    Returns:
        Response confirming execution or error
    """
    try:
        # Extract the order part from the message
        order_text = re.search(r'(?:execute|excute)\s+this\s+order\s+"(.+)"', message_text, re.IGNORECASE | re.DOTALL)
        
        if not order_text:
            return "I couldn't understand the order details. Please provide your order in a clear format."
        
        # Parse the order
        order_params = OrderParser.parse_order(order_text.group(1))
        
        # Validate essential parameters
        missing_params = []
        if not order_params.get("symbol"):
            missing_params.append("Trading pair symbol")
        if not order_params.get("amount") and not order_params.get("position_percentage"):
            missing_params.append("Position size")
        
        if missing_params:
            return f"I couldn't execute your order because the following information is missing: {', '.join(missing_params)}"
        
        # Execute the order
        result = await OrderParser.execute_order(order_params)
        
        if result["success"]:
            response = f"""
✅ **Order Executed Successfully**

**Details:**
- **Trading Pair:** {order_params['symbol']}
- **Side:** {order_params['side'].upper()}
- **Type:** {order_params['order_type'].upper()}
- **Amount:** {order_params['amount'] or f"{order_params['position_percentage']}% of balance"}
- **Leverage:** {order_params['leverage']}x

**Risk Management:**
"""
            if result.get("stop_loss"):
                response += f"- **Stop Loss:** Set at ${order_params['stop_loss']}\n"
            
            if result.get("take_profit"):
                response += f"- **Take Profit:** Set at ${order_params['take_profit']}\n"
            
            if result.get("additional_take_profits"):
                tp_levels = order_params.get("take_profit_levels", [])[1:]
                response += f"- **Additional Take Profit Levels:** {', '.join([f'${level}' for level in tp_levels])}\n"
            
            return response
        else:
            return f"❌ **Order Execution Failed**\n\nError: {result['error']}\n\nPlease check your order details and try again."
    
    except Exception as e:
        logger.exception(f"Error processing execution request: {e}")
        return f"I encountered an error while trying to execute your order: {str(e)}"

async def main():
    """Main function to run the enhanced crypto market analysis agent system."""
    try:
        agent = await setup_crypto_agents()
        
        print("Enhanced Crypto Market Analysis System - Type 'exit' to quit")
        print("----------------------------------------------------------")
        
        while True:
            user_input = input("\nEnter your question (or 'exit' to quit): ")
            
            if user_input.lower() == 'exit':
                print("Exiting the agent system. Goodbye!")
                break
                
            try:
                print("\nProcessing your request...")
                
                # Run the agent
                result = await Runner.run(agent, user_input)
                
                # Display the response
                print("\n🤖 Agent Response:")
                print("-----------------")
                print(result.final_output)
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"\nAn error occurred while processing your request: {e}")
                
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"\nFailed to initialize the agent system: {e}")

if __name__ == "__main__":
    asyncio.run(main())
