"""
Order parser module for handling natural language trade order execution.
"""
import re
import logging
from typing import Dict, Any, Optional, Tuple
import asyncio

from ..exchange.connector import ExchangeConnector

logger = logging.getLogger(__name__)

class OrderParser:
    """
    Parser for natural language trade orders.
    Converts text descriptions of orders into executable parameters
    and interfaces with the ExchangeConnector for execution.
    """
    
    @staticmethod
    def parse_order(order_text: str) -> Dict[str, Any]:
        """
        Parse a natural language order text into structured parameters.
        
        Args:
            order_text: Natural language description of the order
            
        Returns:
            Dictionary with parsed order parameters
        """
        # Initialize result with defaults
        result = {
            "symbol": None,
            "order_type": "market",
            "side": "buy",
            "amount": None,
            "price": None,
            "leverage": 1,
            "stop_loss": None,
            "take_profit": None,
            "exchange": None,
            "take_profit_levels": [],
            "position_percentage": None
        }
        
        # Extract trading pair
        symbol_match = re.search(r'(?:Trading Pair|Symbol|Pair):\s*([A-Za-z0-9]+/[A-Za-z0-9]+)', order_text)
        if symbol_match:
            result["symbol"] = symbol_match.group(1).strip()
        
        # Extract order type
        if re.search(r'(?:Order Type|Type):\s*Limit', order_text, re.IGNORECASE):
            result["order_type"] = "limit"
        elif re.search(r'(?:Order Type|Type):\s*Market', order_text, re.IGNORECASE):
            result["order_type"] = "market"
        
        # Extract side
        if re.search(r'(?:Side|Direction):\s*(?:Sell|Short)', order_text, re.IGNORECASE) or 'sell' in order_text.lower() or 'short' in order_text.lower():
            result["side"] = "sell"
        
        # Extract leverage
        leverage_match = re.search(r'(?:Leverage):\s*(\d+)x', order_text)
        if leverage_match:
            result["leverage"] = int(leverage_match.group(1).strip())
        
        # Extract position size
        size_match = re.search(r'(?:Position Size|Size):\s*(\d+)%', order_text)
        if size_match:
            result["position_percentage"] = int(size_match.group(1).strip())
        else:
            # Look for position size in other formats
            size_match = re.search(r'(?:Position Size|Size):\s*([\d.]+)', order_text)
            if size_match:
                result["amount"] = float(size_match.group(1).strip())
        
        # Extract exchange
        exchange_match = re.search(r'(?:Exchange):\s*([A-Za-z0-9]+)', order_text)
        if exchange_match:
            result["exchange"] = exchange_match.group(1).strip().lower()
        
        # Extract entry price for limit orders
        price_match = re.search(r'(?:Entry Point|Entry Price|Price):\s*(?:Above|Below)?\s*\$?([\d.]+)', order_text)
        if price_match:
            result["price"] = float(price_match.group(1).strip())
        
        # Extract stop loss
        sl_match = re.search(r'(?:Stop Loss|SL):\s*(?:Below|Above)?\s*\$?([\d.]+)', order_text)
        if sl_match:
            result["stop_loss"] = float(sl_match.group(1).strip())
        
        # Extract take profit targets
        tp_match = re.search(r'(?:Targets|Take Profit|TP):\s*\$?([\d.]+)(?:\s*and\s*\$?([\d.]+))?', order_text)
        if tp_match:
            result["take_profit"] = float(tp_match.group(1).strip())
            result["take_profit_levels"] = [float(tp_match.group(1).strip())]
            if tp_match.group(2):
                result["take_profit_levels"].append(float(tp_match.group(2).strip()))
        
        return result
    
    @staticmethod
    async def execute_order(order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an order with the given parameters.
        
        Args:
            order_params: Dictionary with order parameters
            
        Returns:
            Dictionary with execution results
        """
        try:
            logger.info(f"Executing order: {order_params}")
            
            # Validate required parameters
            if not order_params.get("symbol"):
                return {
                    "success": False,
                    "error": "Missing trading pair symbol"
                }
            
            # Get exchange connector
            async with ExchangeConnector(order_params.get("exchange")) as connector:
                # Get balance to calculate position size if percentage provided
                if order_params.get("position_percentage") and not order_params.get("amount"):
                    try:
                        balance = await connector.fetch_balance()
                        total_value = balance.get('total', {}).get('USDT', 0)
                        
                        # Calculate position size based on percentage
                        position_size_percentage = order_params.get("position_percentage", 100) / 100.0
                        order_params["amount"] = total_value * position_size_percentage
                        
                        logger.info(f"Calculated position size: {order_params['amount']} USDT ({position_size_percentage * 100}% of {total_value} USDT)")
                    except Exception as e:
                        logger.error(f"Error calculating position size: {e}")
                        return {
                            "success": False,
                            "error": f"Failed to calculate position size: {str(e)}"
                        }
                
                # Set leverage if provided
                if order_params.get("leverage", 1) > 1:
                    try:
                        await connector.set_leverage(
                            symbol=order_params["symbol"],
                            leverage=order_params["leverage"]
                        )
                        logger.info(f"Set leverage to {order_params['leverage']}x for {order_params['symbol']}")
                    except Exception as e:
                        logger.error(f"Error setting leverage: {e}")
                        return {
                            "success": False,
                            "error": f"Failed to set leverage: {str(e)}"
                        }
                
                # Execute the primary order
                order_result = None
                if order_params["order_type"].lower() == "market":
                    order_result = await connector.create_market_order(
                        symbol=order_params["symbol"],
                        side=order_params["side"].lower(),
                        amount=order_params["amount"]
                    )
                elif order_params["order_type"].lower() == "limit" and order_params.get("price"):
                    order_result = await connector.create_limit_order(
                        symbol=order_params["symbol"],
                        side=order_params["side"].lower(),
                        amount=order_params["amount"],
                        price=order_params["price"]
                    )
                else:
                    return {
                        "success": False,
                        "error": "Invalid order type or missing price for limit order"
                    }
                
                results = {
                    "success": True,
                    "message": f"Successfully executed {order_params['side']} {order_params['order_type']} order for {order_params['amount']} {order_params['symbol']}",
                    "order": order_result,
                    "stop_loss": None,
                    "take_profit": None
                }
                
                # Set stop loss if provided
                if order_params.get("stop_loss"):
                    try:
                        # Determine stop loss side (opposite of entry)
                        sl_side = "sell" if order_params["side"].lower() == "buy" else "buy"
                        
                        stop_loss_order = await connector.create_stop_loss_order(
                            symbol=order_params["symbol"],
                            side=sl_side,
                            amount=order_params["amount"],
                            stop_price=order_params["stop_loss"]
                        )
                        
                        results["stop_loss"] = stop_loss_order
                        logger.info(f"Set stop loss at {order_params['stop_loss']} for {order_params['symbol']}")
                    except Exception as e:
                        logger.error(f"Error setting stop loss: {e}")
                
                # Set take profit if provided
                if order_params.get("take_profit"):
                    try:
                        # Determine take profit side (opposite of entry)
                        tp_side = "sell" if order_params["side"].lower() == "buy" else "buy"
                        
                        take_profit_order = await connector.create_take_profit_order(
                            symbol=order_params["symbol"],
                            side=tp_side,
                            amount=order_params["amount"],
                            price=order_params["take_profit"]
                        )
                        
                        results["take_profit"] = take_profit_order
                        logger.info(f"Set take profit at {order_params['take_profit']} for {order_params['symbol']}")
                    except Exception as e:
                        logger.error(f"Error setting take profit: {e}")
                
                # Set multiple take profit levels if provided
                if order_params.get("take_profit_levels") and len(order_params["take_profit_levels"]) > 1:
                    multi_tp_results = []
                    tp_side = "sell" if order_params["side"].lower() == "buy" else "buy"
                    
                    # Calculate amounts for partial take profits
                    position_divided = order_params["amount"] / len(order_params["take_profit_levels"])
                    
                    for idx, tp_level in enumerate(order_params["take_profit_levels"]):
                        try:
                            if idx > 0:  # Skip first TP level as it's already set above
                                tp_order = await connector.create_take_profit_order(
                                    symbol=order_params["symbol"],
                                    side=tp_side,
                                    amount=position_divided,
                                    price=tp_level
                                )
                                multi_tp_results.append(tp_order)
                                logger.info(f"Set additional take profit at {tp_level} for {position_divided} {order_params['symbol']}")
                        except Exception as e:
                            logger.error(f"Error setting additional take profit at {tp_level}: {e}")
                    
                    results["additional_take_profits"] = multi_tp_results
                
                return results
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return {
                "success": False,
                "error": str(e)
            }
