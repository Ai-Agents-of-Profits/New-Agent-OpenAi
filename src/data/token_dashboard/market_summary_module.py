"""
Market Summary Module

This module handles fetching and processing market summary data for tokens.
"""

from typing import Any, Dict
from datetime import datetime
import ccxt.async_support as ccxt


async def get_market_summary_data(exchange: ccxt.Exchange, exchange_id: str, symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive market summary information.
    
    Args:
        exchange: The exchange instance
        exchange_id: ID of the exchange (e.g., 'binanceusdm')
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
    
    Returns:
        Dictionary with market summary data
    """
    # Fetch ticker information
    ticker = await exchange.fetch_ticker(symbol)
    
    # Calculate additional metrics
    base_currency, quote_currency = symbol.split('/')
    quote_currency = quote_currency.split(':')[0] if ':' in quote_currency else quote_currency
    
    bid_ask_spread = 0
    spread_percent = 0
    
    if ticker.get('bid') and ticker.get('ask'):
        bid_ask_spread = ticker['ask'] - ticker['bid']
        if ticker['bid'] > 0:
            spread_percent = (bid_ask_spread / ticker['bid']) * 100
    
    # Format timestamps properly
    timestamp = datetime.fromtimestamp(ticker.get('timestamp', 0)/1000).strftime('%Y-%m-%d %H:%M:%S') if ticker.get('timestamp') else 'N/A'
    
    # Compile market summary
    return {
        "symbol": symbol,
        "exchange": exchange_id,
        "last_price": ticker.get('last'),
        "quote_currency": quote_currency,
        "bid": ticker.get('bid'),
        "ask": ticker.get('ask'),
        "spread": bid_ask_spread,
        "spread_percent": spread_percent,
        "24h_high": ticker.get('high'),
        "24h_low": ticker.get('low'),
        "24h_change": ticker.get('change'),
        "24h_change_percent": ticker.get('percentage'),
        "24h_base_volume": ticker.get('baseVolume'),
        "24h_quote_volume": ticker.get('quoteVolume'),
        "timestamp": timestamp,
        "datetime": timestamp
    }
