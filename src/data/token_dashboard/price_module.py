"""
Price Module

This module handles fetching and processing current price data for tokens.
"""

from typing import Any, Dict, Optional
import ccxt.async_support as ccxt


async def get_price_data(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Get current price data including last price, bid/ask, and price changes.
    
    Args:
        exchange: The exchange instance
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
    
    Returns:
        Dictionary with price data
    """
    # Fetch current ticker information
    ticker = await exchange.fetch_ticker(symbol)
    
    # Extract relevant price data
    price_data = {
        "last": ticker.get('last'),
        "bid": ticker.get('bid'),
        "ask": ticker.get('ask'),
        "high": ticker.get('high'),
        "low": ticker.get('low'),
        "percentage": ticker.get('percentage'),
        "change": ticker.get('change'),
        "average": ticker.get('average'),
        "baseVolume": ticker.get('baseVolume'),
        "quoteVolume": ticker.get('quoteVolume'),
        "timestamp": ticker.get('timestamp')
    }
    
    # Try to get funding rate if available (for futures exchanges)
    try:
        funding_info = await exchange.fetch_funding_rate(symbol)
        if funding_info and 'fundingRate' in funding_info:
            price_data['fundingRate'] = funding_info['fundingRate']
            price_data['nextFundingTime'] = funding_info.get('nextFundingTime')
    except Exception:
        pass  # Skip if not available
    
    # Try to get mark price if available (for futures exchanges)
    try:
        mark = await exchange.fetch_mark_price(symbol)
        if mark and 'markPrice' in mark:
            price_data['markPrice'] = mark['markPrice']
            price_data['indexPrice'] = mark.get('indexPrice')
    except Exception:
        pass  # Skip if not available
    
    return price_data
