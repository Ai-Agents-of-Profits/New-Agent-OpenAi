"""
Futures Module

This module handles fetching and processing futures-specific data for tokens.
"""

from typing import Any, Dict, Optional
import ccxt.async_support as ccxt


async def get_futures_specific_data(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Get futures-specific data including funding rates, open interest, etc.
    
    Args:
        exchange: The exchange instance
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
    
    Returns:
        Dictionary with futures-specific data
    """
    futures_data = {}
    
    # Funding rate
    try:
        funding_info = await exchange.fetch_funding_rate(symbol)
        if funding_info:
            funding_rate = funding_info.get('fundingRate')
            next_funding_time = funding_info.get('nextFundingTime')
            
            # Format next funding time to readable datetime
            if next_funding_time:
                from datetime import datetime
                next_funding_datetime = datetime.fromtimestamp(next_funding_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            else:
                next_funding_datetime = None
                
            futures_data['funding_rate'] = funding_rate
            futures_data['funding_rate_percent'] = funding_rate * 100 if funding_rate else None
            futures_data['next_funding_time'] = next_funding_time
            futures_data['next_funding_datetime'] = next_funding_datetime
            
            # Calculate annual funding yield if applicable
            if funding_rate:
                # Assuming 3 funding payments per day (common for many exchanges)
                annual_funding_yield = funding_rate * 3 * 365 * 100  # annualized percentage
                futures_data['annual_funding_yield'] = annual_funding_yield
    except Exception:
        pass
    
    # Mark/Index price and fair basis
    try:
        mark_price_info = await exchange.fetch_mark_price(symbol)
        if mark_price_info:
            futures_data['mark_price'] = mark_price_info.get('markPrice')
            futures_data['index_price'] = mark_price_info.get('indexPrice')
            
            # Calculate fair basis if both prices are available
            mark = mark_price_info.get('markPrice')
            index = mark_price_info.get('indexPrice')
            if mark and index and index > 0:
                basis = (mark - index) / index * 100  # percentage
                futures_data['fair_basis'] = basis
    except Exception:
        pass
    
    # Open interest
    try:
        open_interest = await exchange.fetch_open_interest(symbol)
        if open_interest:
            futures_data['open_interest'] = open_interest.get('openInterest')
            futures_data['open_interest_value'] = open_interest.get('openInterestValue')
    except Exception:
        pass
    
    # Leverage information
    try:
        market = await exchange.fetch_market(symbol)
        if market:
            max_leverage = market.get('limits', {}).get('leverage', {}).get('max')
            if max_leverage:
                futures_data['max_leverage'] = max_leverage
    except Exception:
        pass
    
    return futures_data
