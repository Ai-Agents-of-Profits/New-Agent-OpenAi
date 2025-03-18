"""
Dashboard Controller Module

This module coordinates the collection and formatting of comprehensive token data
from various sources.
"""

import asyncio
from typing import Any, Dict, List, Optional
import ccxt.async_support as ccxt
from datetime import datetime, timedelta

# Local imports
from .price_module import get_price_data
from .market_summary_module import get_market_summary_data
from .orderbook_module import get_orderbook_analysis
from .technical_module import get_technical_analysis_data
from .historical_module import get_historical_data, get_volume_history
from .futures_module import get_futures_specific_data
from .formatter import format_token_dashboard


async def get_token_dashboard(
    exchange: ccxt.Exchange,
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    days_back: int = 7,
    detail_level: str = "medium"
) -> Dict[str, Any]:
    """
    Generate a comprehensive dashboard for a specific token.
    
    Args:
        exchange: The exchange instance
        exchange_id: ID of the exchange (e.g., 'binanceusdm')
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for historical data (e.g., '1h', '4h', '1d')
        days_back: Number of days of historical data to analyze
        detail_level: Level of detail to include ('low', 'medium', 'high')
    
    Returns:
        Formatted dashboard data
    """
    # Parallelize data collection to minimize waiting time
    tasks = [
        get_price_data(exchange, symbol),
        get_market_summary_data(exchange, exchange_id, symbol),
        get_orderbook_analysis(exchange, symbol),
        get_technical_analysis_data(exchange, symbol, timeframe, days_back),
        get_historical_data(exchange, symbol, timeframe, days_back),
        get_volume_history(exchange, symbol, min(days_back, 30))
    ]
    
    # Add futures-specific data if applicable
    is_futures = exchange_id.lower() in ['binanceusdm', 'binancecoinm', 'bybit']
    if is_futures:
        tasks.append(get_futures_specific_data(exchange, symbol))
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and handle any exceptions
    dashboard_data = {
        "timestamp": datetime.now().isoformat(),
        "exchange": exchange_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "detail_level": detail_level
    }
    
    # Extract results from the tasks
    price_data = results[0] if not isinstance(results[0], Exception) else None
    market_summary = results[1] if not isinstance(results[1], Exception) else None
    orderbook = results[2] if not isinstance(results[2], Exception) else None
    technical = results[3] if not isinstance(results[3], Exception) else None
    historical = results[4] if not isinstance(results[4], Exception) else None
    volume = results[5] if not isinstance(results[5], Exception) else None
    
    # Add futures data if available
    futures_data = None
    if is_futures and len(results) > 6:
        futures_data = results[6] if not isinstance(results[6], Exception) else None
    
    # Combine all data
    dashboard_data.update({
        "price_data": price_data,
        "market_summary": market_summary,
        "orderbook_analysis": orderbook,
        "technical_analysis": technical,
        "historical_data": historical,
        "volume_history": volume,
        "futures_data": futures_data
    })
    
    # Debug output to see data structure
    print("Technical Analysis Data has patterns:", "patterns" in technical if technical else False)
    if technical and "patterns" in technical:
        print("Patterns found:", technical["patterns"])
    
    # Format the dashboard according to the requested detail level
    formatted_dashboard = format_token_dashboard(dashboard_data, detail_level)
    
    return formatted_dashboard
