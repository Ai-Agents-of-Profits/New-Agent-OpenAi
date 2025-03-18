"""
Historical Data Module

This module handles fetching and processing historical price and volume data for tokens.
"""

from typing import Any, Dict, List
from datetime import datetime, timedelta
import ccxt.async_support as ccxt


async def get_historical_data(
    exchange: ccxt.Exchange, 
    symbol: str, 
    timeframe: str = "1h",
    days_back: int = 7
) -> Dict[str, Any]:
    """
    Get historical OHLCV data with price change analysis.
    
    Args:
        exchange: The exchange instance
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
        days_back: Number of days of historical data to analyze
    
    Returns:
        Dictionary with historical data and price change analysis
    """
    # Calculate timestamps
    since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    # Fetch historical data
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since)
    
    # Process the data
    processed_data = []
    prev_close = None
    
    for candle in ohlcv:
        timestamp, open_price, high, low, close, volume = candle
        
        # Calculate price change from previous candle
        change = 0
        change_percent = 0
        if prev_close is not None and prev_close > 0:
            change = close - prev_close
            change_percent = (change / prev_close) * 100
        
        # Calculate candle body and wick sizes
        body_size = abs(close - open_price)
        if close >= open_price:
            upper_wick = high - close
            lower_wick = open_price - low
            is_bullish = True
        else:
            upper_wick = high - open_price
            lower_wick = close - low
            is_bullish = False
        
        # Add to processed data
        processed_data.append({
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S'),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "change": change,
            "change_percent": change_percent,
            "body_size": body_size,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "is_bullish": is_bullish
        })
        
        prev_close = close
    
    # Analyze the complete dataset
    analysis = analyze_historical_data(processed_data)
    
    return {
        "timeframe": timeframe,
        "candles": processed_data,
        "analysis": analysis
    }


def analyze_historical_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze historical price data to extract key metrics and insights.
    
    Args:
        data: List of processed OHLCV candles
    
    Returns:
        Dictionary with analysis results
    """
    if not data:
        return {"error": "No data to analyze"}
    
    # Extract lists for easier processing
    closes = [candle["close"] for candle in data]
    highs = [candle["high"] for candle in data]
    lows = [candle["low"] for candle in data]
    volumes = [candle["volume"] for candle in data]
    changes = [candle["change_percent"] for candle in data if "change_percent" in candle]
    
    # Key price levels
    current_price = closes[-1] if closes else 0
    highest_price = max(highs) if highs else 0
    lowest_price = min(lows) if lows else 0
    
    # Calculate volatility (standard deviation of percentage changes)
    volatility = 0
    if len(changes) > 1:
        import numpy as np
        volatility = np.std(changes)
    
    # Count bullish vs bearish candles
    bullish_candles = sum(1 for candle in data if candle.get("is_bullish", False))
    bearish_candles = len(data) - bullish_candles
    
    # Calculate total price change over period
    start_price = data[0]["open"] if data else 0
    total_change = current_price - start_price if start_price > 0 else 0
    total_change_percent = (total_change / start_price) * 100 if start_price > 0 else 0
    
    # Calculate average volume
    avg_volume = sum(volumes) / len(volumes) if volumes else 0
    
    # Identify volume trends
    volume_trend = "neutral"
    if len(volumes) > 5:
        recent_avg = sum(volumes[-5:]) / 5
        earlier_avg = sum(volumes[:-5]) / (len(volumes) - 5)
        
        if recent_avg > earlier_avg * 1.2:
            volume_trend = "increasing"
        elif recent_avg < earlier_avg * 0.8:
            volume_trend = "decreasing"
    
    # Identify price trend
    price_trend = "neutral"
    if len(closes) > 20:
        short_term_avg = sum(closes[-5:]) / 5
        medium_term_avg = sum(closes[-10:]) / 10
        long_term_avg = sum(closes) / len(closes)
        
        if short_term_avg > medium_term_avg > long_term_avg:
            price_trend = "strong uptrend"
        elif short_term_avg > medium_term_avg:
            price_trend = "uptrend"
        elif short_term_avg < medium_term_avg < long_term_avg:
            price_trend = "strong downtrend"
        elif short_term_avg < medium_term_avg:
            price_trend = "downtrend"
    
    return {
        "current_price": current_price,
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "price_range_percent": ((highest_price - lowest_price) / lowest_price) * 100 if lowest_price > 0 else 0,
        "total_change": total_change,
        "total_change_percent": total_change_percent,
        "volatility": volatility,
        "bullish_candles": bullish_candles,
        "bearish_candles": bearish_candles,
        "sentiment": "bullish" if bullish_candles > bearish_candles else "bearish",
        "avg_volume": avg_volume,
        "volume_trend": volume_trend,
        "price_trend": price_trend
    }


async def get_volume_history(
    exchange: ccxt.Exchange, 
    symbol: str, 
    days: int = 7
) -> Dict[str, Any]:
    """
    Get trading volume history with analysis.
    
    Args:
        exchange: The exchange instance
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
        days: Number of days of volume history to retrieve
    
    Returns:
        Dictionary with volume history and analysis
    """
    # Get daily volume data
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    ohlcv = await exchange.fetch_ohlcv(symbol, "1d", since=since)
    
    # Process the data
    volume_data = []
    for candle in ohlcv:
        timestamp, open_price, high, low, close, volume = candle
        dt = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
        
        volume_data.append({
            "date": dt,
            "volume": volume,
            "price_open": open_price,
            "price_close": close,
            "is_up_day": close > open_price
        })
    
    # Analyze volume trends
    analysis = analyze_volume_data(volume_data)
    
    return {
        "volume_history": volume_data,
        "analysis": analysis
    }


def analyze_volume_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze volume data to identify trends and anomalies.
    
    Args:
        data: List of processed volume data
    
    Returns:
        Dictionary with volume analysis
    """
    if not data:
        return {"error": "No data to analyze"}
    
    volumes = [day["volume"] for day in data]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0
    
    # Find highest and lowest volume days
    highest_volume = max(volumes) if volumes else 0
    highest_volume_day = next((day for day in data if day["volume"] == highest_volume), None)
    
    lowest_volume = min(volumes) if volumes else 0
    lowest_volume_day = next((day for day in data if day["volume"] == lowest_volume), None)
    
    # Check if volume is increasing or decreasing
    volume_trend = "neutral"
    if len(volumes) > 3:
        recent_avg = sum(volumes[-3:]) / 3
        earlier_avg = sum(volumes[:-3]) / (len(volumes) - 3)
        
        if recent_avg > earlier_avg * 1.2:
            volume_trend = "increasing"
        elif recent_avg < earlier_avg * 0.8:
            volume_trend = "decreasing"
    
    # Check volume on up days vs down days
    up_day_volumes = [day["volume"] for day in data if day.get("is_up_day", False)]
    down_day_volumes = [day["volume"] for day in data if not day.get("is_up_day", True)]
    
    avg_up_volume = sum(up_day_volumes) / len(up_day_volumes) if up_day_volumes else 0
    avg_down_volume = sum(down_day_volumes) / len(down_day_volumes) if down_day_volumes else 0
    
    volume_bias = "neutral"
    if avg_up_volume > avg_down_volume * 1.2:
        volume_bias = "bullish (higher volume on up days)"
    elif avg_down_volume > avg_up_volume * 1.2:
        volume_bias = "bearish (higher volume on down days)"
    
    # Check for volume spikes
    volume_spikes = []
    for i, day in enumerate(data):
        if i == 0:
            continue
            
        prev_volume = data[i-1]["volume"]
        if day["volume"] > prev_volume * 2:
            volume_spikes.append({
                "date": day["date"],
                "volume": day["volume"],
                "increase": (day["volume"] / prev_volume) - 1,
                "price_change": (day["price_close"] - day["price_open"]) / day["price_open"] * 100
            })
    
    return {
        "avg_daily_volume": avg_volume,
        "highest_volume": {
            "date": highest_volume_day["date"] if highest_volume_day else "N/A",
            "volume": highest_volume
        },
        "lowest_volume": {
            "date": lowest_volume_day["date"] if lowest_volume_day else "N/A",
            "volume": lowest_volume
        },
        "volume_trend": volume_trend,
        "volume_bias": volume_bias,
        "volume_spikes": volume_spikes,
        "volume_consistency": calculate_volume_consistency(volumes)
    }


def calculate_volume_consistency(volumes: List[float]) -> str:
    """
    Calculate how consistent trading volume has been.
    
    Args:
        volumes: List of volume values
    
    Returns:
        Description of volume consistency
    """
    if not volumes or len(volumes) < 3:
        return "insufficient data"
    
    import numpy as np
    avg = np.mean(volumes)
    std = np.std(volumes)
    
    coefficient_of_variation = std / avg if avg > 0 else float('inf')
    
    if coefficient_of_variation < 0.3:
        return "very consistent"
    elif coefficient_of_variation < 0.5:
        return "consistent"
    elif coefficient_of_variation < 1.0:
        return "moderately variable"
    else:
        return "highly variable"
