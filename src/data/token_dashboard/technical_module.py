"""
Technical Analysis Module

This module handles fetching and analyzing technical analysis data for tokens.
"""

from typing import Any, Dict, List
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
import sys
import os

# Add parent directory to sys.path to import technicalsample
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import technicalsample as ta


async def get_technical_analysis_data(
    exchange: ccxt.Exchange, 
    symbol: str, 
    timeframe: str = "1h",
    days_back: int = 7
) -> Dict[str, Any]:
    """
    Get comprehensive technical analysis data.
    
    Args:
        exchange: The exchange instance
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
        days_back: Number of days of historical data to analyze
    
    Returns:
        Dictionary with technical analysis data
    """
    # Calculate timestamps
    since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    # Fetch historical data
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since)
    
    # Extract OHLCV components
    timestamps = [candle[0] for candle in ohlcv]
    opens = [candle[1] for candle in ohlcv]
    highs = [candle[2] for candle in ohlcv]
    lows = [candle[3] for candle in ohlcv]
    closes = [candle[4] for candle in ohlcv]
    volumes = [candle[5] for candle in ohlcv]
    
    # Calculate standard indicators
    results = {
        "timestamps": timestamps, 
        "open": opens, 
        "high": highs, 
        "low": lows, 
        "close": closes, 
        "volume": volumes
    }
    
    # Moving Averages
    results["sma"] = ta.moving_average(closes)
    results["ema"] = ta.exponential_moving_average(closes)
    results["vwma"] = ta.volume_weighted_ma(closes, volumes)
    
    # Oscillators
    results["rsi"] = ta.relative_strength_index(closes)
    results["macd"] = ta.macd(closes)
    results["stoch"] = ta.stochastic_oscillator(highs, lows, closes)
    
    # Volatility indicators
    results["bollinger"] = ta.bollinger_bands(closes)
    results["atr"] = ta.average_true_range(highs, lows, closes)
    
    # Volume indicators
    results["obv"] = ta.on_balance_volume(closes, volumes)
    
    # Patterns and Support/Resistance
    results["patterns"] = ta.detect_candlestick_patterns(opens, highs, lows, closes)
    results["support_resistance"] = ta.identify_support_resistance(highs, lows)
    
    # Add indicator interpretations
    results["interpretations"] = interpret_indicators(results)
    
    return results


def interpret_indicators(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Interpret the technical indicators to provide trading insights.
    
    Args:
        data: Dictionary containing all calculated indicators
    
    Returns:
        Dictionary with interpretations of each indicator
    """
    interpretations = {}
    
    # Get the close prices for reference
    close_prices = data.get("close", [])
    
    # RSI Interpretation
    if "rsi" in data and data["rsi"]:
        if isinstance(data["rsi"], list):
            rsi_values = data["rsi"]
        else:
            rsi_values = []
            
        last_rsi = rsi_values[-1] if rsi_values else None
        
        if last_rsi is not None:
            if last_rsi < 30:
                interpretations["rsi"] = "Oversold - potential buy signal"
            elif last_rsi > 70:
                interpretations["rsi"] = "Overbought - potential sell signal"
            else:
                interpretations["rsi"] = "Neutral"
    
    # MACD Interpretation
    if "macd" in data and data["macd"]:
        # Handle the case when macd is a tuple or a dictionary
        if isinstance(data["macd"], dict):
            macd_line = data["macd"].get("macd", [])
            signal_line = data["macd"].get("signal", [])
            histogram = data["macd"].get("histogram", [])
        elif isinstance(data["macd"], tuple) and len(data["macd"]) >= 3:
            # If it's a tuple, assume it contains (macd, signal, histogram)
            macd_line = data["macd"][0] if len(data["macd"]) > 0 else []
            signal_line = data["macd"][1] if len(data["macd"]) > 1 else []
            histogram = data["macd"][2] if len(data["macd"]) > 2 else []
        else:
            macd_line, signal_line, histogram = [], [], []
        
        if macd_line and signal_line and histogram:
            last_macd = macd_line[-1]
            last_signal = signal_line[-1]
            last_hist = histogram[-1]
            prev_hist = histogram[-2] if len(histogram) > 1 else 0
            
            if last_macd > last_signal and last_hist > 0 and last_hist > prev_hist:
                interpretations["macd"] = "Bullish - buy signal"
            elif last_macd < last_signal and last_hist < 0 and last_hist < prev_hist:
                interpretations["macd"] = "Bearish - sell signal"
            elif last_macd > last_signal:
                interpretations["macd"] = "Bullish - momentum building"
            elif last_macd < last_signal:
                interpretations["macd"] = "Bearish - momentum declining"
            else:
                interpretations["macd"] = "Neutral"
    
    # Bollinger Bands Interpretation
    if "bollinger" in data and data["bollinger"]:
        # Handle the case when bollinger is a tuple or a dictionary
        if isinstance(data["bollinger"], dict):
            upper_band = data["bollinger"].get("upper", [])
            middle_band = data["bollinger"].get("middle", [])
            lower_band = data["bollinger"].get("lower", [])
        elif isinstance(data["bollinger"], tuple) and len(data["bollinger"]) >= 3:
            # If it's a tuple, assume it contains (upper, middle, lower)
            upper_band = data["bollinger"][0] if len(data["bollinger"]) > 0 else []
            middle_band = data["bollinger"][1] if len(data["bollinger"]) > 1 else []
            lower_band = data["bollinger"][2] if len(data["bollinger"]) > 2 else []
        else:
            upper_band, middle_band, lower_band = [], [], []
        
        if upper_band and middle_band and lower_band and len(close_prices) > 0:
            last_upper = upper_band[-1]
            last_middle = middle_band[-1]
            last_lower = lower_band[-1]
            last_close = close_prices[-1]
            
            band_width = (last_upper - last_lower) / last_middle
            
            if last_close > last_upper:
                interpretations["bollinger"] = "Price above upper band - potentially overbought"
            elif last_close < last_lower:
                interpretations["bollinger"] = "Price below lower band - potentially oversold"
            else:
                interpretations["bollinger"] = "Price within bands - neutral"
                
            # Band width for volatility
            if band_width > 0.05:  # Arbitrary threshold
                interpretations["volatility"] = "High volatility"
            else:
                interpretations["volatility"] = "Low volatility"
    
    # Stochastic Oscillator Interpretation
    if "stoch" in data and data["stoch"]:
        # Handle the case when stoch is a tuple or a dictionary
        if isinstance(data["stoch"], dict):
            k_line = data["stoch"].get("k", [])
            d_line = data["stoch"].get("d", [])
        elif isinstance(data["stoch"], tuple) and len(data["stoch"]) >= 2:
            # If it's a tuple, assume it contains (k, d)
            k_line = data["stoch"][0] if len(data["stoch"]) > 0 else []
            d_line = data["stoch"][1] if len(data["stoch"]) > 1 else []
        else:
            k_line, d_line = [], []
            
        if k_line and d_line:
            last_k = k_line[-1]
            last_d = d_line[-1]
            
            if last_k < 20 and last_d < 20:
                interpretations["stoch"] = "Oversold - potential buy signal"
            elif last_k > 80 and last_d > 80:
                interpretations["stoch"] = "Overbought - potential sell signal"
            elif last_k > last_d and last_k > k_line[-2]:
                interpretations["stoch"] = "Bullish - momentum building"
            elif last_k < last_d and last_k < k_line[-2]:
                interpretations["stoch"] = "Bearish - momentum declining"
            else:
                interpretations["stoch"] = "Neutral"
    
    # On-Balance Volume (OBV) Interpretation
    if "obv" in data and data["obv"] and isinstance(data["obv"], list) and len(data["obv"]) > 5:
        obv_values = data["obv"]
        recent_obv = obv_values[-5:]
        
        # Calculate trend
        obv_trend = sum(1 if recent_obv[i] > recent_obv[i-1] else -1 for i in range(1, len(recent_obv)))
        
        if obv_trend > 2:
            interpretations["obv"] = "Strong volume confirming uptrend"
        elif obv_trend < -2:
            interpretations["obv"] = "Strong volume confirming downtrend"
        elif obv_trend > 0:
            interpretations["obv"] = "Moderate volume supporting uptrend"
        elif obv_trend < 0:
            interpretations["obv"] = "Moderate volume supporting downtrend"
        else:
            interpretations["obv"] = "Neutral volume pattern"
    
    # ATR Interpretation for volatility
    if "atr" in data and data["atr"] and isinstance(data["atr"], list) and len(data["atr"]) > 0:
        atr_values = data["atr"]
        current_atr = atr_values[-1]
        
        if len(close_prices) > 0:
            atr_percent = current_atr / close_prices[-1] * 100
            
            if atr_percent > 3:
                interpretations["atr"] = "Very high volatility"
            elif atr_percent > 2:
                interpretations["atr"] = "High volatility"
            elif atr_percent > 1:
                interpretations["atr"] = "Moderate volatility"
            else:
                interpretations["atr"] = "Low volatility"
    
    # Support and Resistance Levels
    if "support_resistance" in data and data["support_resistance"]:
        if isinstance(data["support_resistance"], dict):
            support = data["support_resistance"].get("support", [])
            resistance = data["support_resistance"].get("resistance", [])
        elif isinstance(data["support_resistance"], tuple) and len(data["support_resistance"]) >= 2:
            support = data["support_resistance"][0] if len(data["support_resistance"]) > 0 else []
            resistance = data["support_resistance"][1] if len(data["support_resistance"]) > 1 else []
        else:
            support, resistance = [], []
        
        if support and resistance and len(close_prices) > 0:
            current_price = close_prices[-1]
            
            # Find nearest support and resistance
            nearest_support = max([s for s in support if s < current_price], default=None)
            nearest_resistance = min([r for r in resistance if r > current_price], default=None)
            
            if nearest_support and nearest_resistance:
                support_dist = (current_price - nearest_support) / current_price * 100
                resistance_dist = (nearest_resistance - current_price) / current_price * 100
                
                if support_dist < 1.0:
                    interpretations["sr_levels"] = f"Price near support at {nearest_support:.2f}"
                elif resistance_dist < 1.0:
                    interpretations["sr_levels"] = f"Price near resistance at {nearest_resistance:.2f}"
                else:
                    interpretations["sr_levels"] = f"Support at {nearest_support:.2f}, resistance at {nearest_resistance:.2f}"
    
    # Overall trend determination based on SMA
    if "sma" in data and data["sma"]:
        if isinstance(data["sma"], dict):
            sma_50 = data["sma"].get("50", [])
            sma_200 = data["sma"].get("200", [])
        elif isinstance(data["sma"], tuple) and len(data["sma"]) >= 2:
            sma_50 = data["sma"][0] if len(data["sma"]) > 0 else []
            sma_200 = data["sma"][1] if len(data["sma"]) > 1 else []
        else:
            sma_50, sma_200 = [], []
            
        if sma_50 and sma_200 and len(close_prices) > 0:
            current_price = close_prices[-1]
            last_sma_50 = sma_50[-1]
            last_sma_200 = sma_200[-1]
            
            if current_price > last_sma_50 and current_price > last_sma_200:
                if last_sma_50 > last_sma_200:
                    interpretations["trend"] = "Strong uptrend (golden cross)"
                else:
                    interpretations["trend"] = "Potential trend reversal (bullish)"
            elif current_price < last_sma_50 and current_price < last_sma_200:
                if last_sma_50 < last_sma_200:
                    interpretations["trend"] = "Strong downtrend (death cross)"
                else:
                    interpretations["trend"] = "Potential trend reversal (bearish)"
            elif current_price > last_sma_50:
                interpretations["trend"] = "Short-term uptrend"
            elif current_price < last_sma_50:
                interpretations["trend"] = "Short-term downtrend"
    
    return interpretations
