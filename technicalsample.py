"""
Technical Analysis module for cryptocurrency trading signals
Provides common indicators and pattern detection for trading decisions
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union


def moving_average(data: List[float], period: int = 14) -> List[float]:
    """
    Calculate Simple Moving Average (SMA) over a series of values
    
    Args:
        data: List of price values
        period: Window period for calculation
        
    Returns:
        List of SMA values (with NaN/None values for the first period-1 positions)
    """
    if len(data) < period:
        return [None] * len(data)
    
    sma = []
    for i in range(len(data)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(data[i-(period-1):i+1]) / period)
    
    return sma


def exponential_moving_average(data: List[float], period: int = 14) -> List[float]:
    """
    Calculate Exponential Moving Average (EMA) over a series of values
    
    Args:
        data: List of price values
        period: Window period for calculation
        
    Returns:
        List of EMA values (with NaN/None values for the first period-1 positions)
    """
    if len(data) < period:
        return [None] * len(data)
    
    ema = [None] * (period - 1)
    # First EMA value is SMA
    ema.append(sum(data[:period]) / period)
    
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema.append((data[i] * multiplier) + (ema[i-1] * (1 - multiplier)))
    
    return ema


def volume_weighted_ma(prices: List[float], volumes: List[float], period: int = 14) -> List[float]:
    """
    Calculate Volume Weighted Moving Average (VWMA)
    
    Args:
        prices: List of price values
        volumes: List of volume values
        period: Window period for calculation
        
    Returns:
        List of VWMA values (with NaN/None values for the first period-1 positions)
    """
    if len(prices) < period or len(prices) != len(volumes):
        return [None] * len(prices)
    
    vwma = []
    for i in range(len(prices)):
        if i < period - 1:
            vwma.append(None)
        else:
            price_vol_sum = sum(prices[i-(period-1) + j] * volumes[i-(period-1) + j] for j in range(period))
            vol_sum = sum(volumes[i-(period-1):i+1])
            vwma.append(price_vol_sum / vol_sum if vol_sum > 0 else None)
    
    return vwma


def relative_strength_index(data: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index (RSI) over a series of values
    
    Args:
        data: List of price values
        period: Window period for calculation
        
    Returns:
        List of RSI values (with None values for the first period positions)
    """
    if len(data) <= period:
        return [None] * len(data)
    
    # Calculate price changes
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    
    # Create initial lists
    rsi = [None] * (period + 1)  # +1 because we already used first price for deltas
    
    # Calculate initial averages
    avg_gain = sum(max(delta, 0) for delta in deltas[:period]) / period
    avg_loss = sum(abs(min(delta, 0)) for delta in deltas[:period]) / period
    
    # Calculate first RSI
    if avg_loss == 0:
        rsi.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi.append(100 - (100 / (1 + rs)))
    
    # Calculate remaining RSIs
    for i in range(period + 1, len(data)):
        delta = data[i] - data[i-1]
        
        gain = max(delta, 0)
        loss = abs(min(delta, 0))
        
        # Smoothed averages
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
    
    return rsi


def bollinger_bands(data: List[float], period: int = 20, deviation_multiplier: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands (middle band, upper band, lower band)
    
    Args:
        data: List of price values
        period: Period for moving average calculation
        deviation_multiplier: Number of standard deviations for bands
        
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    if len(data) < period:
        empty = [None] * len(data)
        return empty, empty, empty
    
    # Calculate middle band (SMA)
    middle_band = moving_average(data, period)
    
    # Calculate standard deviation and bands
    upper_band = []
    lower_band = []
    
    for i in range(len(data)):
        if i < period - 1:
            upper_band.append(None)
            lower_band.append(None)
        else:
            std_dev = np.std(data[i-(period-1):i+1])
            upper_band.append(middle_band[i] + (deviation_multiplier * std_dev))
            lower_band.append(middle_band[i] - (deviation_multiplier * std_dev))
    
    return middle_band, upper_band, lower_band


def macd(data: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: List of price values
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Calculate EMAs
    fast_ema = exponential_moving_average(data, fast_period)
    slow_ema = exponential_moving_average(data, slow_period)
    
    # Calculate MACD line
    macd_line = []
    for i in range(len(data)):
        if i < slow_period - 1 or fast_ema[i] is None or slow_ema[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(fast_ema[i] - slow_ema[i])
    
    # Calculate signal line (EMA of MACD line)
    # We need to create a version of the MACD line without None values for the EMA calculation
    valid_macd = [x for x in macd_line if x is not None]
    valid_signal = exponential_moving_average(valid_macd, signal_period)
    
    # Reconstruct signal line with proper None values
    signal_line = [None] * (len(macd_line) - len(valid_signal))
    signal_line.extend(valid_signal)
    
    # Calculate histogram
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is None or signal_line[i] is None:
            histogram.append(None)
        else:
            histogram.append(macd_line[i] - signal_line[i])
    
    return macd_line, signal_line, histogram


def average_true_range(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """
    Calculate Average True Range (ATR)
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: Period for ATR calculation
        
    Returns:
        List of ATR values
    """
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return [None] * len(highs)
    
    # Calculate True Range
    tr = [None]  # TR requires previous close
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_prev_close = abs(highs[i] - closes[i-1])
        low_prev_close = abs(lows[i] - closes[i-1])
        
        tr.append(max(high_low, high_prev_close, low_prev_close))
    
    # Calculate ATR
    atr = [None] * period
    # First ATR is simple average of first 'period' TR values
    atr.append(sum(tr[1:period+1]) / period)
    
    # Remaining ATRs use smoothing
    for i in range(period + 1, len(closes)):
        atr.append(((atr[i-1] * (period - 1)) + tr[i]) / period)
    
    return atr


def stochastic_oscillator(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        k_period: Period for %K line
        d_period: Period for %D line (SMA of %K)
        
    Returns:
        Tuple of (%K, %D)
    """
    if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
        return [None] * len(closes), [None] * len(closes)
    
    # Calculate %K
    k_values = []
    for i in range(len(closes)):
        if i < k_period - 1:
            k_values.append(None)
        else:
            window_high = max(highs[i-(k_period-1):i+1])
            window_low = min(lows[i-(k_period-1):i+1])
            
            if window_high == window_low:
                k_values.append(50)  # Prevent division by zero
            else:
                k_values.append(((closes[i] - window_low) / (window_high - window_low)) * 100)
    
    # Calculate %D (SMA of %K)
    valid_k = [x for x in k_values if x is not None]
    d_values = []
    
    for i in range(len(k_values)):
        if i < k_period - 1 + d_period - 1:
            d_values.append(None)
        elif k_values[i-(d_period-1):i+1].count(None) > 0:
            d_values.append(None)
        else:
            d_values.append(sum(k_values[i-(d_period-1):i+1]) / d_period)
    
    return k_values, d_values


def on_balance_volume(closes: List[float], volumes: List[float]) -> List[float]:
    """
    Calculate On Balance Volume (OBV)
    
    Args:
        closes: List of close prices
        volumes: List of volume values
        
    Returns:
        List of OBV values
    """
    if not closes or not volumes or len(closes) != len(volumes):
        return [None] * len(closes)
    
    obv = [volumes[0]]  # First OBV is just the first volume
    
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[i-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[i-1] - volumes[i])
        else:
            obv.append(obv[i-1])
    
    return obv


def detect_candlestick_patterns(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict[int, List[str]]:
    """
    Detect common candlestick patterns
    
    Args:
        opens: List of open prices
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        
    Returns:
        Dictionary mapping candle indices to detected patterns
    """
    if not opens or not highs or not lows or not closes:
        return {}
    
    patterns = {}
    
    for i in range(len(closes)):
        if i < 1:  # Need at least 2 candles for patterns
            continue
        
        current_patterns = []
        
        # Calculate some common metrics
        body_size = abs(closes[i] - opens[i])
        range_size = highs[i] - lows[i]
        prev_body_size = abs(closes[i-1] - opens[i-1])
        prev_range_size = highs[i-1] - lows[i-1]
        
        # Is bullish candle
        is_bullish = closes[i] > opens[i]
        prev_bullish = closes[i-1] > opens[i-1]
        
        # Doji
        if body_size <= 0.05 * range_size:
            current_patterns.append("Doji")
        
        # Hammer / Hanging Man
        if (body_size > 0) and (highs[i] - max(opens[i], closes[i])) <= 0.1 * body_size and (min(opens[i], closes[i]) - lows[i]) >= 2 * body_size:
            if i > 1 and closes[i-2] < closes[i-1]:  # Downtrend before
                current_patterns.append("Hammer")
            else:
                current_patterns.append("Hanging Man")
        
        # Bullish Engulfing
        if is_bullish and not prev_bullish and opens[i] < closes[i-1] and closes[i] > opens[i-1]:
            current_patterns.append("Bullish Engulfing")
        
        # Bearish Engulfing
        if not is_bullish and prev_bullish and opens[i] > closes[i-1] and closes[i] < opens[i-1]:
            current_patterns.append("Bearish Engulfing")
        
        # Morning Star (needs 3 candles)
        if i >= 2 and not prev_bullish and is_bullish and not closes[i-2] > opens[i-2] and body_size > 0:
            second_candle_small = abs(closes[i-1] - opens[i-1]) < 0.3 * abs(closes[i-2] - opens[i-2])
            if second_candle_small and closes[i] > (opens[i-2] + closes[i-2]) / 2:
                current_patterns.append("Morning Star")
        
        # Evening Star (needs 3 candles)
        if i >= 2 and prev_bullish and not is_bullish and closes[i-2] > opens[i-2] and body_size > 0:
            second_candle_small = abs(closes[i-1] - opens[i-1]) < 0.3 * abs(closes[i-2] - opens[i-2])
            if second_candle_small and closes[i] < (opens[i-2] + closes[i-2]) / 2:
                current_patterns.append("Evening Star")
        
        if current_patterns:
            patterns[i] = current_patterns
    
    return patterns


def identify_support_resistance(highs: List[float], lows: List[float], period: int = 10, threshold: float = 0.03) -> Tuple[List[float], List[float]]:
    """
    Identify support and resistance levels from price action
    
    Args:
        highs: List of high prices
        lows: List of low prices
        period: Look-back period for identifying pivot points
        threshold: Price proximity threshold for combining levels
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    if len(highs) < period * 2 or len(lows) < period * 2:
        return [], []
    
    # Identify pivot highs and lows
    pivot_highs = []
    pivot_lows = []
    
    for i in range(period, len(highs) - period):
        # Check if this is a pivot high
        if all(highs[i] > highs[i-j] for j in range(1, period+1)) and all(highs[i] > highs[i+j] for j in range(1, period+1)):
            pivot_highs.append(highs[i])
        
        # Check if this is a pivot low
        if all(lows[i] < lows[i-j] for j in range(1, period+1)) and all(lows[i] < lows[i+j] for j in range(1, period+1)):
            pivot_lows.append(lows[i])
    
    # Combine nearby levels
    resistance_levels = []
    support_levels = []
    
    # Function to combine levels
    def combine_levels(levels, threshold_pct):
        if not levels:
            return []
        
        levels.sort()
        combined = [levels[0]]
        
        for level in levels[1:]:
            if level > combined[-1] * (1 + threshold_pct):
                combined.append(level)
            else:
                combined[-1] = (combined[-1] + level) / 2  # Average the levels
        
        return combined
    
    resistance_levels = combine_levels(pivot_highs, threshold)
    support_levels = combine_levels(pivot_lows, threshold)
    
    return support_levels, resistance_levels
