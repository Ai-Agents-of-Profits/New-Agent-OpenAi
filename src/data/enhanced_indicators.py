"""
Enhanced Technical Analysis Indicators for cryptocurrency trading
Provides additional indicators and pattern detection to complement existing DataProcessor functionality
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union

class IndicatorResult:
    """Container for indicator calculation results."""
    
    def __init__(self, values=None, metadata=None):
        """Initialize with calculated values and optional metadata."""
        self.values = values or {}
        self.metadata = metadata or {}

def calculate_atr(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """
    Calculate Average True Range (ATR) for measuring volatility
    
    Args:
        df: DataFrame with OHLCV data
        period: Period for ATR calculation
        
    Returns:
        IndicatorResult with ATR values
    """
    try:
        if len(df) < 2:
            return IndicatorResult(values={"atr": []})
        
        # Make a copy to avoid modifying original
        _df = df.copy()
        
        # Calculate True Range
        _df['previous_close'] = _df['close'].shift(1)
        _df['tr1'] = abs(_df['high'] - _df['low'])
        _df['tr2'] = abs(_df['high'] - _df['previous_close'])
        _df['tr3'] = abs(_df['low'] - _df['previous_close'])
        _df['tr'] = _df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        _df['atr'] = _df['tr'].rolling(window=period).mean()
        
        # Convert to list and handle NaN values
        atr_values = _df['atr'].fillna(0).tolist()
        
        return IndicatorResult(values={"atr": atr_values})
        
    except Exception as e:
        # Return empty result on error
        return IndicatorResult(values={"atr": []}, metadata={"error": str(e)})

def calculate_obv(df: pd.DataFrame) -> IndicatorResult:
    """
    Calculate On Balance Volume (OBV) for volume flow analysis
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        IndicatorResult with OBV values
    """
    try:
        if len(df) < 2:
            return IndicatorResult(values={"obv": []})
        
        # Make a copy to avoid modifying original
        _df = df.copy()
        
        # Initialize OBV with first row volume
        _df['obv'] = 0
        _df.loc[0, 'obv'] = _df.loc[0, 'volume']
        
        # Calculate OBV
        for i in range(1, len(_df)):
            if _df.loc[i, 'close'] > _df.loc[i-1, 'close']:
                _df.loc[i, 'obv'] = _df.loc[i-1, 'obv'] + _df.loc[i, 'volume']
            elif _df.loc[i, 'close'] < _df.loc[i-1, 'close']:
                _df.loc[i, 'obv'] = _df.loc[i-1, 'obv'] - _df.loc[i, 'volume']
            else:
                _df.loc[i, 'obv'] = _df.loc[i-1, 'obv']
        
        # Convert to list
        obv_values = _df['obv'].tolist()
        
        return IndicatorResult(values={"obv": obv_values})
        
    except Exception as e:
        # Return empty result on error
        return IndicatorResult(values={"obv": []}, metadata={"error": str(e)})

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorResult:
    """
    Calculate Stochastic Oscillator for momentum analysis
    
    Args:
        df: DataFrame with OHLCV data
        k_period: Period for %K line
        d_period: Period for %D line (SMA of %K)
        
    Returns:
        IndicatorResult with Stochastic Oscillator values
    """
    try:
        if len(df) < k_period:
            return IndicatorResult(values={"k_line": [], "d_line": []})
        
        # Make a copy to avoid modifying original
        _df = df.copy()
        
        # Calculate %K
        _df['lowest_low'] = _df['low'].rolling(window=k_period).min()
        _df['highest_high'] = _df['high'].rolling(window=k_period).max()
        
        # Handle potential division by zero
        _df['highest_low_range'] = _df['highest_high'] - _df['lowest_low']
        _df.loc[_df['highest_low_range'] == 0, 'highest_low_range'] = 0.001  # Small value to prevent division by zero
        
        _df['k_line'] = 100 * ((_df['close'] - _df['lowest_low']) / _df['highest_low_range'])
        
        # Calculate %D (SMA of %K)
        _df['d_line'] = _df['k_line'].rolling(window=d_period).mean()
        
        # Convert to list and handle NaN values
        k_values = _df['k_line'].fillna(0).tolist()
        d_values = _df['d_line'].fillna(0).tolist()
        
        return IndicatorResult(values={"k_line": k_values, "d_line": d_values})
        
    except Exception as e:
        # Return empty result on error
        return IndicatorResult(values={"k_line": [], "d_line": []}, metadata={"error": str(e)})

def calculate_vwma(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """
    Calculate Volume Weighted Moving Average (VWMA)
    
    Args:
        df: DataFrame with OHLCV data
        period: Period for VWMA calculation
        
    Returns:
        IndicatorResult with VWMA values
    """
    try:
        if len(df) < period:
            return IndicatorResult(values={"vwma": []})
        
        # Make a copy to avoid modifying original
        _df = df.copy()
        
        # Calculate VWMA
        _df['price_volume'] = _df['close'] * _df['volume']
        _df['price_volume_sum'] = _df['price_volume'].rolling(window=period).sum()
        _df['volume_sum'] = _df['volume'].rolling(window=period).sum()
        
        # Handle potential division by zero
        _df.loc[_df['volume_sum'] == 0, 'volume_sum'] = 0.001  # Small value to prevent division by zero
        
        _df['vwma'] = _df['price_volume_sum'] / _df['volume_sum']
        
        # Convert to list and handle NaN values
        vwma_values = _df['vwma'].fillna(0).tolist()
        
        return IndicatorResult(values={"vwma": vwma_values})
        
    except Exception as e:
        # Return empty result on error
        return IndicatorResult(values={"vwma": []}, metadata={"error": str(e)})

def identify_support_resistance(df: pd.DataFrame, period: int = 10, threshold: float = 0.03) -> IndicatorResult:
    """
    Identify support and resistance levels from price action
    
    Args:
        df: DataFrame with OHLCV data
        period: Look-back period for identifying pivot points
        threshold: Price proximity threshold for combining levels
        
    Returns:
        IndicatorResult with support and resistance levels
    """
    try:
        if len(df) < period * 2:
            return IndicatorResult(values={"support": [], "resistance": []})
        
        # Make a copy to avoid modifying original
        _df = df.copy()
        
        # Identify pivot highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(period, len(_df) - period):
            # Check if this is a pivot high
            if all(_df.iloc[i]['high'] > _df.iloc[i-j]['high'] for j in range(1, period+1)) and \
               all(_df.iloc[i]['high'] > _df.iloc[i+j]['high'] for j in range(1, period+1)):
                pivot_highs.append(_df.iloc[i]['high'])
            
            # Check if this is a pivot low
            if all(_df.iloc[i]['low'] < _df.iloc[i-j]['low'] for j in range(1, period+1)) and \
               all(_df.iloc[i]['low'] < _df.iloc[i+j]['low'] for j in range(1, period+1)):
                pivot_lows.append(_df.iloc[i]['low'])
        
        # Combine nearby levels
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
        
        return IndicatorResult(values={"support": support_levels, "resistance": resistance_levels})
        
    except Exception as e:
        # Return empty result on error
        return IndicatorResult(values={"support": [], "resistance": []}, metadata={"error": str(e)})

def detect_candlestick_patterns(df: pd.DataFrame) -> IndicatorResult:
    """
    Detect common candlestick patterns
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        IndicatorResult with detected patterns
    """
    try:
        if len(df) < 3:  # Need at least 3 candles for most patterns
            return IndicatorResult(values={"patterns": {}})
        
        # Make a copy to avoid modifying original
        _df = df.copy()
        
        patterns = {}
        
        for i in range(len(_df)):
            if i < 1:  # Need at least 2 candles for patterns
                continue
            
            current_patterns = []
            
            # Calculate some common metrics
            body_size = abs(_df.iloc[i]['close'] - _df.iloc[i]['open'])
            range_size = _df.iloc[i]['high'] - _df.iloc[i]['low']
            prev_body_size = abs(_df.iloc[i-1]['close'] - _df.iloc[i-1]['open'])
            prev_range_size = _df.iloc[i-1]['high'] - _df.iloc[i-1]['low']
            
            # Is bullish candle
            is_bullish = _df.iloc[i]['close'] > _df.iloc[i]['open']
            prev_bullish = _df.iloc[i-1]['close'] > _df.iloc[i-1]['open']
            
            # Doji
            if body_size <= 0.05 * range_size:
                current_patterns.append("Doji")
            
            # Hammer / Hanging Man
            if (body_size > 0) and \
               (_df.iloc[i]['high'] - max(_df.iloc[i]['open'], _df.iloc[i]['close'])) <= 0.1 * body_size and \
               (min(_df.iloc[i]['open'], _df.iloc[i]['close']) - _df.iloc[i]['low']) >= 2 * body_size:
                if i > 1 and _df.iloc[i-2]['close'] < _df.iloc[i-1]['close']:  # Downtrend before
                    current_patterns.append("Hammer")
                else:
                    current_patterns.append("Hanging Man")
            
            # Bullish Engulfing
            if is_bullish and not prev_bullish and \
               _df.iloc[i]['open'] < _df.iloc[i-1]['close'] and _df.iloc[i]['close'] > _df.iloc[i-1]['open']:
                current_patterns.append("Bullish Engulfing")
            
            # Bearish Engulfing
            if not is_bullish and prev_bullish and \
               _df.iloc[i]['open'] > _df.iloc[i-1]['close'] and _df.iloc[i]['close'] < _df.iloc[i-1]['open']:
                current_patterns.append("Bearish Engulfing")
            
            # Morning Star (needs 3 candles)
            if i >= 2 and not prev_bullish and is_bullish and not _df.iloc[i-2]['close'] > _df.iloc[i-2]['open'] and body_size > 0:
                second_candle_small = abs(_df.iloc[i-1]['close'] - _df.iloc[i-1]['open']) < 0.3 * abs(_df.iloc[i-2]['close'] - _df.iloc[i-2]['open'])
                if second_candle_small and _df.iloc[i]['close'] > (_df.iloc[i-2]['open'] + _df.iloc[i-2]['close']) / 2:
                    current_patterns.append("Morning Star")
            
            # Evening Star (needs 3 candles)
            if i >= 2 and prev_bullish and not is_bullish and _df.iloc[i-2]['close'] > _df.iloc[i-2]['open'] and body_size > 0:
                second_candle_small = abs(_df.iloc[i-1]['close'] - _df.iloc[i-1]['open']) < 0.3 * abs(_df.iloc[i-2]['close'] - _df.iloc[i-2]['open'])
                if second_candle_small and _df.iloc[i]['close'] < (_df.iloc[i-2]['open'] + _df.iloc[i-2]['close']) / 2:
                    current_patterns.append("Evening Star")
            
            if current_patterns:
                patterns[i] = current_patterns
        
        return IndicatorResult(values={"patterns": patterns})
        
    except Exception as e:
        # Return empty result on error
        return IndicatorResult(values={"patterns": {}}, metadata={"error": str(e)})
