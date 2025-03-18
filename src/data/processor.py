"""
Data processing module for market data analysis.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Import needed for type hints only
from ..exchange.connector import MarketData, OrderbookData
from ..data.enhanced_indicators import calculate_atr, calculate_obv, calculate_stochastic, calculate_vwma, identify_support_resistance, detect_candlestick_patterns

logger = logging.getLogger(__name__)


class IndicatorResult(BaseModel):
    """Model for technical indicator calculation results."""
    name: str
    values: Dict[str, List[float]]
    metadata: Optional[Dict] = None


class OrderbookAnalysisResult(BaseModel):
    """Model for orderbook analysis results."""
    symbol: str
    timestamp: int
    buy_sell_ratio: float
    liquidity_distribution: Dict[str, float]
    support_levels: List[float]
    resistance_levels: List[float]
    imbalance_points: List[Dict[str, Any]]
    microstructure: Optional[Dict[str, Any]] = None
    bid_walls: Optional[List[Dict[str, float]]] = None
    ask_walls: Optional[List[Dict[str, float]]] = None
    bid_distribution: Optional[Dict[str, float]] = None
    ask_distribution: Optional[Dict[str, float]] = None
    bid_cumulative_distribution: Optional[Dict[str, float]] = None
    ask_cumulative_distribution: Optional[Dict[str, float]] = None
    metadata: Optional[Dict] = None


class DataProcessor:
    """
    Processes market data for analysis.
    Handles data cleaning, normalization, and technical indicator calculations.
    """
    
    @staticmethod
    def clean_ohlcv_data(data: List[MarketData]) -> pd.DataFrame:
        """
        Clean and preprocess OHLCV data.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            Cleaned pandas DataFrame
        """
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in data])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        
        # Handle missing values
        df = df.ffill()
        
        return df
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int = 20, column: str = 'close') -> IndicatorResult:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for the moving average
            column: Column to calculate SMA on
            
        Returns:
            IndicatorResult with SMA values
        """
        try:
            sma = data[column].rolling(window=period).mean()
            return IndicatorResult(
                name=f"SMA_{period}",
                values={"sma": sma.dropna().tolist()},
                metadata={"period": period, "column": column}
            )
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return IndicatorResult(
                name=f"SMA_{period}",
                values={"sma": []},
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int = 20, column: str = 'close') -> IndicatorResult:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for the moving average
            column: Column to calculate EMA on
            
        Returns:
            IndicatorResult with EMA values
        """
        try:
            ema = data[column].ewm(span=period, adjust=False).mean()
            return IndicatorResult(
                name=f"EMA_{period}",
                values={"ema": ema.dropna().tolist()},
                metadata={"period": period, "column": column}
            )
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return IndicatorResult(
                name=f"EMA_{period}",
                values={"ema": []},
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'close') -> IndicatorResult:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for RSI calculation
            column: Column to calculate RSI on
            
        Returns:
            IndicatorResult with RSI values
        """
        try:
            delta = data[column].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
            
            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))
            
            return IndicatorResult(
                name=f"RSI_{period}",
                values={"rsi": rsi.dropna().tolist()},
                metadata={"period": period, "column": column}
            )
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return IndicatorResult(
                name=f"RSI_{period}",
                values={"rsi": []},
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def calculate_macd(
        data: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9,
        column: str = 'close'
    ) -> IndicatorResult:
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            data: DataFrame with OHLCV data
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
            column: Column to calculate MACD on
            
        Returns:
            IndicatorResult with MACD values
        """
        try:
            # Calculate fast and slow EMAs
            fast_ema = data[column].ewm(span=fast_period, adjust=False).mean()
            slow_ema = data[column].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return IndicatorResult(
                name="MACD",
                values={
                    "macd_line": macd_line.dropna().tolist(),
                    "signal_line": signal_line.dropna().tolist(),
                    "histogram": histogram.dropna().tolist()
                },
                metadata={
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                    "column": column
                }
            )
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return IndicatorResult(
                name="MACD",
                values={
                    "macd_line": [],
                    "signal_line": [],
                    "histogram": []
                },
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def calculate_bollinger_bands(
        data: pd.DataFrame, 
        period: int = 20, 
        num_std: float = 2.0,
        column: str = 'close'
    ) -> IndicatorResult:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for moving average
            num_std: Number of standard deviations for bands
            column: Column to calculate Bollinger Bands on
            
        Returns:
            IndicatorResult with Bollinger Bands values
        """
        try:
            # Calculate middle band (SMA)
            middle_band = data[column].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = data[column].rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            
            return IndicatorResult(
                name="BollingerBands",
                values={
                    "middle_band": middle_band.dropna().tolist(),
                    "upper_band": upper_band.dropna().tolist(),
                    "lower_band": lower_band.dropna().tolist()
                },
                metadata={
                    "period": period,
                    "num_std": num_std,
                    "column": column
                }
            )
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return IndicatorResult(
                name="BollingerBands",
                values={
                    "middle_band": [],
                    "upper_band": [],
                    "lower_band": []
                },
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def analyze_orderbook(orderbook: OrderbookData) -> OrderbookAnalysisResult:
        """
        Analyze orderbook data to identify market structure and potential price action.
        
        Args:
            orderbook: OrderbookData object
            
        Returns:
            OrderbookAnalysisResult with analysis
        """
        try:
            # Extract bids and asks
            bids = np.array(orderbook.bids) if orderbook.bids else np.array([])
            asks = np.array(orderbook.asks) if orderbook.asks else np.array([])
            
            if len(bids) == 0 or len(asks) == 0:
                raise ValueError("Orderbook contains no bids or asks")
            
            # Calculate total bid and ask volumes
            bid_volume = sum(bid[1] for bid in bids) if bids.size > 0 else 0
            ask_volume = sum(ask[1] for ask in asks) if asks.size > 0 else 0
            
            # Calculate buy/sell ratio
            buy_sell_ratio = bid_volume / max(ask_volume, 0.0001)  # Avoid division by zero
            
            # Sort bids and asks by price
            sorted_bids = bids[bids[:, 0].argsort()][::-1]  # Descending
            sorted_asks = asks[asks[:, 0].argsort()]  # Ascending
            
            # Get the current price
            current_bid = sorted_bids[0][0] if len(sorted_bids) > 0 else 0
            current_ask = sorted_asks[0][0] if len(sorted_asks) > 0 else float('inf')
            current_price = (current_bid + current_ask) / 2
            
            # Calculate liquidity distribution as a percentage of total volume
            bid_quarters = np.array_split(sorted_bids, 4) if len(sorted_bids) >= 4 else [sorted_bids]
            ask_quarters = np.array_split(sorted_asks, 4) if len(sorted_asks) >= 4 else [sorted_asks]
            
            liquidity_distribution = {
                "lower": sum(bid[1] for bid in bid_quarters[0]) / max(bid_volume, 0.0001) if len(bid_quarters) >= 1 and bid_volume > 0 else 0,
                "lower_mid": sum(bid[1] for bid in bid_quarters[1]) / max(bid_volume, 0.0001) if len(bid_quarters) >= 2 and bid_volume > 0 else 0,
                "upper_mid": sum(ask[1] for ask in ask_quarters[0]) / max(ask_volume, 0.0001) if len(ask_quarters) >= 1 and ask_volume > 0 else 0,
                "upper": sum(ask[1] for ask in ask_quarters[1]) / max(ask_volume, 0.0001) if len(ask_quarters) >= 2 and ask_volume > 0 else 0
            }
            
            # Identify support levels (areas with large bid volume)
            bid_volume_threshold = np.percentile([bid[1] for bid in bids], 80) if len(bids) > 5 else 0
            support_levels = [bid[0] for bid in sorted_bids if bid[1] > bid_volume_threshold]
            
            # If we don't have enough levels from the threshold approach, take the top 3 bid levels
            if len(support_levels) < 2 and len(sorted_bids) >= 2:
                support_levels = [sorted_bids[i][0] for i in range(min(3, len(sorted_bids)))]
            
            # Identify resistance levels (areas with large ask volume)
            ask_volume_threshold = np.percentile([ask[1] for ask in asks], 80) if len(asks) > 5 else 0
            resistance_levels = [ask[0] for ask in sorted_asks if ask[1] > ask_volume_threshold]
            
            # If we don't have enough levels from the threshold approach, take the top 3 ask levels
            if len(resistance_levels) < 2 and len(sorted_asks) >= 2:
                resistance_levels = [sorted_asks[i][0] for i in range(min(3, len(sorted_asks)))]
            
            # Identify imbalance points (areas with large gaps in liquidity)
            bid_prices = [bid[0] for bid in sorted_bids]
            ask_prices = [ask[0] for ask in sorted_asks]
            
            # Check for gaps in the orderbook
            imbalance_points = []
            
            # Add bid imbalances
            for i in range(len(bid_prices) - 1):
                if i < len(bid_prices) - 1:
                    price_gap = abs(bid_prices[i] - bid_prices[i + 1])
                    avg_gap = (current_ask - current_bid) / 10  # Use 1/10 of the spread as reference
                    
                    if price_gap > avg_gap * 2:  # If gap is more than 2x the average gap
                        strength = min(price_gap / (avg_gap * 3), 1.0)  # Normalize strength between 0-1
                        imbalance_points.append({
                            "price": bid_prices[i],
                            "type": "bid",
                            "strength": float(strength)
                        })
            
            # Add ask imbalances
            for i in range(len(ask_prices) - 1):
                if i < len(ask_prices) - 1:
                    price_gap = abs(ask_prices[i + 1] - ask_prices[i])
                    avg_gap = (current_ask - current_bid) / 10  # Use 1/10 of the spread as reference
                    
                    if price_gap > avg_gap * 2:  # If gap is more than 2x the average gap
                        strength = min(price_gap / (avg_gap * 3), 1.0)  # Normalize strength between 0-1
                        imbalance_points.append({
                            "price": ask_prices[i],
                            "type": "ask",
                            "strength": float(strength)
                        })
            
            # Calculate bid/ask distributions
            bid_prices_array = np.array([bid[0] for bid in bids])
            ask_prices_array = np.array([ask[0] for ask in asks])
            bid_volumes_array = np.array([bid[1] for bid in bids])
            ask_volumes_array = np.array([ask[1] for ask in asks])
            
            # Create bid/ask distribution
            bid_distribution = {}
            ask_distribution = {}
            
            if len(bid_prices_array) > 0:
                bid_price_ranges = np.linspace(min(bid_prices_array), max(bid_prices_array), 10)
                for i in range(len(bid_price_ranges) - 1):
                    price_key = f"{bid_price_ranges[i]:.2f}-{bid_price_ranges[i+1]:.2f}"
                    mask = (bid_prices_array >= bid_price_ranges[i]) & (bid_prices_array < bid_price_ranges[i+1])
                    bid_distribution[price_key] = float(sum(bid_volumes_array[mask]) / max(sum(bid_volumes_array), 0.0001))
            
            if len(ask_prices_array) > 0:
                ask_price_ranges = np.linspace(min(ask_prices_array), max(ask_prices_array), 10)
                for i in range(len(ask_price_ranges) - 1):
                    price_key = f"{ask_price_ranges[i]:.2f}-{ask_price_ranges[i+1]:.2f}"
                    mask = (ask_prices_array >= ask_price_ranges[i]) & (ask_prices_array < ask_price_ranges[i+1])
                    ask_distribution[price_key] = float(sum(ask_volumes_array[mask]) / max(sum(ask_volumes_array), 0.0001))
            
            # Find bid/ask walls (large orders that might act as barriers)
            bid_wall_threshold = np.percentile(bid_volumes_array, 90) if len(bid_volumes_array) > 5 else 0
            ask_wall_threshold = np.percentile(ask_volumes_array, 90) if len(ask_volumes_array) > 5 else 0
            
            bid_walls = [{"price": float(bid[0]), "volume": float(bid[1])} 
                        for bid in bids if bid[1] > bid_wall_threshold]
            ask_walls = [{"price": float(ask[0]), "volume": float(ask[1])} 
                        for ask in asks if ask[1] > ask_wall_threshold]
            
            # Return the complete analysis
            return OrderbookAnalysisResult(
                symbol=orderbook.symbol,
                timestamp=orderbook.timestamp,
                buy_sell_ratio=float(buy_sell_ratio),
                liquidity_distribution=liquidity_distribution,
                support_levels=[float(level) for level in support_levels],
                resistance_levels=[float(level) for level in resistance_levels],
                imbalance_points=imbalance_points,
                bid_walls=bid_walls,
                ask_walls=ask_walls,
                bid_distribution=bid_distribution,
                ask_distribution=ask_distribution,
                microstructure={
                    "spread": float(current_ask - current_bid),
                    "mid_price": float(current_price),
                    "depth_imbalance": float(buy_sell_ratio),
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing orderbook: {e}")
            # Return a minimal result with error information
            return OrderbookAnalysisResult(
                symbol=orderbook.symbol if hasattr(orderbook, 'symbol') else "unknown",
                timestamp=orderbook.timestamp if hasattr(orderbook, 'timestamp') else 0,
                buy_sell_ratio=1.0,
                liquidity_distribution={"lower": 0.25, "lower_mid": 0.25, "upper_mid": 0.25, "upper": 0.25},
                support_levels=[],
                resistance_levels=[],
                imbalance_points=[],
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def _calculate_depth_distribution(orders: np.ndarray, bins: int = 5) -> Dict[str, float]:
        """
        Calculate the distribution of volume across price levels.
        
        Args:
            orders: Array of [price, amount] pairs
            bins: Number of bins to divide the price range into
            
        Returns:
            Dictionary mapping price ranges to volume
        """
        if len(orders) == 0:
            return {}
        
        min_price = orders[-1][0]
        max_price = orders[0][0]
        price_range = max_price - min_price
        
        if price_range <= 0:
            return {f"{min_price}": sum(orders[:, 1])}
        
        bin_size = price_range / bins
        distribution = {}
        
        for i in range(bins):
            bin_start = min_price + i * bin_size
            bin_end = bin_start + bin_size if i < bins - 1 else max_price + 0.0000001  # Include the last price
            bin_volume = sum(amount for price, amount in orders if bin_start <= price < bin_end)
            distribution[f"{bin_start:.8f}-{bin_end:.8f}"] = bin_volume
        
        return distribution

    @staticmethod
    def _calculate_cumulative_volume_distribution(orders: np.ndarray, price_levels: int = 10) -> Dict[str, float]:
        """
        Calculate cumulative volume at different price levels from the best price.
        
        Args:
            orders: Array of [price, amount] pairs
            price_levels: Number of price levels to calculate
            
        Returns:
            Dictionary mapping price range descriptions to cumulative volume
        """
        if len(orders) == 0:
            return {}
        
        cumulative_distribution = {}
        cumulative_volume = 0
        
        # Use price percentage steps for more meaningful distribution
        best_price = orders[0][0]
        
        # Calculate percentage steps based on typical crypto volatility
        if len(orders) > 1:
            worst_price = orders[-1][0]
            price_range_percent = abs((worst_price - best_price) / best_price) * 100
            step_size = max(0.1, min(1.0, price_range_percent / price_levels))  # Between 0.1% and 1% steps
        else:
            step_size = 0.5  # Default 0.5% steps
        
        # Create distribution with percentage steps from best price
        for i in range(price_levels):
            if len(orders) == 0:
                break
                
            # Calculate price threshold for this level (percentage from best price)
            if best_price > 0:
                if orders[0][0] > orders[-1][0]:  # Bids (buy orders) - descending prices
                    threshold = best_price * (1 - (i * step_size / 100))
                    level_orders = orders[orders[:, 0] >= threshold]
                    level_name = f"Within -{i*step_size:.2f}%"
                else:  # Asks (sell orders) - ascending prices
                    threshold = best_price * (1 + (i * step_size / 100))
                    level_orders = orders[orders[:, 0] <= threshold]
                    level_name = f"Within +{i*step_size:.2f}%"
            else:
                level_orders = np.array([])
                level_name = f"Level {i+1}"
            
            # Sum the volume at this level
            level_volume = np.sum(level_orders[:, 1]) if len(level_orders) > 0 else 0
            cumulative_volume += level_volume
            cumulative_distribution[level_name] = cumulative_volume
        
        return cumulative_distribution

    @staticmethod
    def _identify_liquidity_walls(orders: np.ndarray, threshold_factor: float = 3.0) -> List[Dict[str, float]]:
        """
        Identify price levels with significantly higher liquidity than average.
        
        Args:
            orders: Array of [price, amount] pairs
            threshold_factor: Factor above average to identify as a wall
            
        Returns:
            List of dictionaries with price and volume information for each wall
        """
        if len(orders) == 0:
            return []
        
        volumes = orders[:, 1]
        avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
        threshold = avg_volume * threshold_factor
        
        walls = []
        for price, amount in orders:
            if amount > threshold:
                walls.append({"price": price, "volume": amount})
        
        return walls

    @staticmethod
    def _detect_microstructure_patterns(bids: np.ndarray, asks: np.ndarray) -> Dict[str, Any]:
        """
        Detect market microstructure patterns in order book data.
        
        Args:
            bids: Array of [price, amount] pairs for bids
            asks: Array of [price, amount] pairs for asks
            
        Returns:
            Dictionary with microstructure analysis results
        """
        if len(bids) == 0 or len(asks) == 0:
            return {}
        
        patterns = {}
        
        # Get some basic metrics for pattern detection
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Calculate volume-weighted average prices (VWAP) for both sides
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        
        bid_vwap = np.sum(bids[:, 0] * bids[:, 1]) / bid_volume if bid_volume > 0 else best_bid
        ask_vwap = np.sum(asks[:, 0] * asks[:, 1]) / ask_volume if ask_volume > 0 else best_ask
        
        # Calculate volume profile skew (higher values mean more volume toward top of book)
        bid_skew = 0
        ask_skew = 0
        
        if len(bids) > 0:
            top_5_bids_volume = np.sum(bids[:min(5, len(bids)), 1])
            bid_skew = top_5_bids_volume / bid_volume if bid_volume > 0 else 0
        
        if len(asks) > 0:
            top_5_asks_volume = np.sum(asks[:min(5, len(asks)), 1])
            ask_skew = top_5_asks_volume / ask_volume if ask_volume > 0 else 0
        
        # Detect specific patterns
        
        # 1. Iceberg detection (hidden liquidity) - look for repeated same-sized orders
        bid_sizes = bids[:min(10, len(bids)), 1]
        ask_sizes = asks[:min(10, len(asks)), 1]
        
        # Count repeated sizes (potential icebergs)
        bid_size_counts = {}
        ask_size_counts = {}
        
        for size in bid_sizes:
            rounded_size = round(size, 6)  # Round to handle floating point variations
            bid_size_counts[rounded_size] = bid_size_counts.get(rounded_size, 0) + 1
        
        for size in ask_sizes:
            rounded_size = round(size, 6)
            ask_size_counts[rounded_size] = ask_size_counts.get(rounded_size, 0) + 1
        
        # Look for sizes that appear multiple times
        potential_bid_icebergs = [(size, count) for size, count in bid_size_counts.items() if count >= 3 and size > 0]
        potential_ask_icebergs = [(size, count) for size, count in ask_size_counts.items() if count >= 3 and size > 0]
        
        # 2. Spoofing detection (large orders away from mid price that may be canceled)
        far_bid_volume = np.sum([amount for price, amount in bids if (mid_price - price) / mid_price > 0.02])
        far_ask_volume = np.sum([amount for price, amount in asks if (price - mid_price) / mid_price > 0.02])
        
        spoofing_threshold = 5  # 5x normal volume would be suspicious
        avg_bid_size = bid_volume / len(bids) if len(bids) > 0 else 0
        avg_ask_size = ask_volume / len(asks) if len(asks) > 0 else 0
        
        large_far_bids = [(price, amount) for price, amount in bids 
                         if amount > avg_bid_size * spoofing_threshold and (mid_price - price) / mid_price > 0.01]
        large_far_asks = [(price, amount) for price, amount in asks 
                         if amount > avg_ask_size * spoofing_threshold and (price - mid_price) / mid_price > 0.01]
        
        # 3. Order book pressure and potential price direction
        # Stronger pressure = more likely price movement in that direction
        bid_pressure = (bid_volume * bid_skew) / (ask_volume * ask_skew) if (ask_volume * ask_skew) > 0 else float('inf')
        pressure_direction = "buy" if bid_pressure > 1.2 else "sell" if bid_pressure < 0.8 else "neutral"
        
        # Build pattern dictionary
        patterns = {
            "bid_vwap": bid_vwap,
            "ask_vwap": ask_vwap,
            "bid_skew": bid_skew,
            "ask_skew": ask_skew,
            "bid_pressure": bid_pressure,
            "pressure_direction": pressure_direction,
            "potential_bid_icebergs": potential_bid_icebergs,
            "potential_ask_icebergs": potential_ask_icebergs,
            "large_far_bids": large_far_bids,
            "large_far_asks": large_far_asks,
        }
        
        # Add some interpretations
        interpretations = []
        
        if bid_pressure > 1.5:
            interpretations.append("Strong buying pressure detected, potential upward price movement")
        elif bid_pressure < 0.5:
            interpretations.append("Strong selling pressure detected, potential downward price movement")
        
        if potential_bid_icebergs:
            interpretations.append(f"Potential iceberg buy orders detected at sizes: {[size for size, _ in potential_bid_icebergs]}")
        if potential_ask_icebergs:
            interpretations.append(f"Potential iceberg sell orders detected at sizes: {[size for size, _ in potential_ask_icebergs]}")
        
        if large_far_bids:
            interpretations.append(f"Possible spoofing detected: {len(large_far_bids)} unusually large buy orders far from mid price")
        if large_far_asks:
            interpretations.append(f"Possible spoofing detected: {len(large_far_asks)} unusually large sell orders far from mid price")
        
        if ask_skew > 0.7:
            interpretations.append("Sell volume heavily concentrated at top of book, potential resistance")
        if bid_skew > 0.7:
            interpretations.append("Buy volume heavily concentrated at top of book, potential support")
            
        patterns["interpretations"] = interpretations
        
        return patterns

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for ATR calculation
            
        Returns:
            IndicatorResult with ATR values
        """
        try:
            result = calculate_atr(data, period)
            return IndicatorResult(
                name=f"ATR_{period}",
                values={"atr": result.values["atr"]},
                metadata={"period": period}
            )
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return IndicatorResult(
                name=f"ATR_{period}",
                values={"atr": []},
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate On Balance Volume (OBV).
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            IndicatorResult with OBV values
        """
        try:
            result = calculate_obv(data)
            return IndicatorResult(
                name="OBV",
                values={"obv": result.values["obv"]},
                metadata={}
            )
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return IndicatorResult(
                name="OBV",
                values={"obv": []},
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorResult:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Args:
            data: DataFrame with OHLCV data
            k_period: Period for %K line
            d_period: Period for %D line (SMA of %K)
            
        Returns:
            IndicatorResult with Stochastic values
        """
        try:
            result = calculate_stochastic(data, k_period, d_period)
            return IndicatorResult(
                name="Stochastic",
                values={
                    "k_line": result.values["k_line"],
                    "d_line": result.values["d_line"]
                },
                metadata={
                    "k_period": k_period,
                    "d_period": d_period
                }
            )
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return IndicatorResult(
                name="Stochastic",
                values={
                    "k_line": [],
                    "d_line": []
                },
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def calculate_vwma(data: pd.DataFrame, period: int = 14, price_column: str = 'close', volume_column: str = 'volume') -> IndicatorResult:
        """
        Calculate Volume Weighted Moving Average (VWMA).
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for the moving average
            price_column: Column to use for price data
            volume_column: Column to use for volume data
            
        Returns:
            IndicatorResult with VWMA values
        """
        try:
            result = calculate_vwma(data, period)
            return IndicatorResult(
                name=f"VWMA_{period}",
                values={"vwma": result.values["vwma"]},
                metadata={
                    "period": period,
                    "price_column": price_column,
                    "volume_column": volume_column
                }
            )
        except Exception as e:
            logger.error(f"Error calculating VWMA: {e}")
            return IndicatorResult(
                name=f"VWMA_{period}",
                values={"vwma": []},
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def identify_support_resistance(data: pd.DataFrame, period: int = 10, threshold: float = 0.03) -> IndicatorResult:
        """
        Identify support and resistance levels from price action.
        
        Args:
            data: DataFrame with OHLCV data
            period: Look-back period for identifying pivot points
            threshold: Price proximity threshold for combining levels
            
        Returns:
            IndicatorResult with support and resistance levels
        """
        try:
            result = identify_support_resistance(data, period, threshold)
            return IndicatorResult(
                name="Support_Resistance",
                values={
                    "support": result.values["support"],
                    "resistance": result.values["resistance"]
                },
                metadata={
                    "period": period,
                    "threshold": threshold
                }
            )
        except Exception as e:
            logger.error(f"Error identifying support and resistance: {e}")
            return IndicatorResult(
                name="Support_Resistance",
                values={
                    "support": [],
                    "resistance": []
                },
                metadata={"error": str(e)}
            )
    
    @staticmethod
    def detect_candlestick_patterns(data: pd.DataFrame) -> IndicatorResult:
        """
        Detect common candlestick patterns in the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            IndicatorResult with detected patterns
        """
        try:
            result = detect_candlestick_patterns(data)
            # Convert the patterns dict to a format compatible with IndicatorResult
            detected_patterns = {}
            for idx, patterns in result.values["patterns"].items():
                detected_patterns[str(idx)] = patterns
                
            return IndicatorResult(
                name="Candlestick_Patterns",
                values={"patterns": detected_patterns},
                metadata={}
            )
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return IndicatorResult(
                name="Candlestick_Patterns",
                values={"patterns": {}},
                metadata={"error": str(e)}
            )

    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """
        Calculate a comprehensive set of technical indicators for the provided data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping indicator names to their calculation results
        """
        indicators = {}
        
        # Calculate moving averages
        indicators["sma_20"] = DataProcessor.calculate_sma(data, period=20)
        indicators["sma_50"] = DataProcessor.calculate_sma(data, period=50)
        indicators["sma_200"] = DataProcessor.calculate_sma(data, period=200)
        indicators["ema_12"] = DataProcessor.calculate_ema(data, period=12)
        indicators["ema_26"] = DataProcessor.calculate_ema(data, period=26)
        
        # Calculate momentum indicators
        indicators["rsi_14"] = DataProcessor.calculate_rsi(data, period=14)
        indicators["macd"] = DataProcessor.calculate_macd(data)
        
        # Calculate volatility indicators
        indicators["bollinger_bands"] = DataProcessor.calculate_bollinger_bands(data)
        indicators["atr_14"] = DataProcessor.calculate_atr(data, period=14)
        
        # Calculate volume indicators
        indicators["obv"] = DataProcessor.calculate_obv(data)
        indicators["vwma_20"] = DataProcessor.calculate_vwma(data, period=20)
        
        # Calculate oscillators
        indicators["stochastic"] = DataProcessor.calculate_stochastic(data)
        
        # Calculate support/resistance and patterns
        indicators["support_resistance"] = DataProcessor.identify_support_resistance(data)
        indicators["candlestick_patterns"] = DataProcessor.detect_candlestick_patterns(data)
        
        return indicators

    @staticmethod
    def identify_support_resistance_levels(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """
        Identify technical support and resistance levels from historical price data.
        
        Args:
            data: DataFrame with OHLCV data
            window: Window size for detecting pivot points
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            if data.empty:
                return {"support": [], "resistance": []}
            
            # Get high and low prices
            highs = data['high'].values
            lows = data['low'].values
            
            # Initialize levels
            support_levels = []
            resistance_levels = []
            
            # Find pivot points (local minima and maxima)
            for i in range(window, len(data) - window):
                # Check for local minima (support)
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    support_levels.append(float(lows[i]))
                
                # Check for local maxima (resistance)
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(float(highs[i]))
            
            # Cluster close levels to avoid duplication
            if support_levels:
                support_levels = DataProcessor._cluster_price_levels(support_levels)
            
            if resistance_levels:
                resistance_levels = DataProcessor._cluster_price_levels(resistance_levels)
            
            # If not enough levels found, use recent highs and lows as fallback
            if len(support_levels) < 2:
                # Find lowest points in different windows
                support_levels = [
                    float(min(lows[-20:])), 
                    float(min(lows[-50:]) if len(lows) >= 50 else min(lows))
                ]
            
            if len(resistance_levels) < 2:
                # Find highest points in different windows
                resistance_levels = [
                    float(max(highs[-20:])), 
                    float(max(highs[-50:]) if len(highs) >= 50 else max(highs))
                ]
            
            return {
                "support": sorted(support_levels),
                "resistance": sorted(resistance_levels)
            }
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance levels: {e}")
            return {"support": [], "resistance": []}

    @staticmethod
    def _cluster_price_levels(levels: List[float], threshold_pct: float = 0.01) -> List[float]:
        """
        Cluster price levels that are within a certain percentage of each other.
        
        Args:
            levels: List of price levels
            threshold_pct: Percentage threshold for clustering
            
        Returns:
            List of clustered price levels
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Initialize clusters
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        # Cluster close levels
        for level in sorted_levels[1:]:
            if level / current_cluster[0] - 1 <= threshold_pct:
                # Level is within threshold of first element in cluster
                current_cluster.append(level)
            else:
                # Level is outside threshold, start a new cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
