"""
Technical analysis agent for analyzing price data and indicators.
"""
import logging
from typing import Dict, List, Any, Optional
import json
import asyncio

from agents import Agent, Tool
from pydantic import BaseModel, Field

from .base_agent import BaseMarketAgent
from ..exchange.connector import ExchangeConnector, MarketData
from ..data.processor import DataProcessor

logger = logging.getLogger(__name__)


class TechnicalAnalysisRequest(BaseModel):
    """Model for technical analysis requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BERA/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")
    timeframe: str = Field("1h", description="Timeframe for analysis (e.g., '1m', '5m', '15m', '1h', '4h', '1d')")
    limit: Optional[int] = Field(100, description="Number of candles to analyze")
    indicators: Optional[List[str]] = Field(None, description="List of indicators to calculate (e.g., ['sma', 'rsi', 'macd'])")


class TechnicalAnalysisAgent(BaseMarketAgent):
    """
    Specialized agent for technical analysis of price data.
    Focuses on indicators, chart patterns, and identifying potential market moves.
    """
    
    def __init__(self):
        """Initialize the technical analysis agent."""
        super().__init__(
            name="Technical Analysis Agent",
            description="A specialized agent for performing technical analysis using indicators and chart patterns."
        )
        
        # Initialize tools list
        self.tools = []
        
        # Register tools
        self._register_tools()
    
    def _get_instructions(self) -> str:
        """
        Get specific instructions for the technical analysis agent.
        
        Returns:
            String with agent instructions
        """
        return """
        You are the Technical Analysis Agent, a specialized agent for analyzing cryptocurrency price data using
        technical analysis methods, indicators, and chart patterns.
        
        Your primary responsibility is to analyze price data to provide insights about:
        
        1. **Trend Analysis**: Identify market trends (uptrend, downtrend, consolidation)
        2. **Momentum Assessment**: Evaluate price momentum using indicators like RSI, MACD, and Stochastic
        3. **Support and Resistance**: Identify key price levels based on historical price action
        4. **Chart Patterns**: Recognize patterns like triangles, head and shoulders, double tops/bottoms
        5. **Moving Averages**: Analyze simple and exponential moving averages and their crossovers
        6. **Volatility Analysis**: Assess market volatility using Bollinger Bands, ATR, and other methods
        
        ### Your Capabilities:
        
        - Analyze historical price data from various exchanges and timeframes
        - Calculate and interpret common technical indicators
        - Identify chart patterns and potential breakout points
        - Compare current market conditions with historical analogues
        - Provide probabilistic assessments of potential price movements
        
        ### Communication Guidelines:
        
        1. Present numerical data with appropriate precision
        2. Use markdown formatting for clarity
        3. Explain the significance of indicators and patterns
        4. Highlight confluences where multiple indicators suggest the same outcome
        5. Be specific about timeframes and market conditions
        6. Always include potential counterarguments to your analysis
        
        Use your tools to fetch and analyze price data from various exchanges and timeframes.
        Always specify which trading pair, timeframe, and exchange you're analyzing.
        """
    
    def _register_tools(self) -> None:
        """Register tools for the technical analysis agent."""
        # Tool for performing technical analysis
        self.add_tool(
            Tool(
                name="perform_technical_analysis",
                description="Perform technical analysis on price data for a specific trading pair",
                function=self._perform_technical_analysis,
                parameters=[
                    TechnicalAnalysisRequest
                ]
            )
        )
    
    def add_tool(self, tool):
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool instance to add
        """
        self.tools.append(tool)
    
    async def _perform_technical_analysis(self, params: TechnicalAnalysisRequest) -> Dict[str, Any]:
        """
        Perform technical analysis on price data.
        
        Args:
            params: Parameters for the analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Determine which indicators to calculate
            indicators = params.indicators or [
                "sma", "ema", "rsi", "macd", "bollinger", 
                "atr", "obv", "stochastic", "vwma", 
                "support_resistance", "candlestick_patterns"
            ]
            
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                # Fetch OHLCV data
                ohlcv_data = await connector.fetch_ohlcv(
                    symbol=params.symbol,
                    timeframe=params.timeframe,
                    limit=params.limit
                )
                
                # Convert to DataFrame for analysis
                df = connector.ohlcv_to_dataframe(ohlcv_data)
                
                # Calculate requested indicators
                indicator_results = {}
                
                # Helper function for safely accessing indicator values
                def safe_get_values(indicator_result, key, limit=20):
                    try:
                        if indicator_result and hasattr(indicator_result, 'values') and key in indicator_result.values:
                            values = indicator_result.values[key]
                            if values and len(values) > 0:
                                return values[-limit:] if len(values) > limit else values
                        return []
                    except Exception as e:
                        logger.error(f"Error accessing {key} from indicator result: {e}")
                        return []
                
                if "sma" in indicators:
                    try:
                        sma_20 = DataProcessor.calculate_sma(df, period=20)
                        sma_50 = DataProcessor.calculate_sma(df, period=50)
                        sma_200 = DataProcessor.calculate_sma(df, period=200)
                        indicator_results["sma"] = {
                            "sma_20": safe_get_values(sma_20, "sma"),
                            "sma_50": safe_get_values(sma_50, "sma"),
                            "sma_200": safe_get_values(sma_200, "sma")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating SMA: {e}")
                        indicator_results["sma"] = {"error": str(e)}
                
                if "ema" in indicators:
                    try:
                        ema_12 = DataProcessor.calculate_ema(df, period=12)
                        ema_26 = DataProcessor.calculate_ema(df, period=26)
                        ema_50 = DataProcessor.calculate_ema(df, period=50)
                        indicator_results["ema"] = {
                            "ema_12": safe_get_values(ema_12, "ema"),
                            "ema_26": safe_get_values(ema_26, "ema"),
                            "ema_50": safe_get_values(ema_50, "ema")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating EMA: {e}")
                        indicator_results["ema"] = {"error": str(e)}
                
                if "rsi" in indicators:
                    try:
                        rsi_14 = DataProcessor.calculate_rsi(df, period=14)
                        indicator_results["rsi"] = {
                            "rsi_14": safe_get_values(rsi_14, "rsi")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating RSI: {e}")
                        indicator_results["rsi"] = {"error": str(e)}
                
                if "macd" in indicators:
                    try:
                        macd = DataProcessor.calculate_macd(df)
                        indicator_results["macd"] = {
                            "macd_line": safe_get_values(macd, "macd_line"),
                            "signal_line": safe_get_values(macd, "signal_line"),
                            "histogram": safe_get_values(macd, "histogram")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating MACD: {e}")
                        indicator_results["macd"] = {"error": str(e)}
                
                if "bollinger" in indicators:
                    try:
                        bollinger = DataProcessor.calculate_bollinger_bands(df)
                        indicator_results["bollinger"] = {
                            "middle_band": safe_get_values(bollinger, "middle_band"),
                            "upper_band": safe_get_values(bollinger, "upper_band"),
                            "lower_band": safe_get_values(bollinger, "lower_band")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating Bollinger Bands: {e}")
                        indicator_results["bollinger"] = {"error": str(e)}
                
                # Add new enhanced indicators
                if "atr" in indicators:
                    try:
                        atr = DataProcessor.calculate_atr(df, period=14)
                        indicator_results["atr"] = {
                            "atr_14": safe_get_values(atr, "atr")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating ATR: {e}")
                        indicator_results["atr"] = {"error": str(e)}
                
                if "obv" in indicators:
                    try:
                        obv = DataProcessor.calculate_obv(df)
                        indicator_results["obv"] = {
                            "obv": safe_get_values(obv, "obv")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating OBV: {e}")
                        indicator_results["obv"] = {"error": str(e)}
                
                if "stochastic" in indicators:
                    try:
                        stoch = DataProcessor.calculate_stochastic(df)
                        indicator_results["stochastic"] = {
                            "k_line": safe_get_values(stoch, "k_line"),
                            "d_line": safe_get_values(stoch, "d_line")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating Stochastic: {e}")
                        indicator_results["stochastic"] = {"error": str(e)}
                
                if "vwma" in indicators:
                    try:
                        vwma = DataProcessor.calculate_vwma(df, period=20)
                        indicator_results["vwma"] = {
                            "vwma_20": safe_get_values(vwma, "vwma")
                        }
                    except Exception as e:
                        logger.error(f"Error calculating VWMA: {e}")
                        indicator_results["vwma"] = {"error": str(e)}
                
                if "support_resistance" in indicators:
                    try:
                        sr = DataProcessor.identify_support_resistance(df)
                        indicator_results["support_resistance"] = {
                            "support": safe_get_values(sr, "support", limit=10),
                            "resistance": safe_get_values(sr, "resistance", limit=10)
                        }
                    except Exception as e:
                        logger.error(f"Error calculating Support/Resistance: {e}")
                        indicator_results["support_resistance"] = {"error": str(e)}
                
                if "candlestick_patterns" in indicators:
                    try:
                        patterns = DataProcessor.detect_candlestick_patterns(df)
                        # We'll limit to most recent 20 candles for patterns
                        recent_patterns = {}
                        
                        if patterns and hasattr(patterns, 'values') and "patterns" in patterns.values:
                            pattern_dict = patterns.values["patterns"]
                            for idx, pattern_list in pattern_dict.items():
                                if int(idx) >= len(df) - 20:
                                    recent_patterns[idx] = pattern_list
                        
                        indicator_results["candlestick_patterns"] = recent_patterns
                    except Exception as e:
                        logger.error(f"Error detecting candlestick patterns: {e}")
                        indicator_results["candlestick_patterns"] = {"error": str(e)}
                
                # Get recent price data for context
                try:
                    recent_prices = df['close'].tolist()[-20:] if not df.empty else []
                    recent_highs = df['high'].tolist()[-20:] if not df.empty else []
                    recent_lows = df['low'].tolist()[-20:] if not df.empty else []
                    recent_timestamps = [int(ts) for ts in df.index.astype(int).tolist()[-20:] // 10**6] if not df.empty else []
                except Exception as e:
                    logger.error(f"Error extracting recent price data: {e}")
                    recent_prices = []
                    recent_highs = []
                    recent_lows = []
                    recent_timestamps = []
                
                # Calculate enhanced trend detection with new indicators
                try:
                    trend = self._detect_trend(df)
                except Exception as e:
                    logger.error(f"Error detecting trend: {e}")
                    trend = {"direction": "unknown", "error": str(e)}
                
                # Result object
                result = {
                    "symbol": params.symbol,
                    "exchange": connector.exchange_id,
                    "timeframe": params.timeframe,
                    "indicators": indicator_results,
                    "recent_data": {
                        "prices": recent_prices,
                        "highs": recent_highs,
                        "lows": recent_lows,
                        "timestamps": recent_timestamps
                    },
                    "trend": trend,
                    "current_price": recent_prices[-1] if recent_prices else None,
                }
                
                # Add interpretation
                result["interpretation"] = self._interpret_technical_analysis(result)
                
                return result
                
        except Exception as e:
            logger.error(f"Error performing technical analysis: {e}")
            return {
                "error": str(e),
                "symbol": params.symbol,
                "exchange": params.exchange,
                "timeframe": params.timeframe
            }
    
    def _detect_trend(self, df: Any) -> Dict[str, Any]:
        """
        Detect market trend based on technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with trend information
        """
        try:
            # Get current price and recent price history
            current_price = df['close'].iloc[-1]
            price_5_ago = df['close'].iloc[-6] if len(df) > 5 else df['close'].iloc[0]
            price_20_ago = df['close'].iloc[-21] if len(df) > 20 else df['close'].iloc[0]
            
            # Calculate short-term price change
            short_term_change = (current_price - price_5_ago) / price_5_ago * 100
            
            # Calculate medium-term price change
            medium_term_change = (current_price - price_20_ago) / price_20_ago * 100
            
            # Calculate SMA 20 and 50
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # Calculate EMA 20
            ema_20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1] if len(df) >= 20 else None
            
            # Calculate RSI
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = delta.clip(lower=0).rolling(window=14).mean()
                loss = -delta.clip(upper=0).rolling(window=14).mean()
                rs = gain / loss.replace(0, 0.001)  # Avoid division by zero
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
            else:
                current_rsi = 50  # Neutral if not enough data
            
            # Enhanced trend detection with additional indicators
            # ATR for volatility
            atr_result = DataProcessor.calculate_atr(df)
            atr_value = atr_result.values["atr"][-1] if atr_result.values["atr"] else 0
            
            # Stochastic for momentum
            stoch_result = DataProcessor.calculate_stochastic(df)
            k_value = stoch_result.values["k_line"][-1] if stoch_result.values["k_line"] else 50
            d_value = stoch_result.values["d_line"][-1] if stoch_result.values["d_line"] else 50
            
            # Determine trend direction
            trend = "sideways"  # Default
            
            if sma_20 is not None and sma_50 is not None:
                if current_price > sma_20 > sma_50 and short_term_change > 0.5:
                    trend = "strong_uptrend"
                elif current_price > sma_20 and current_price > sma_50:
                    trend = "uptrend"
                elif current_price < sma_20 < sma_50 and short_term_change < -0.5:
                    trend = "strong_downtrend"
                elif current_price < sma_20 and current_price < sma_50:
                    trend = "downtrend"
                elif abs(short_term_change) < 0.5 and abs(medium_term_change) < 2:
                    trend = "sideways"
            
            # Enhance trend analysis with new indicators
            trend_strength = "medium"  # Default
            
            # Determine trend strength using ATR and price change
            if atr_value > 0:
                volatility_ratio = abs(short_term_change) / atr_value
                if volatility_ratio > 2:
                    trend_strength = "strong"
                elif volatility_ratio < 0.5:
                    trend_strength = "weak"
            
            # Determine momentum from stochastic
            momentum = "neutral"
            if k_value > 80:
                momentum = "overbought"
            elif k_value < 20:
                momentum = "oversold"
            
            # RSI confirmation
            rsi_signal = "neutral"
            if current_rsi > 70:
                rsi_signal = "overbought"
            elif current_rsi < 30:
                rsi_signal = "oversold"
            
            return {
                "direction": trend,
                "strength": trend_strength,
                "short_term_change_pct": round(short_term_change, 2),
                "medium_term_change_pct": round(medium_term_change, 2),
                "current_price": round(current_price, 4),
                "momentum": momentum,
                "rsi_signal": rsi_signal,
                "above_sma_20": current_price > sma_20 if sma_20 is not None else None,
                "above_sma_50": current_price > sma_50 if sma_50 is not None else None,
                "sma_20_50_cross": (sma_20 > sma_50) if (sma_20 is not None and sma_50 is not None) else None,
                "volatility": {
                    "atr": round(atr_value, 4) if atr_value else None,
                },
                "momentum_indicators": {
                    "stochastic_k": round(k_value, 2) if k_value else None,
                    "stochastic_d": round(d_value, 2) if d_value else None,
                    "rsi": round(current_rsi, 2) if current_rsi else None
                }
            }
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            return {
                "direction": "unknown",
                "error": str(e)
            }
    
    def _interpret_technical_analysis(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret technical analysis results.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Dictionary with interpretations
        """
        interpretations = {}
        
        # Interpret trend
        trend = analysis.get("trend", {})
        if trend:
            direction = trend.get("direction", "unknown")
            strength = trend.get("strength", "unknown")
            
            if direction == "uptrend" and strength == "strong":
                interpretations["trend"] = "Strong uptrend. The price is moving higher with conviction."
            elif direction == "uptrend" and strength == "moderate":
                interpretations["trend"] = "Moderate uptrend. The price is moving higher but showing some hesitation."
            elif direction == "downtrend" and strength == "strong":
                interpretations["trend"] = "Strong downtrend. The price is moving lower with conviction."
            elif direction == "downtrend" and strength == "moderate":
                interpretations["trend"] = "Moderate downtrend. The price is moving lower but showing some hesitation."
            elif direction == "sideways":
                interpretations["trend"] = "Sideways movement. The price is consolidating without a clear direction."
            else:
                interpretations["trend"] = "Unclear trend direction. Mixed signals in the price action."
        
        # Interpret RSI if available
        if "rsi" in analysis.get("indicators", {}):
            rsi_values = analysis["indicators"]["rsi"].get("rsi_14", [])
            if rsi_values:
                latest_rsi = rsi_values[-1]
                
                if latest_rsi > 70:
                    interpretations["rsi"] = f"Overbought conditions (RSI: {round(latest_rsi, 2)}). The market may be due for a correction."
                elif latest_rsi < 30:
                    interpretations["rsi"] = f"Oversold conditions (RSI: {round(latest_rsi, 2)}). The market may be due for a bounce."
                elif 40 <= latest_rsi <= 60:
                    interpretations["rsi"] = f"Neutral momentum (RSI: {round(latest_rsi, 2)}). No extreme conditions detected."
                elif 30 <= latest_rsi < 40:
                    interpretations["rsi"] = f"Recovering from oversold conditions (RSI: {round(latest_rsi, 2)})."
                elif 60 < latest_rsi <= 70:
                    interpretations["rsi"] = f"Approaching overbought conditions (RSI: {round(latest_rsi, 2)})."
        
        # Interpret MACD if available
        if "macd" in analysis.get("indicators", {}):
            macd_line = analysis["indicators"]["macd"].get("macd_line", [])
            signal_line = analysis["indicators"]["macd"].get("signal_line", [])
            histogram = analysis["indicators"]["macd"].get("histogram", [])
            
            if macd_line and signal_line and histogram:
                latest_macd = macd_line[-1]
                latest_signal = signal_line[-1]
                latest_histogram = histogram[-1]
                prev_histogram = histogram[-2] if len(histogram) > 1 else 0
                
                if latest_macd > latest_signal and latest_macd > 0:
                    interpretations["macd"] = "Bullish MACD crossover above zero. Strong bullish momentum."
                elif latest_macd > latest_signal and latest_macd < 0:
                    interpretations["macd"] = "Bullish MACD crossover below zero. Potential bullish momentum building."
                elif latest_macd < latest_signal and latest_macd > 0:
                    interpretations["macd"] = "Bearish MACD crossover above zero. Potential reversal of bullish momentum."
                elif latest_macd < latest_signal and latest_macd < 0:
                    interpretations["macd"] = "Bearish MACD crossover below zero. Strong bearish momentum."
                
                # Check for divergence between histogram direction
                if latest_histogram > prev_histogram:
                    interpretations["macd_histogram"] = "MACD histogram increasing, indicating strengthening momentum in the current direction."
                elif latest_histogram < prev_histogram:
                    interpretations["macd_histogram"] = "MACD histogram decreasing, indicating weakening momentum in the current direction."
        
        # Interpret Bollinger Bands if available
        if "bollinger" in analysis.get("indicators", {}):
            middle_band = analysis["indicators"]["bollinger"].get("middle_band", [])
            upper_band = analysis["indicators"]["bollinger"].get("upper_band", [])
            lower_band = analysis["indicators"]["bollinger"].get("lower_band", [])
            
            if middle_band and upper_band and lower_band and analysis.get("recent_data", {}).get("prices", []):
                latest_price = analysis["recent_data"]["prices"][-1]
                latest_middle = middle_band[-1]
                latest_upper = upper_band[-1]
                latest_lower = lower_band[-1]
                
                band_width = (latest_upper - latest_lower) / latest_middle
                
                if latest_price > latest_upper:
                    interpretations["bollinger"] = f"Price above upper Bollinger Band. Potential overbought conditions or strong upward momentum."
                elif latest_price < latest_lower:
                    interpretations["bollinger"] = f"Price below lower Bollinger Band. Potential oversold conditions or strong downward momentum."
                elif latest_price > latest_middle:
                    interpretations["bollinger"] = f"Price between middle and upper Bollinger Bands. Bullish bias within normal volatility."
                elif latest_price < latest_middle:
                    interpretations["bollinger"] = f"Price between middle and lower Bollinger Bands. Bearish bias within normal volatility."
                
                if band_width < 0.1:
                    interpretations["bollinger_width"] = "Narrow Bollinger Bands indicating low volatility. Potential for a volatility expansion soon."
                elif band_width > 0.3:
                    interpretations["bollinger_width"] = "Wide Bollinger Bands indicating high volatility. Potential for mean reversion."
        
        # Interpret Moving Averages if available
        ma_interpretations = []
        
        if "sma" in analysis.get("indicators", {}) and analysis.get("recent_data", {}).get("prices", []):
            sma_20 = analysis["indicators"]["sma"].get("sma_20", [])
            sma_50 = analysis["indicators"]["sma"].get("sma_50", [])
            sma_200 = analysis["indicators"]["sma"].get("sma_200", [])
            latest_price = analysis["recent_data"]["prices"][-1]
            
            if sma_20 and sma_50:
                latest_sma20 = sma_20[-1]
                latest_sma50 = sma_50[-1]
                
                if latest_sma20 > latest_sma50:
                    ma_interpretations.append("SMA 20 above SMA 50 (Golden Cross). Bullish signal.")
                elif latest_sma20 < latest_sma50:
                    ma_interpretations.append("SMA 20 below SMA 50 (Death Cross). Bearish signal.")
            
            if sma_200 and latest_price:
                latest_sma200 = sma_200[-1]
                
                if latest_price > latest_sma200:
                    ma_interpretations.append("Price above SMA 200. Long-term bullish bias.")
                else:
                    ma_interpretations.append("Price below SMA 200. Long-term bearish bias.")
        
        if ma_interpretations:
            interpretations["moving_averages"] = " ".join(ma_interpretations)
        
        # Overall market structure interpretation
        if "trend" in interpretations and ("rsi" in interpretations or "macd" in interpretations):
            trend_text = interpretations.get("trend", "")
            momentum_text = interpretations.get("rsi", interpretations.get("macd", ""))
            
            if "uptrend" in trend_text.lower() and "bullish" in momentum_text.lower():
                interpretations["overall"] = "Strong bullish market structure. Trend and momentum are aligned to the upside."
            elif "downtrend" in trend_text.lower() and "bearish" in momentum_text.lower():
                interpretations["overall"] = "Strong bearish market structure. Trend and momentum are aligned to the downside."
            elif "uptrend" in trend_text.lower() and "bearish" in momentum_text.lower():
                interpretations["overall"] = "Mixed signals with uptrend but weakening momentum. Caution advised."
            elif "downtrend" in trend_text.lower() and "bullish" in momentum_text.lower():
                interpretations["overall"] = "Mixed signals with downtrend but improving momentum. Potential reversal building."
            elif "sideways" in trend_text.lower():
                interpretations["overall"] = "Ranging market with no clear direction. Suitable for range-bound trading strategies."
        
        return interpretations
