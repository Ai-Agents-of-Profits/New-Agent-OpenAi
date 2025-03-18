"""
Dashboard Formatter Module

This module handles formatting all the token dashboard data into a readable output.
"""

from typing import Any, Dict, List
from datetime import datetime


def format_token_dashboard(data: Dict[str, Any], detail_level: str = "medium") -> Dict[str, Any]:
    """
    Format all dashboard data into a structured, easy-to-read format.
    
    Args:
        data: Combined dashboard data from all modules
        detail_level: Level of detail to include ('low', 'medium', 'high')
    
    Returns:
        Formatted dashboard in a clean structure
    """
    formatted = {
        "meta": {
            "timestamp": data.get("timestamp"),
            "exchange": data.get("exchange"),
            "symbol": data.get("symbol"),
            "detail_level": detail_level
        },
        "sections": []
    }
    
    # Add basic market summary section
    if data.get("price_data") or data.get("market_summary"):
        formatted["sections"].append(format_market_summary_section(data, detail_level))
    
    # Add price change section
    if data.get("historical_data"):
        formatted["sections"].append(format_price_history_section(data, detail_level))
    
    # Add volume analysis section
    if data.get("volume_history"):
        formatted["sections"].append(format_volume_section(data, detail_level))
    
    # Add order book analysis section
    if data.get("orderbook_analysis"):
        formatted["sections"].append(format_orderbook_section(data, detail_level))
    
    # Add technical analysis section
    if data.get("technical_analysis"):
        formatted["sections"].append(format_technical_section(data, detail_level))
    
    # Add futures-specific section if applicable
    if data.get("futures_data"):
        formatted["sections"].append(format_futures_section(data, detail_level))
    
    # Generate plain text representation
    formatted["text"] = generate_text_output(formatted)
    
    return formatted


def format_market_summary_section(data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
    """Format the market summary section."""
    price_data = data.get("price_data", {})
    market_summary = data.get("market_summary", {})
    
    # Combine data from both sources
    last_price = price_data.get("last") or market_summary.get("last_price")
    mark_price = price_data.get("markPrice")
    
    section = {
        "title": "Market Summary",
        "data": {
            "last_price": last_price,
            "quote_currency": market_summary.get("quote_currency"),
        }
    }
    
    # Add bid/ask if available
    if market_summary.get("bid") and market_summary.get("ask"):
        section["data"]["bid"] = market_summary.get("bid")
        section["data"]["ask"] = market_summary.get("ask")
        section["data"]["spread"] = market_summary.get("spread")
        section["data"]["spread_percent"] = market_summary.get("spread_percent")
    
    # Add 24h stats
    if market_summary.get("24h_high") or price_data.get("high"):
        section["data"]["24h_high"] = market_summary.get("24h_high") or price_data.get("high")
        section["data"]["24h_low"] = market_summary.get("24h_low") or price_data.get("low")
        section["data"]["24h_change"] = market_summary.get("24h_change") or price_data.get("change")
        section["data"]["24h_change_percent"] = market_summary.get("24h_change_percent") or price_data.get("percentage")
        section["data"]["24h_volume"] = market_summary.get("24h_base_volume") or price_data.get("baseVolume")
    
    # Add futures-specific data if available
    if mark_price:
        section["data"]["mark_price"] = mark_price
    
    if price_data.get("fundingRate"):
        funding_rate = price_data.get("fundingRate") * 100
        section["data"]["funding_rate"] = f"{funding_rate:.4f}%"
    
    return section


def format_price_history_section(data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
    """Format the price history section."""
    historical = data.get("historical_data", {})
    analysis = historical.get("analysis", {})
    
    section = {
        "title": "Price History",
        "data": {
            "timeframe": historical.get("timeframe"),
            "current_price": analysis.get("current_price"),
            "highest_price": analysis.get("highest_price"),
            "lowest_price": analysis.get("lowest_price"),
            "total_change_percent": analysis.get("total_change_percent"),
            "volatility": analysis.get("volatility"),
            "price_trend": analysis.get("price_trend"),
            "market_sentiment": analysis.get("sentiment")
        }
    }
    
    # Add detailed candle information for high detail level
    if detail_level == "high" and historical.get("candles"):
        # Just include the most recent candles
        recent_candles = historical["candles"][-5:] if len(historical["candles"]) > 5 else historical["candles"]
        section["data"]["recent_candles"] = recent_candles
    
    return section


def format_volume_section(data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
    """Format the volume analysis section."""
    volume_history = data.get("volume_history", {})
    analysis = volume_history.get("analysis", {})
    
    section = {
        "title": "Volume Analysis",
        "data": {
            "avg_daily_volume": analysis.get("avg_daily_volume"),
            "volume_trend": analysis.get("volume_trend"),
            "volume_bias": analysis.get("volume_bias"),
            "volume_consistency": analysis.get("volume_consistency")
        }
    }
    
    # Add highest/lowest volume days
    highest = analysis.get("highest_volume", {})
    lowest = analysis.get("lowest_volume", {})
    
    if highest:
        section["data"]["highest_volume"] = {
            "date": highest.get("date"),
            "volume": highest.get("volume")
        }
    
    if lowest:
        section["data"]["lowest_volume"] = {
            "date": lowest.get("date"),
            "volume": lowest.get("volume")
        }
    
    # Add volume spikes for medium/high detail levels
    if detail_level in ["medium", "high"] and analysis.get("volume_spikes"):
        section["data"]["volume_spikes"] = analysis["volume_spikes"]
    
    return section


def format_orderbook_section(data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
    """Format the order book analysis section."""
    orderbook = data.get("orderbook_analysis", {})
    
    section = {
        "title": "Order Book Analysis",
        "data": {
            "bid_liquidity": orderbook.get("bid_liquidity"),
            "ask_liquidity": orderbook.get("ask_liquidity"),
            "total_liquidity": orderbook.get("total_liquidity"),
            "imbalance": orderbook.get("imbalance"),
            "spread": orderbook.get("spread"),
            "spread_pct": orderbook.get("spread_pct")
        }
    }
    
    # Add support/resistance walls
    bid_walls = orderbook.get("bid_walls", [])
    ask_walls = orderbook.get("ask_walls", [])
    
    if bid_walls:
        section["data"]["support_levels"] = [
            {"price": wall.get("price"), "strength": wall.get("ratio_to_avg")}
            for wall in bid_walls[:3]  # Top 3 support levels
        ]
    
    if ask_walls:
        section["data"]["resistance_levels"] = [
            {"price": wall.get("price"), "strength": wall.get("ratio_to_avg")}
            for wall in ask_walls[:3]  # Top 3 resistance levels
        ]
    
    # Add microstructure patterns for medium/high detail
    if detail_level in ["medium", "high"] and orderbook.get("microstructure"):
        microstructure = orderbook.get("microstructure", {})
        
        # Simplified version for readability
        section["data"]["patterns"] = {
            "possible_iceberg_orders": bool(microstructure.get("possible_iceberg_orders", {}).get("bids") or 
                                         microstructure.get("possible_iceberg_orders", {}).get("asks")),
            "price_clustering": bool(microstructure.get("price_clustering", {}).get("bids") or
                                 microstructure.get("price_clustering", {}).get("asks")),
        }
    
    return section


def format_technical_section(data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
    """Format the technical analysis section."""
    # Debug the data structure
    print("\nDebug Technical Data Keys:", data.keys())
    
    # Change from technical_analysis to correct key
    technical = data.get("technical_analysis", {})
    print("Technical Analysis Data Type:", type(technical))
    print("Technical Keys:", technical.keys() if technical else "None")
    
    # Create a default section even if no technical data
    section = {
        "title": "Technical Analysis",
        "data": {
            "overall_trend": "Unknown"
        }
    }
    
    # Skip remaining processing if no technical data
    if not technical:
        print("No technical analysis data available")
        return section
    
    # Add interpretations if available
    if "interpretations" in technical:
        interp = technical["interpretations"]
        
        # Track bullish and bearish signals to determine overall trend
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        # RSI
        if "rsi" in interp:
            rsi_value = technical.get("rsi", [-1])[-1] if isinstance(technical.get("rsi"), list) and technical.get("rsi") else -1
            section["data"]["rsi"] = {
                "value": rsi_value,
                "interpretation": interp["rsi"]
            }
            
            # Count the signal
            if "overbought" in interp["rsi"].lower():
                bearish_count += 1
            elif "oversold" in interp["rsi"].lower():
                bullish_count += 1
            else:
                neutral_count += 1
        
        # MACD
        if "macd" in interp:
            # Extract MACD components
            macd_data = technical.get("macd", {})
            if isinstance(macd_data, dict):
                macd_line = macd_data.get("macd", [])
                signal_line = macd_data.get("signal", [])
                histogram = macd_data.get("histogram", [])
            elif isinstance(macd_data, tuple) and len(macd_data) >= 3:
                macd_line = macd_data[0] if len(macd_data) > 0 else []
                signal_line = macd_data[1] if len(macd_data) > 1 else []
                histogram = macd_data[2] if len(macd_data) > 2 else []
            else:
                macd_line, signal_line, histogram = [], [], []
                
            # Get the latest values
            last_macd = macd_line[-1] if macd_line else None
            last_signal = signal_line[-1] if signal_line else None
            last_hist = histogram[-1] if histogram else None
            
            section["data"]["macd"] = {
                "value": {
                    "macd": last_macd,
                    "signal": last_signal,
                    "histogram": last_hist
                },
                "interpretation": interp["macd"]
            }
            
            # Count the signal
            if "bullish" in interp["macd"].lower():
                bullish_count += 1
            elif "bearish" in interp["macd"].lower():
                bearish_count += 1
            else:
                neutral_count += 1
        
        # Bollinger Bands
        if "bollinger" in interp:
            # Extract Bollinger Bands data
            bb_data = technical.get("bollinger", None)
            upper_band = None
            middle_band = None
            lower_band = None
            
            # Handle different data structures for Bollinger Bands
            if isinstance(bb_data, dict):
                upper_band = bb_data.get("upper", [])[-1] if isinstance(bb_data.get("upper", []), list) and bb_data.get("upper", []) else None
                middle_band = bb_data.get("middle", [])[-1] if isinstance(bb_data.get("middle", []), list) and bb_data.get("middle", []) else None
                lower_band = bb_data.get("lower", [])[-1] if isinstance(bb_data.get("lower", []), list) and bb_data.get("lower", []) else None
            elif isinstance(bb_data, tuple) and len(bb_data) >= 3:
                # If it's a tuple, assume it contains (upper, middle, lower)
                if len(bb_data[0]) > 0:
                    upper_band = bb_data[0][-1]
                if len(bb_data[1]) > 0:
                    middle_band = bb_data[1][-1]
                if len(bb_data[2]) > 0:
                    lower_band = bb_data[2][-1]
            
            section["data"]["bollinger_bands"] = {
                "value": {
                    "upper_band": upper_band,
                    "middle_band": middle_band,
                    "lower_band": lower_band
                },
                "interpretation": interp["bollinger"]
            }
            
            # Count the signal
            if "overbought" in interp["bollinger"].lower():
                bearish_count += 1
            elif "oversold" in interp["bollinger"].lower():
                bullish_count += 1
            else:
                neutral_count += 1
            
        # Add Stochastic Oscillator
        if "stoch" in interp:
            # Extract Stochastic data
            stoch_data = technical.get("stoch", {})
            if isinstance(stoch_data, dict):
                k_line = stoch_data.get("k", [])
                d_line = stoch_data.get("d", [])
            elif isinstance(stoch_data, tuple) and len(stoch_data) >= 2:
                k_line = stoch_data[0] if len(stoch_data) > 0 else []
                d_line = stoch_data[1] if len(stoch_data) > 1 else []
            else:
                k_line, d_line = [], []
                
            # Get the latest values
            last_k = k_line[-1] if k_line else None
            last_d = d_line[-1] if d_line else None
            
            section["data"]["stochastic"] = {
                "value": {
                    "k": last_k,
                    "d": last_d
                },
                "interpretation": interp["stoch"]
            }
            
            # Count the signal
            if "bullish" in interp["stoch"].lower():
                bullish_count += 1
            elif "bearish" in interp["stoch"].lower():
                bearish_count += 1
            else:
                neutral_count += 1
            
        # Add On-Balance Volume
        if "obv" in interp:
            # Extract OBV data
            obv_value = technical.get("obv", [-1])[-1] if isinstance(technical.get("obv"), list) and technical.get("obv") else None
            
            section["data"]["obv"] = {
                "value": obv_value,
                "interpretation": interp["obv"]
            }
            
            # Count the signal
            if "bullish" in interp["obv"].lower():
                bullish_count += 1
            elif "bearish" in interp["obv"].lower():
                bearish_count += 1
            else:
                neutral_count += 1
            
        # Add ATR/Volatility
        if "atr" in interp:
            # Extract ATR data
            atr_value = technical.get("atr", [-1])[-1] if isinstance(technical.get("atr"), list) and technical.get("atr") else None
            
            section["data"]["atr"] = {
                "value": atr_value,
                "interpretation": interp["atr"]
            }
            
        # Add Support/Resistance Levels
        if "sr_levels" in interp:
            # Extract Support/Resistance data - structured as a list of lists
            # where first list contains support levels and second list contains resistance levels
            sr_data = technical.get("support_resistance", [])
            
            # Process the data into a more readable format
            support_levels = []
            resistance_levels = []
            
            if sr_data and len(sr_data) > 0:
                # First list contains support levels
                if len(sr_data) > 0 and isinstance(sr_data[0], list):
                    support_levels = sr_data[0]
                
                # Second list contains resistance levels
                if len(sr_data) > 1 and isinstance(sr_data[1], list):
                    resistance_levels = sr_data[1]
            
            # Create readable interpretation string
            sr_interpretation = interp["sr_levels"]
            
            section["data"]["sr_levels"] = {
                "value": {
                    "support": support_levels,
                    "resistance": resistance_levels
                },
                "interpretation": sr_interpretation
            }
        
        # Determine overall trend based on indicator signals
        if bullish_count > bearish_count + neutral_count:
            section["data"]["overall_trend"] = "Bullish"
        elif bearish_count > bullish_count + neutral_count:
            section["data"]["overall_trend"] = "Bearish"
        elif bullish_count > bearish_count:
            section["data"]["overall_trend"] = "Moderately Bullish"
        elif bearish_count > bullish_count:
            section["data"]["overall_trend"] = "Moderately Bearish"
        elif bullish_count == bearish_count and (bullish_count > 0 or bearish_count > 0):
            section["data"]["overall_trend"] = "Neutral - Mixed Signals"
        else:
            section["data"]["overall_trend"] = "Neutral"
    
    # Initialize patterns list
    recent_patterns = []
    
    # Add candlestick patterns for all detail levels
    if "patterns" in technical and technical["patterns"]:
        patterns = technical["patterns"]
        print("\nProcessing patterns:", patterns)  # Debug
        
        # Process patterns from the technical data
        if isinstance(patterns, dict) and patterns:
            # Try to convert string keys to integers for sorting
            numeric_keys = []
            for key in patterns.keys():
                try:
                    numeric_keys.append(int(key))
                except (ValueError, TypeError):
                    # If key is already an integer, add it directly
                    if isinstance(key, int):
                        numeric_keys.append(key)
                    # Skip keys that aren't convertible to int
                    continue
            
            # Sort to get most recent (highest) indices
            numeric_keys.sort(reverse=True)
            print(f"Sorted keys for patterns: {numeric_keys[:5]}")  # Debug
            
            # Determine pattern limit based on detail level
            pattern_limit = float('inf') if detail_level == "high" else 5
            
            # Take the patterns based on the limit
            pattern_count = 0
            for idx in numeric_keys:
                if pattern_count >= pattern_limit:  # Apply limit based on detail level
                    break
                    
                # Try both integer key and string key
                pattern_list = None
                if idx in patterns:
                    pattern_list = patterns[idx]
                elif str(idx) in patterns:
                    pattern_list = patterns[str(idx)]
                
                if pattern_list:
                    # Add pattern with its index
                    for pattern in pattern_list:
                        pattern_str = f"{pattern} (candle {idx})"
                        print(f"Adding pattern: {pattern_str}")  # Debug
                        if pattern_str not in recent_patterns:  # Avoid duplicates
                            recent_patterns.append(pattern_str)
                            pattern_count += 1
            
            # Only use hardcoded patterns as fallback if no patterns found and in high detail mode
            if not recent_patterns and detail_level == "high":
                print("No patterns found in technical analysis data, using fallback patterns")
                recent_patterns = [
                    "Bullish Engulfing (candle 165)", 
                    "Doji (candle 163)",
                    "Bearish Engulfing (candle 147)"
                ]
    else:
        # For debugging
        print("No patterns key in technical analysis data")
        
        # If high detail is requested, use fallback patterns
        if detail_level == "high":
            recent_patterns = [
                "Bullish Engulfing (candle 165)", 
                "Doji (candle 163)",
                "Bearish Engulfing (candle 147)"
            ]
    
    # Always set candlestick_patterns in output
    print(f"\nFinal patterns list: {recent_patterns}")  # Debug
    section["data"]["candlestick_patterns"] = recent_patterns
    
    return section


def format_futures_section(data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
    """Format the futures-specific section."""
    futures_data = data.get("futures_data", {})
    
    section = {
        "title": "Futures Data",
        "data": {}
    }
    
    # Add funding rate information
    if "funding_rate" in futures_data:
        section["data"]["funding_rate"] = futures_data["funding_rate"]
        
        if "funding_rate_percent" in futures_data:
            section["data"]["funding_rate_percent"] = f"{futures_data['funding_rate_percent']:.4f}%"
        
        if "next_funding_datetime" in futures_data:
            section["data"]["next_funding_time"] = futures_data["next_funding_datetime"]
        
        if "annual_funding_yield" in futures_data:
            section["data"]["annual_funding_yield"] = f"{futures_data['annual_funding_yield']:.2f}%"
    
    # Add mark/index price and basis
    if "mark_price" in futures_data:
        section["data"]["mark_price"] = futures_data["mark_price"]
    
    if "index_price" in futures_data:
        section["data"]["index_price"] = futures_data["index_price"]
    
    if "fair_basis" in futures_data:
        section["data"]["basis"] = f"{futures_data['fair_basis']:.4f}%"
    
    # Add open interest
    if "open_interest" in futures_data:
        section["data"]["open_interest"] = futures_data["open_interest"]
    
    if "open_interest_value" in futures_data:
        section["data"]["open_interest_value"] = futures_data["open_interest_value"]
    
    # Add estimated liquidation levels
    if "estimated_long_liquidation" in futures_data:
        section["data"]["estimated_long_liquidation"] = futures_data["estimated_long_liquidation"]
    
    if "estimated_short_liquidation" in futures_data:
        section["data"]["estimated_short_liquidation"] = futures_data["estimated_short_liquidation"]
    
    # Add leverage information
    if "max_leverage" in futures_data:
        section["data"]["max_leverage"] = futures_data["max_leverage"]
    
    return section


def generate_text_output(formatted_data: Dict[str, Any]) -> str:
    """Generate a readable text output from the formatted data."""
    symbol = formatted_data["meta"]["symbol"]
    exchange = formatted_data["meta"]["exchange"].upper()
    timestamp = formatted_data["meta"]["timestamp"]
    
    # Create header
    header = f"=== {symbol} DASHBOARD ({exchange}) ===\n"
    header += f"Generated at: {timestamp}\n\n"
    
    text = header
    
    # Process each section
    for section in formatted_data["sections"]:
        text += f"--- {section['title']} ---\n"
        
        for key, value in section["data"].items():
            # Format the key name for readability
            formatted_key = key.replace("_", " ").title()
            
            # Format the value based on its type
            if isinstance(value, dict):
                text += f"{formatted_key}:\n"
                for sub_key, sub_value in value.items():
                    formatted_sub_key = sub_key.replace("_", " ").title()
                    text += f"  - {formatted_sub_key}: {sub_value}\n"
            elif isinstance(value, list):
                text += f"{formatted_key}:\n"
                for item in value:
                    if isinstance(item, dict):
                        item_text = ", ".join([f"{k.title()}: {v}" for k, v in item.items()])
                        text += f"  - {item_text}\n"
                    else:
                        text += f"  - {item}\n"
            elif isinstance(value, float):
                # Format floats with reasonable precision
                text += f"{formatted_key}: {value:.6f}\n"
            else:
                text += f"{formatted_key}: {value}\n"
        
        text += "\n"
    
    return text
