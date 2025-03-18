import asyncio
from typing import Any, Dict, List
import ccxt.async_support as ccxt
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from datetime import datetime, timedelta
import sys
import os
import platform

# Fix for Windows: Use SelectorEventLoop instead of ProactorEventLoop (default on Windows)
# This is needed because aiodns requires SelectorEventLoop
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import technical_analysis as ta

# Import token dashboard
from src.token_dashboard import get_token_dashboard

# Initialize server
server = Server("crypto-server")

# Define supported exchanges and their instances
SUPPORTED_EXCHANGES = {
    'binance': ccxt.binance,
    'binanceusdm': ccxt.binanceusdm,   # USD-Margined futures (USDT/BUSD settled)
    'binancecoinm': ccxt.binancecoinm, # COIN-Margined futures (BTC/ETH settled)
    'coinbase': ccxt.coinbase,
    'kraken': ccxt.kraken,
    'kucoin': ccxt.kucoin,
    'hyperliquid': ccxt.hyperliquid,
    'huobi': ccxt.huobi,
    'bitfinex': ccxt.bitfinex,
    'bybit': ccxt.bybit,
    'okx': ccxt.okx,
    'mexc': ccxt.mexc
}

# Exchange instances cache
exchange_instances = {}


async def get_exchange(exchange_id: str) -> ccxt.Exchange:
    """Get or create an exchange instance."""
    exchange_id = exchange_id.lower()
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Unsupported exchange: {exchange_id}")

    if exchange_id not in exchange_instances:
        exchange_class = SUPPORTED_EXCHANGES[exchange_id]
        exchange_instances[exchange_id] = exchange_class()
        
        # Configure futures-specific settings if applicable
        if exchange_id in ['binanceusdm', 'binancecoinm']:
            exchange_instances[exchange_id].options['defaultType'] = 'future'

    return exchange_instances[exchange_id]


def is_futures_exchange(exchange_id: str) -> bool:
    """Check if the exchange is a futures exchange."""
    return exchange_id.lower() in ['binanceusdm', 'binancecoinm', 'bybit']


def normalize_symbol(symbol: str, exchange_id: str) -> str:
    """
    Normalize the symbol format based on the exchange type.
    For futures exchanges, if the symbol doesn't contain a colon it 
    might need to be converted to the appropriate futures format.
    """
    if not is_futures_exchange(exchange_id):
        return symbol
        
    # Handle different futures symbol formats
    if exchange_id.lower() == 'binanceusdm':
        # For USDM futures, format is typically BASE/QUOTE:QUOTE
        # e.g., BTC/USDT:USDT or simply BTC/USDT
        if ':' not in symbol and '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base}/{quote}:{quote}"
    elif exchange_id.lower() == 'binancecoinm':
        # For COINM futures, format is typically BASE/USD:BASE
        # e.g., BTC/USD:BTC
        if ':' not in symbol and '/' in symbol:
            base, quote = symbol.split('/')
            if quote == 'USD':
                return f"{base}/{quote}:{base}"
                
    return symbol


async def format_ticker(ticker: Dict[str, Any], exchange_id: str) -> str:
    """Format ticker data into a readable string."""
    return (
        f"Exchange: {exchange_id.upper()}\n"
        f"Symbol: {ticker.get('symbol')}\n"
        f"Last Price: {ticker.get('last', 'N/A')}\n"
        f"24h High: {ticker.get('high', 'N/A')}\n"
        f"24h Low: {ticker.get('low', 'N/A')}\n"
        f"24h Volume: {ticker.get('baseVolume', 'N/A')}\n"
        f"Bid: {ticker.get('bid', 'N/A')}\n"
        f"Ask: {ticker.get('ask', 'N/A')}\n"
        f"Timestamp: {datetime.fromtimestamp(ticker.get('timestamp', 0)/1000).strftime('%Y-%m-%d %H:%M:%S') if ticker.get('timestamp') else 'N/A'}"
    )


# Helper functions for order book analysis
async def _calculate_orderbook_metrics(orderbook: Dict[str, List]) -> Dict[str, Any]:
    """Calculate core metrics from order book data."""
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {"error": "Invalid order book data"}
    
    bids = orderbook['bids']
    asks = orderbook['asks']
    
    # Basic metrics
    metrics = {}
    
    # Calculate bid/ask liquidity
    bid_liquidity = sum(amount for price, amount in bids)
    ask_liquidity = sum(amount for price, amount in asks)
    
    # Calculate imbalance (positive means more buy orders)
    total_liquidity = bid_liquidity + ask_liquidity
    imbalance = (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0
    
    # Calculate current spread
    if bids and asks:
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100
    else:
        best_bid = best_ask = spread = spread_pct = 0
    
    metrics.update({
        "bid_liquidity": bid_liquidity,
        "ask_liquidity": ask_liquidity,
        "total_liquidity": total_liquidity,
        "imbalance": imbalance,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_pct": spread_pct
    })
    
    return metrics

def _calculate_depth_distribution(orders: List, bins: int = 5) -> Dict[str, float]:
    """Calculate the distribution of volume across price levels."""
    if not orders:
        return {}
    
    min_price = orders[-1][0]
    max_price = orders[0][0]
    price_range = max_price - min_price
    
    if price_range <= 0:
        return {f"{min_price}": sum(amount for _, amount in orders)}
    
    bin_size = price_range / bins
    distribution = {}
    
    for i in range(bins):
        bin_start = min_price + i * bin_size
        bin_end = bin_start + bin_size if i < bins - 1 else max_price + 0.0000001  # Include the last price
        bin_volume = sum(amount for price, amount in orders if bin_start <= price < bin_end)
        distribution[f"{bin_start:.8f}-{bin_end:.8f}"] = bin_volume
    
    return distribution

def _calculate_cumulative_volume_distribution(orders: List, price_levels: int = 10) -> Dict[str, float]:
    """Calculate cumulative volume at different price levels from the best price."""
    if not orders:
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
        if not orders:
            break
            
        # Calculate price threshold for this level (percentage from best price)
        if best_price > 0:
            if orders[0][0] > orders[-1][0]:  # Bids (buy orders) - descending prices
                threshold = best_price * (1 - (i * step_size / 100))
                level_orders = [order for order in orders if order[0] >= threshold]
                level_name = f"Within -{i*step_size:.2f}%"
            else:  # Asks (sell orders) - ascending prices
                threshold = best_price * (1 + (i * step_size / 100))
                level_orders = [order for order in orders if order[0] <= threshold]
                level_name = f"Within +{i*step_size:.2f}%"
        else:
            level_orders = []
            level_name = f"Level {i+1}"
        
        # Sum the volume at this level
        level_volume = sum(amount for _, amount in level_orders)
        cumulative_volume += level_volume
        cumulative_distribution[level_name] = cumulative_volume
    
    return cumulative_distribution

def _identify_liquidity_walls(orders: List, threshold_factor: float = 3.0) -> List[Dict[str, float]]:
    """Identify price levels with significantly higher liquidity than average."""
    if not orders:
        return []
    
    volumes = [amount for _, amount in orders]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0
    threshold = avg_volume * threshold_factor
    
    walls = []
    for price, amount in orders:
        if amount > threshold:
            walls.append({"price": price, "volume": amount})
    
    return walls

def _detect_microstructure_patterns(orderbook: Dict[str, List]) -> Dict[str, Any]:
    """Detect market microstructure patterns in order book data."""
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {}
    
    bids = orderbook['bids']
    asks = orderbook['asks']
    
    if not bids or not asks:
        return {}
    
    patterns = {}
    
    # Get some basic metrics for pattern detection
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    
    # Calculate volume-weighted average prices (VWAP) for both sides
    bid_volume = sum(amount for _, amount in bids)
    ask_volume = sum(amount for _, amount in asks)
    
    bid_vwap = sum(price * amount for price, amount in bids) / bid_volume if bid_volume > 0 else best_bid
    ask_vwap = sum(price * amount for price, amount in asks) / ask_volume if ask_volume > 0 else best_ask
    
    # Calculate volume profile skew (higher values mean more volume toward top of book)
    bid_skew = 0
    ask_skew = 0
    
    if bids:
        top_5_bids_volume = sum(amount for _, amount in bids[:min(5, len(bids))])
        bid_skew = top_5_bids_volume / bid_volume if bid_volume > 0 else 0
    
    if asks:
        top_5_asks_volume = sum(amount for _, amount in asks[:min(5, len(asks))])
        ask_skew = top_5_asks_volume / ask_volume if ask_volume > 0 else 0
    
    # Detect specific patterns
    
    # 1. Iceberg detection (hidden liquidity) - look for repeated same-sized orders
    bid_sizes = [amount for _, amount in bids[:min(10, len(bids))]]
    ask_sizes = [amount for _, amount in asks[:min(10, len(asks))]]
    
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
    far_bid_volume = sum(amount for price, amount in bids if (mid_price - price) / mid_price > 0.02)
    far_ask_volume = sum(amount for price, amount in asks if (price - mid_price) / mid_price > 0.02)
    
    spoofing_threshold = 5  # 5x normal volume would be suspicious
    avg_bid_size = bid_volume / len(bids) if bids else 0
    avg_ask_size = ask_volume / len(asks) if asks else 0
    
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

async def format_orderbook_analysis(analysis: Dict[str, Any]) -> str:
    """Format order book analysis into a readable string."""
    if "error" in analysis:
        return f"Error: {analysis['error']}"
    
    result = [
        f"Order Book Analysis:",
        f"-------------------",
        f"Bid Liquidity: {analysis['bid_liquidity']:.4f}",
        f"Ask Liquidity: {analysis['ask_liquidity']:.4f}",
        f"Total Liquidity: {analysis['total_liquidity']:.4f}",
        f"Imbalance: {analysis['imbalance']:.4f} ({'+' if analysis['imbalance'] > 0 else '-'})",
        f"Best Bid: {analysis['best_bid']:.8f}",
        f"Best Ask: {analysis['best_ask']:.8f}",
        f"Spread: {analysis['spread']:.8f} ({analysis['spread_pct']:.2f}%)",
        f"",
        f"Volume Distribution (Bids):",
        f"-------------------------"
    ]
    
    for price_range, volume in analysis.get('bid_distribution', {}).items():
        result.append(f"{price_range}: {volume:.4f}")
    
    result.extend([
        f"",
        f"Volume Distribution (Asks):",
        f"-------------------------"
    ])
    
    for price_range, volume in analysis.get('ask_distribution', {}).items():
        result.append(f"{price_range}: {volume:.4f}")
    
    # Add cumulative volume distribution
    if 'bid_cumulative_distribution' in analysis:
        result.extend([
            f"",
            f"Cumulative Bid Volume Distribution:",
            f"---------------------------------"
        ])
        for level, volume in analysis.get('bid_cumulative_distribution', {}).items():
            result.append(f"{level}: {volume:.4f}")
    
    if 'ask_cumulative_distribution' in analysis:
        result.extend([
            f"",
            f"Cumulative Ask Volume Distribution:",
            f"---------------------------------"
        ])
        for level, volume in analysis.get('ask_cumulative_distribution', {}).items():
            result.append(f"{level}: {volume:.4f}")
    
    result.extend([
        f"",
        f"Liquidity Walls (Significant support/resistance):",
        f"--------------------------------------------"
    ])
    
    if analysis.get('bid_walls'):
        result.append(f"Support walls:")
        for wall in analysis['bid_walls'][:10]:
            result.append(f"  Price: {wall['price']:.8f}, Volume: {wall['volume']:.4f}")
        if len(analysis['bid_walls']) > 10:
            result.append("...")
    else:
        result.append(f"No significant support walls detected")
    
    if analysis.get('ask_walls'):
        result.append(f"Resistance walls:")
        for wall in analysis['ask_walls'][:10]:
            result.append(f"  Price: {wall['price']:.8f}, Volume: {wall['volume']:.4f}")
        if len(analysis['ask_walls']) > 10:
            result.append("...")
    else:
        result.append(f"No significant resistance walls detected")
    
    # Add microstructure patterns if available
    if 'microstructure' in analysis:
        result.extend([
            f"",
            f"Market Microstructure Analysis:",
            f"-----------------------------"
        ])
        
        micro = analysis['microstructure']
        
        result.append(f"Bid VWAP: {micro.get('bid_vwap', 0):.8f}")
        result.append(f"Ask VWAP: {micro.get('ask_vwap', 0):.8f}")
        result.append(f"Bid Skew: {micro.get('bid_skew', 0):.4f} (higher = volume at top of book)")
        result.append(f"Ask Skew: {micro.get('ask_skew', 0):.4f} (higher = volume at top of book)")
        result.append(f"Order Book Pressure: {micro.get('bid_pressure', 0):.4f} (>1 = buy pressure, <1 = sell pressure)")
        result.append(f"Pressure Direction: {micro.get('pressure_direction', 'neutral')}")
        
        # Iceberg orders
        if micro.get('potential_bid_icebergs'):
            result.append(f"Potential iceberg buy orders:")
            for size, count in micro['potential_bid_icebergs']:
                result.append(f"  Size: {size:.4f}, Count: {count}")
        
        if micro.get('potential_ask_icebergs'):
            result.append(f"Potential iceberg sell orders:")
            for size, count in micro['potential_ask_icebergs']:
                result.append(f"  Size: {size:.4f}, Count: {count}")
        
        # Interpretations
        if micro.get('interpretations'):
            result.extend([
                f"",
                f"Interpretations:"
            ])
            result.extend(os.linesep.join([f"• {interp}" for interp in micro['interpretations']]))
    
    return "\n".join(result)


async def format_compact_orderbook_analysis(analysis: Dict[str, Any], symbol: str) -> str:
    """Format order book analysis into a compact, visually appealing format."""
    if "error" in analysis:
        return f"Error: {analysis['error']}"
    
    # Get the quote currency for better formatting
    quote_currency = symbol.split('/')[1].split(':')[0] if '/' in symbol else 'USDT'
    
    # Calculate some metrics for the visualizations
    bid_ask_ratio = analysis['bid_liquidity'] / analysis['ask_liquidity'] if analysis['ask_liquidity'] > 0 else float('inf')
    market_pressure = "UP" if analysis['imbalance'] > 0.2 else "DOWN" if analysis['imbalance'] < -0.2 else "NEUTRAL"
    
    # Format liquidity walls concisely
    support_walls = [f"{wall['price']:.4f}" for wall in analysis.get('bid_walls', [])[:5]]
    resistance_walls = [f"{wall['price']:.4f}" for wall in analysis.get('ask_walls', [])[:5]]
    
    # Get market microstructure insights
    micro = analysis.get('microstructure', {})
    interpretations = micro.get('interpretations', [])
    
    # Create a compact visualization of the orderbook
    result = [
        f"# ORDERBOOK ANALYSIS FOR {symbol}",
        f"",
        f"## MARKET LIQUIDITY OVERVIEW",
        f"```",
        f"Bid Liquidity: {analysis['bid_liquidity']:.2f} {quote_currency} | Ask Liquidity: {analysis['ask_liquidity']:.2f} {quote_currency}",
        f"Ratio (Bid:Ask): {bid_ask_ratio:.2f} | Market Pressure: {market_pressure} | Imbalance: {analysis['imbalance']*100:.1f}%",
        f"Spread: {analysis['spread']:.6f} {quote_currency} ({analysis['spread_pct']:.2f}%)",
        f"```",
        f"",
        f"## SUPPORT & RESISTANCE",
        f"```",
        f"Key Support (Buy Walls): {', '.join(support_walls) if support_walls else 'None detected'}",
        f"Key Resistance (Sell Walls): {', '.join(resistance_walls) if resistance_walls else 'None detected'}",
        f"```",
        f"",
        f"## MARKET STRUCTURE",
    ]
    
    # Add volume distribution visualization
    if 'bid_distribution' in analysis or 'ask_distribution' in analysis:
        result.append("```")
        result.append("Volume distribution:")
        
        # Simplified volume distribution - just show highest concentration areas
        bid_vol = sorted([(float(k.split('-')[0]), v) for k, v in analysis.get('bid_distribution', {}).items()], 
                        key=lambda x: x[1], reverse=True)[:2]
        ask_vol = sorted([(float(k.split('-')[0]), v) for k, v in analysis.get('ask_distribution', {}).items()], 
                        key=lambda x: x[1], reverse=True)[:2]
        
        if bid_vol:
            result.append(f"Highest buy volume: Around {bid_vol[0][0]:.4f} {quote_currency}")
        if ask_vol:
            result.append(f"Highest sell volume: Around {ask_vol[0][0]:.4f} {quote_currency}")
        
        result.append("```")
    
    # Add microstructure insights
    if micro:
        # Calculate buyer vs seller dominance based on bid/ask skew
        bid_skew = micro.get('bid_skew', 0)
        ask_skew = micro.get('ask_skew', 0)
        
        skew_diff = bid_skew - ask_skew
        if abs(skew_diff) > 0.1:
            skew_insight = f"{'Buyers' if skew_diff > 0 else 'Sellers'} are more aggressive"
        else:
            skew_insight = "Neutral aggression between buyers and sellers"
        
        result.append("```")
        result.append(f"Order Flow: {micro.get('pressure_direction', 'neutral').title()} | {skew_insight}")
        
        # Add potential iceberg detection
        if micro.get('potential_bid_icebergs') or micro.get('potential_ask_icebergs'):
            icebergs = []
            if micro.get('potential_bid_icebergs'):
                icebergs.append(f"Buy icebergs detected")
            if micro.get('potential_ask_icebergs'):
                icebergs.append(f"Sell icebergs detected")
            result.append(f"Hidden Orders: {' & '.join(icebergs)}")
        
        result.append("```")
    
    # Add key interpretations
    if interpretations:
        result.append("")
        result.append("## KEY INSIGHTS")
        result.append("```")
        for i, insight in enumerate(interpretations[:3]):
            result.append(f"• {insight}")
        result.append("```")
    
    return "\n".join(result)


async def format_technical_analysis(data: Dict[str, Any], symbol: str, indicators: List[str]) -> str:
    """Format technical analysis data into a readable string."""
    result = []
    base_currency, quote_currency = symbol.split('/')
    
    # Extract data from results
    if "sma" in data and "sma" in indicators:
        last_sma = next((x for x in reversed(data["sma"]) if x is not None), None)
        if last_sma is not None:
            result.append(f"Simple Moving Average (14): {last_sma:.4f} {quote_currency}")
    
    if "ema" in data and "ema" in indicators:
        last_ema = next((x for x in reversed(data["ema"]) if x is not None), None)
        if last_ema is not None:
            result.append(f"Exponential Moving Average (14): {last_ema:.4f} {quote_currency}")
    
    if "vwma" in data and "vwma" in indicators:
        last_vwma = next((x for x in reversed(data["vwma"]) if x is not None), None)
        if last_vwma is not None:
            result.append(f"Volume Weighted Moving Average (14): {last_vwma:.4f} {quote_currency}")
    
    if "rsi" in data and "rsi" in indicators:
        last_rsi = next((x for x in reversed(data["rsi"]) if x is not None), None)
        if last_rsi is not None:
            rsi_signal = "OVERSOLD" if last_rsi < 30 else "OVERBOUGHT" if last_rsi > 70 else "NEUTRAL"
            result.append(f"Relative Strength Index (14): {last_rsi:.2f} [{rsi_signal}]")
    
    if "bollinger" in data and "bollinger" in indicators:
        mid = next((x for x in reversed(data["bollinger"][0]) if x is not None), None)
        upper = next((x for x in reversed(data["bollinger"][1]) if x is not None), None)
        lower = next((x for x in reversed(data["bollinger"][2]) if x is not None), None)
        
        if mid is not None and upper is not None and lower is not None:
            result.append(f"Bollinger Bands (20, 2.0):")
            result.append(f"  - Upper: {upper:.4f} {quote_currency}")
            result.append(f"  - Middle: {mid:.4f} {quote_currency}")
            result.append(f"  - Lower: {lower:.4f} {quote_currency}")
    
    if "macd" in data and "macd" in indicators:
        macd = next((x for x in reversed(data["macd"][0]) if x is not None), None)
        signal = next((x for x in reversed(data["macd"][1]) if x is not None), None)
        histogram = next((x for x in reversed(data["macd"][2]) if x is not None), None)
        
        if macd is not None and signal is not None and histogram is not None:
            macd_signal = "BULLISH" if macd > signal else "BEARISH"
            result.append(f"MACD (12, 26, 9):")
            result.append(f"  - MACD: {macd:.6f}")
            result.append(f"  - Signal: {signal:.6f}")
            result.append(f"  - Histogram: {histogram:.6f}")
            result.append(f"  - Signal: {macd_signal}")
    
    if "atr" in data and "atr" in indicators:
        last_atr = next((x for x in reversed(data["atr"]) if x is not None), None)
        if last_atr is not None:
            result.append(f"Average True Range (14): {last_atr:.4f} {quote_currency}")
    
    if "stoch" in data and "stoch" in indicators:
        last_k = next((x for x in reversed(data["stoch"][0]) if x is not None), None)
        last_d = next((x for x in reversed(data["stoch"][1]) if x is not None), None)
        
        if last_k is not None and last_d is not None:
            stoch_signal = "OVERSOLD" if last_k < 20 and last_d < 20 else "OVERBOUGHT" if last_k > 80 and last_d > 80 else "NEUTRAL"
            result.append(f"Stochastic Oscillator (14, 3):")
            result.append(f"  - %K: {last_k:.2f}")
            result.append(f"  - %D: {last_d:.2f}")
            result.append(f"  - Signal: {stoch_signal}")
    
    if "obv" in data and "obv" in indicators:
        last_obv = next((x for x in reversed(data["obv"]) if x is not None), None)
        if last_obv is not None:
            result.append(f"On Balance Volume: {last_obv:.2f}")
    
    if "patterns" in data and "patterns" in indicators:
        recent_patterns = []
        for idx, patterns in data["patterns"].items():
            if int(idx) >= len(data["close"]) - 3:  # Show only the last 3 candles' patterns
                for pattern in patterns:
                    recent_patterns.append(f"  - {pattern} (candle {idx})")
        
        if recent_patterns:
            result.append("Recent Candlestick Patterns:")
            result.extend(recent_patterns)
    
    if "support_resistance" in data and "support_resistance" in indicators:
        support = data["support_resistance"][0]
        resistance = data["support_resistance"][1]
        
        if support or resistance:
            result.append("Support and Resistance Levels:")
            
            if support:
                result.append("  Support Levels:")
                for level in support:
                    result.append(f"    - {level:.4f} {quote_currency}")
            
            if resistance:
                result.append("  Resistance Levels:")
                for level in resistance:
                    result.append(f"    - {level:.4f} {quote_currency}")
    
    return "\n".join(result)


async def format_compact_technical_analysis(data: Dict[str, Any], symbol: str) -> str:
    """Format technical analysis data into a compact, visually appealing format."""
    base_currency, quote_currency = symbol.split('/')
    
    # Get the latest price
    last_price = data["close"][-1] if data["close"] else None
    
    # Create markdown document structure
    sections = []
    
    # Add header
    sections.append(f"# TECHNICAL ANALYSIS: {symbol}")
    sections.append(f"*Last Price: {last_price:.4f} {quote_currency}*")
    
    # === TREND INDICATORS SECTION ===
    trend_indicators = []
    
    # Calculate trend analysis
    trend_score = 0
    trend_signals = []
    
    # Moving Averages
    if "sma" in data:
        last_sma = next((x for x in reversed(data["sma"]) if x is not None), None)
        if last_sma is not None and last_price is not None:
            ma_signal = "BULLISH" if last_price > last_sma else "BEARISH"
            trend_score += 1 if ma_signal == "BULLISH" else -1
            trend_signals.append(f"SMA: {ma_signal}")
            trend_indicators.append(f"SMA(14): {last_sma:.4f} [{ma_signal}]")
    
    if "ema" in data:
        last_ema = next((x for x in reversed(data["ema"]) if x is not None), None)
        if last_ema is not None and last_price is not None:
            ema_signal = "BULLISH" if last_price > last_ema else "BEARISH"
            trend_score += 1 if ema_signal == "BULLISH" else -1
            trend_signals.append(f"EMA: {ema_signal}")
            trend_indicators.append(f"EMA(14): {last_ema:.4f} [{ema_signal}]")
    
    if "macd" in data:
        macd = next((x for x in reversed(data["macd"][0]) if x is not None), None)
        signal = next((x for x in reversed(data["macd"][1]) if x is not None), None)
        histogram = next((x for x in reversed(data["macd"][2]) if x is not None), None)
        
        if macd is not None and signal is not None:
            macd_signal = "BULLISH" if macd > signal else "BEARISH"
            macd_cross = ""
            
            # Check for recent cross
            if len(data["macd"][0]) > 2 and len(data["macd"][1]) > 2:
                prev_macd = data["macd"][0][-2]
                prev_signal = data["macd"][1][-2]
                if prev_macd is not None and prev_signal is not None:
                    if macd > signal and prev_macd <= prev_signal:
                        macd_cross = "** BULLISH CROSS **"
                    elif macd < signal and prev_macd >= prev_signal:
                        macd_cross = "!! BEARISH CROSS !!"
            
            trend_score += 1 if macd_signal == "BULLISH" else -1
            trend_score += 2 if "BULLISH CROSS" in macd_cross else -2 if "BEARISH CROSS" in macd_cross else 0
            
            trend_signals.append(f"MACD: {macd_signal}")
            if macd_cross:
                trend_signals.append(macd_cross)
                
            trend_indicators.append(f"MACD: {macd:.6f} Signal: {signal:.6f} [{macd_signal}] {macd_cross}")
    
    # Overall trend assessment
    if trend_score > 2:
        trend_assessment = "STRONG UPTREND"
    elif trend_score > 0:
        trend_assessment = "MODERATE UPTREND"
    elif trend_score < -2:
        trend_assessment = "STRONG DOWNTREND"
    elif trend_score < 0:
        trend_assessment = "MODERATE DOWNTREND"
    else:
        trend_assessment = "SIDEWAYS/NEUTRAL"
    
    # === MOMENTUM INDICATORS SECTION ===
    momentum_indicators = []
    
    # Calculate momentum analysis
    momentum_score = 0
    momentum_signals = []
    
    # RSI
    if "rsi" in data:
        last_rsi = next((x for x in reversed(data["rsi"]) if x is not None), None)
        if last_rsi is not None:
            rsi_signal = "OVERSOLD" if last_rsi < 30 else "OVERBOUGHT" if last_rsi > 70 else "NEUTRAL"
            momentum_score += -2 if rsi_signal == "OVERBOUGHT" else 2 if rsi_signal == "OVERSOLD" else 0
            momentum_signals.append(f"RSI: {rsi_signal}")
            momentum_indicators.append(f"RSI(14): {last_rsi:.2f} [{rsi_signal}]")
    
    # Stochastic
    if "stoch" in data:
        last_k = next((x for x in reversed(data["stoch"][0]) if x is not None), None)
        last_d = next((x for x in reversed(data["stoch"][1]) if x is not None), None)
        
        if last_k is not None and last_d is not None:
            stoch_signal = "OVERSOLD" if last_k < 20 and last_d < 20 else "OVERBOUGHT" if last_k > 80 and last_d > 80 else "NEUTRAL"
            stoch_cross = ""
            
            # Check for crosses
            if len(data["stoch"][0]) > 2 and len(data["stoch"][1]) > 2:
                prev_k = data["stoch"][0][-2]
                prev_d = data["stoch"][1][-2]
                if prev_k is not None and prev_d is not None:
                    if last_k > last_d and prev_k <= prev_d:
                        stoch_cross = "** BULLISH CROSS **"
                    elif last_k < last_d and prev_k >= prev_d:
                        stoch_cross = "!! BEARISH CROSS !!"
            
            momentum_score += -2 if stoch_signal == "OVERBOUGHT" else 2 if stoch_signal == "OVERSOLD" else 0
            momentum_score += 1 if "BULLISH CROSS" in stoch_cross else -1 if "BEARISH CROSS" in stoch_cross else 0
            
            momentum_signals.append(f"Stochastic: {stoch_signal}")
            if stoch_cross:
                momentum_signals.append(stoch_cross)
                
            momentum_indicators.append(f"Stoch %K: {last_k:.2f} %D: {last_d:.2f} [{stoch_signal}] {stoch_cross}")
    
    # Overall momentum assessment
    if momentum_score > 2:
        momentum_assessment = "STRONG BULLISH MOMENTUM"
    elif momentum_score > 0:
        momentum_assessment = "MODERATE BULLISH MOMENTUM"
    elif momentum_score < -2:
        momentum_assessment = "STRONG BEARISH MOMENTUM"
    elif momentum_score < 0:
        momentum_assessment = "MODERATE BEARISH MOMENTUM"
    else:
        momentum_assessment = "NEUTRAL MOMENTUM"
    
    # === VOLATILITY INDICATORS SECTION ===
    volatility_indicators = []
    
    # Bollinger Bands
    if "bollinger" in data:
        mid = next((x for x in reversed(data["bollinger"][0]) if x is not None), None)
        upper = next((x for x in reversed(data["bollinger"][1]) if x is not None), None)
        lower = next((x for x in reversed(data["bollinger"][2]) if x is not None), None)
        
        if mid is not None and upper is not None and lower is not None and last_price is not None:
            bb_width = (upper - lower) / mid if mid > 0 else 0
            bb_position = (last_price - lower) / (upper - lower) if upper != lower else 0.5
            bb_signal = "UPPER_BAND" if bb_position > 0.8 else "LOWER_BAND" if bb_position < 0.2 else "MID_RANGE"
            
            bb_description = ""
            if bb_signal == "UPPER_BAND":
                bb_description = "Price at upper band (potential reversal/continuation)"
            elif bb_signal == "LOWER_BAND":
                bb_description = "Price at lower band (potential reversal/support)"
            
            volatility_indicators.append(f"BB Width: {bb_width:.4f} Position: {bb_position:.2f} [{bb_signal}]")
            if bb_description:
                volatility_indicators.append(f"BB Signal: {bb_description}")
    
    # ATR
    if "atr" in data:
        last_atr = next((x for x in reversed(data["atr"]) if x is not None), None)
        if last_atr is not None and last_price is not None:
            atr_percent = (last_atr / last_price) * 100
            volatility_label = "HIGH" if atr_percent > 5 else "MODERATE" if atr_percent > 2 else "LOW"
            volatility_indicators.append(f"ATR: {last_atr:.4f} ({atr_percent:.2f}% of price) [{volatility_label} VOLATILITY]")
    
    # === SUPPORT & RESISTANCE SECTION ===
    levels = []
    
    if "support_resistance" in data:
        support = data["support_resistance"][0]
        resistance = data["support_resistance"][1]
        
        # Find nearest levels
        nearest_support = None
        nearest_support_dist = float('inf')
        for level in support:
            if level < last_price:
                dist = (last_price - level) / last_price
                if dist < nearest_support_dist:
                    nearest_support = level
                    nearest_support_dist = dist
        
        nearest_resistance = None
        nearest_resistance_dist = float('inf')
        for level in resistance:
            if level > last_price:
                dist = (level - last_price) / last_price
                if dist < nearest_resistance_dist:
                    nearest_resistance = level
                    nearest_resistance_dist = dist
        
        # Format key levels
        if nearest_support:
            support_dist = ((nearest_support / last_price) - 1) * 100
            levels.append(f"Nearest Support: {nearest_support:.4f} ({abs(support_dist):.2f}% below)")
        
        if nearest_resistance:
            resistance_dist = ((nearest_resistance / last_price) - 1) * 100
            levels.append(f"Nearest Resistance: {nearest_resistance:.4f} ({resistance_dist:.2f}% above)")
    
    # === VOLUME ANALYSIS SECTION ===
    volume_indicators = []
    
    # On Balance Volume
    if "obv" in data and len(data["obv"]) > 1:
        last_obv = data["obv"][-1]
        prev_obv = data["obv"][-2]
        if last_obv is not None and prev_obv is not None:
            obv_change = ((last_obv / prev_obv) - 1) * 100 if prev_obv != 0 else 0
            obv_signal = "RISING" if obv_change > 0 else "FALLING" if obv_change < 0 else "STABLE"
            volume_indicators.append(f"OBV: {obv_signal} ({obv_change:.2f}% change)")
    
    # Recent volume trend
    if "volume" in data and len(data["volume"]) > 14:
        recent_vol = sum(data["volume"][-5:]) / 5 if len(data["volume"]) >= 5 else 0
        avg_vol = sum(data["volume"][-20:-5]) / 15 if len(data["volume"]) >= 20 else 0
        
        if recent_vol > 0 and avg_vol > 0:
            vol_change = ((recent_vol / avg_vol) - 1) * 100
            vol_signal = "INCREASING" if vol_change > 20 else "DECREASING" if vol_change < -20 else "STABLE"
            volume_indicators.append(f"Volume Trend: {vol_signal} ({vol_change:.2f}% vs average)")
    
    # === PATTERNS SECTION ===
    patterns_found = []
    
    if "patterns" in data:
        for idx, patterns in data["patterns"].items():
            if int(idx) >= len(data["close"]) - 3:  # Show only the last 3 candles' patterns
                for pattern in patterns:
                    patterns_found.append(f"{pattern} (candle {idx})")
    
    # === COMPILE SECTIONS ===
    
    # Add Trend Analysis Section
    sections.append("\n## TREND ANALYSIS")
    sections.append(f"**Overall: {trend_assessment}**")
    sections.append("```")
    sections.append("\n".join(trend_indicators))
    sections.append("```")
    
    # Add Momentum Analysis Section
    sections.append("\n## MOMENTUM & OSCILLATORS")
    sections.append(f"**Overall: {momentum_assessment}**")
    sections.append("```")
    sections.append("\n".join(momentum_indicators))
    sections.append("```")
    
    # Add Volatility Analysis Section
    if volatility_indicators:
        sections.append("\n## VOLATILITY METRICS")
        sections.append("```")
        sections.append("\n".join(volatility_indicators))
        sections.append("```")
    
    # Add Support/Resistance Section
    if levels:
        sections.append("\n## KEY PRICE LEVELS")
        sections.append("```")
        sections.append("\n".join(levels))
        sections.append("```")
    
    # Add Volume Analysis Section
    if volume_indicators:
        sections.append("\n## VOLUME ANALYSIS")
        sections.append("```")
        sections.append("\n".join(volume_indicators))
        sections.append("```")
    
    # Add Patterns Section
    if patterns_found:
        sections.append("\n## RECENT PATTERNS")
        sections.append("```")
        sections.append("\n".join(patterns_found))
        sections.append("```")
    
    # Add Key Trading Signals Section
    signals = []
    
    # Combine all signals
    signals.extend(trend_signals)
    signals.extend(momentum_signals)
    
    # Add overall market assessment
    if trend_score > 0 and momentum_score > 0:
        market_signal = "BULLISH"
    elif trend_score < 0 and momentum_score < 0:
        market_signal = "BEARISH"
    else:
        market_signal = "MIXED/NEUTRAL"
    
    signals.insert(0, f"MARKET BIAS: {market_signal}")
    
    # Add Key Trading Signals section
    sections.append("\n## TRADING SIGNALS")
    sections.append("```")
    sections.append("\n".join(signals))
    sections.append("```")
    
    return "\n".join(sections)

def get_exchange_schema() -> Dict[str, Any]:
    """Get the JSON schema for exchange selection."""
    return {
        "type": "string",
        "description": f"Exchange to use (supported: {', '.join(SUPPORTED_EXCHANGES.keys())})",
        "enum": list(SUPPORTED_EXCHANGES.keys()),
        "default": "binance"
    }


def format_ohlcv_data(ohlcv_data: List[List], timeframe: str) -> str:
    """Format OHLCV data into a readable string with price changes."""
    formatted_data = []

    for i, candle in enumerate(ohlcv_data):
        timestamp, open_price, high, low, close, volume = candle

        # Calculate price change from previous close if available
        price_change = ""
        if i > 0:
            prev_close = ohlcv_data[i-1][4]
            change_pct = ((close - prev_close) / prev_close) * 100
            price_change = f"Change: {change_pct:+.2f}%"

        # Format the candle data
        dt = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
        candle_str = (
            f"Time: {dt}\n"
            f"Open: {open_price:.8f}\n"
            f"High: {high:.8f}\n"
            f"Low: {low:.8f}\n"
            f"Close: {close:.8f}\n"
            f"Volume: {volume:.2f}\n"
            f"{price_change}\n"
            "---"
        )
        formatted_data.append(candle_str)

    return "\n".join(formatted_data)


def format_ohlcv_data_summary(ohlcv_data: List[List], timeframe: str) -> str:
    """Create a concise summary of OHLCV data highlighting key information only."""
    if not ohlcv_data or len(ohlcv_data) == 0:
        return "No data available"
    
    # Get first and last candles
    first_candle = ohlcv_data[0]
    last_candle = ohlcv_data[-1]
    
    # Calculate overall price change
    start_price = first_candle[4]  # Close price of first candle
    end_price = last_candle[4]     # Close price of last candle
    price_change_pct = ((end_price - start_price) / start_price) * 100
    
    # Find highest high and lowest low
    highest_high = max(candle[2] for candle in ohlcv_data)
    lowest_low = min(candle[3] for candle in ohlcv_data)
    
    # Calculate price range
    price_range_pct = ((highest_high - lowest_low) / lowest_low) * 100
    
    # Get timeframe dates
    start_date = datetime.fromtimestamp(first_candle[0]/1000).strftime('%Y-%m-%d %H:%M')
    end_date = datetime.fromtimestamp(last_candle[0]/1000).strftime('%Y-%m-%d %H:%M')
    
    # Identify significant candles (big moves)
    significant_candles = []
    for i, candle in enumerate(ohlcv_data):
        if i == 0 or i == len(ohlcv_data) - 1:
            continue  # Skip first and last which we already display
            
        timestamp, open_price, high, low, close, volume = candle
        candle_range = high - low
        candle_body = abs(close - open_price)
        
        # If candle body is more than 1% of price or has high volume
        if candle_body / open_price > 0.01 or (i > 0 and volume > 2 * ohlcv_data[i-1][5]):
            direction = "Bullish" if close > open_price else "Bearish"
            dt = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M')
            change_pct = ((close - open_price) / open_price) * 100
            significant_candles.append(f"• {dt}: {direction} move of {change_pct:.2f}% (O:{open_price:.6f} C:{close:.6f})")
    
    # Limit to at most 3 significant candles
    if len(significant_candles) > 3:
        significant_candles = significant_candles[:3]
    
    # Build summary
    result = [
        f"Period: {start_date} to {end_date}",
        f"Price Change: {price_change_pct:+.2f}% (from {start_price:.6f} to {end_price:.6f})",
        f"Range: {price_range_pct:.2f}% (Low: {lowest_low:.6f}, High: {highest_high:.6f})"
    ]
    
    if significant_candles:
        result.append("\nSignificant Moves:")
        result.extend(significant_candles)
    
    # Add trend direction
    if price_change_pct > 5:
        trend = "Strong Uptrend"
    elif price_change_pct > 2:
        trend = "Uptrend"
    elif price_change_pct < -5:
        trend = "Strong Downtrend"
    elif price_change_pct < -2:
        trend = "Downtrend"
    else:
        trend = "Sideways/Consolidation"
    
    result.append(f"\nOverall Trend: {trend}")
    
    return "\n".join(result)


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available cryptocurrency tools."""
    return [
        types.Tool(
            name="list-exchanges",
            description="List all supported cryptocurrency exchanges",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get-price",
            description="Get current price of a cryptocurrency pair from a specific exchange",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": get_exchange_schema()
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-market-summary",
            description="Get detailed market summary for a cryptocurrency pair from a specific exchange",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": get_exchange_schema()
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-historical-ohlcv",
            description="Get historical OHLCV (candlestick) data for a trading pair",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": get_exchange_schema(),
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe for candlesticks (e.g., 1m, 5m, 15m, 1h, 4h, 1d)",
                        "default": "1h",
                        "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]
                    },
                    "days_back": {
                        "type": "number",
                        "description": "Number of days of historical data to fetch (default: 7, max: 30)",
                        "default": 7,
                        "maximum": 30
                    }
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-top-volumes",
            description="Get top cryptocurrencies by trading volume from a specific exchange",
            inputSchema={
                "type": "object",
                "properties": {
                    "exchange": get_exchange_schema(),
                    "limit": {
                        "type": "number",
                        "description": "Number of pairs to return (default: 5)"
                    }
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get-price-change",
            description="Get price change statistics over different time periods",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": get_exchange_schema()
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-volume-history",
            description="Get trading volume history over time",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": get_exchange_schema(),
                    "days": {
                        "type": "number",
                        "description": "Number of days of volume history (default: 7, max: 30)",
                        "default": 7,
                        "maximum": 30
                    }
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-orderbook-analysis",
            description="Analyze order book for liquidity distribution, imbalances, and support/resistance walls",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": get_exchange_schema()
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-technical-analysis",
            description="Perform technical analysis on a trading pair",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": get_exchange_schema(),
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe for analysis (e.g., 1m, 5m, 15m, 1h, 4h, 1d)",
                        "default": "1h",
                        "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]
                    },
                    "days_back": {
                        "type": "number",
                        "description": "Number of days of historical data to analyze (default: 14, max: 30)",
                        "default": 14,
                        "maximum": 30
                    },
                    "indicators": {
                        "type": "string",
                        "description": "Comma-separated list of indicators to calculate (default: all)",
                        "default": "sma,ema,rsi,bollinger,macd,atr,stoch,obv,patterns,support_resistance"
                    }
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-token-dashboard",
            description="Get comprehensive token dashboard with market data, technical analysis, and orderbook analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair symbol (e.g., BTC/USDT, ETH/USDT)"
                    },
                    "exchange": {
                        "type": "string",
                        "description": "Exchange to use (default: binanceusdm)",
                        "default": "binanceusdm",
                        "enum": list(SUPPORTED_EXCHANGES.keys())
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe for analysis (e.g., 1m, 5m, 15m, 1h, 4h, 1d)",
                        "default": "1h",
                        "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days of historical data to analyze (default: 7, max: 30)",
                        "default": 7,
                        "maximum": 30
                    },
                    "detail_level": {
                        "type": "string",
                        "description": "Level of detail to include (low, medium, high)",
                        "default": "medium",
                        "enum": ["low", "medium", "high"]
                    }
                },
                "required": ["symbol"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle tool execution requests."""
    try:
        if name == "list-exchanges":
            exchange_list = "\n".join([f"- {ex.upper()}" for ex in SUPPORTED_EXCHANGES.keys()])
            return [
                types.TextContent(
                    type="text",
                    text=f"Supported exchanges:\n\n{exchange_list}"
                )
            ]

        # Get exchange from arguments or use default
        exchange_id = arguments.get("exchange", "binance")
        exchange = await get_exchange(exchange_id)

        if name == "get-price":
            symbol = arguments.get("symbol", "").upper()
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)
            ticker = await exchange.fetch_ticker(symbol)

            # For futures, we might want to show additional info
            if is_futures_exchange(exchange_id):
                funding_rate = "N/A"
                mark_price = "N/A"
                
                try:
                    # Try to get funding rate if available
                    funding_info = await exchange.fetch_funding_rate(symbol)
                    if funding_info and 'fundingRate' in funding_info:
                        funding_rate = f"{funding_info['fundingRate'] * 100:.4f}%"
                except Exception:
                    pass  # Skip if not available
                    
                try:
                    # Try to get mark price if available
                    mark = await exchange.fetch_mark_price(symbol)
                    if mark and 'markPrice' in mark:
                        mark_price = mark['markPrice']
                except Exception:
                    pass  # Skip if not available
                
                quote_currency = symbol.split('/')[1].split(':')[0] if '/' in symbol else 'USDT'
                return [
                    types.TextContent(
                        type="text",
                        text=f"Current price of {symbol} on {exchange_id.upper()} (PERP):\n"
                             f"Last Price: {ticker['last']} {quote_currency}\n"
                             f"Mark Price: {mark_price} {quote_currency}\n"
                             f"Funding Rate: {funding_rate}"
                    )
                ]
            else:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Current price of {symbol} on {exchange_id.upper()}: {ticker['last']} {symbol.split('/')[1]}"
                    )
                ]

        elif name == "get-market-summary":
            symbol = arguments.get("symbol", "").upper()
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)
            ticker = await exchange.fetch_ticker(symbol)

            formatted_data = await format_ticker(ticker, exchange_id)
            return [
                types.TextContent(
                    type="text",
                    text=f"Market summary for {symbol}:\n\n{formatted_data}"
                )
            ]

        elif name == "get-top-volumes":
            limit = int(arguments.get("limit", 5))
            tickers = await exchange.fetch_tickers()

            # Sort by volume and get top N
            sorted_tickers = sorted(
                tickers.values(),
                key=lambda x: float(x.get('baseVolume', 0) or 0),
                reverse=True
            )[:limit]

            formatted_results = []
            for ticker in sorted_tickers:
                formatted_data = await format_ticker(ticker, exchange_id)
                formatted_results.append(formatted_data)

            return [
                types.TextContent(
                    type="text",
                    text=f"Top {limit} pairs by volume on {exchange_id.upper()}:\n\n" + "\n".join(formatted_results)
                )
            ]

        elif name == "get-historical-ohlcv":
            symbol = arguments.get("symbol", "").upper()
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)
            timeframe = arguments.get("timeframe", "1h")
            days_back = min(int(arguments.get("days_back", 7)), 30)

            # Calculate timestamps
            since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

            # Fetch historical data
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since)

            formatted_data = format_ohlcv_data(ohlcv, timeframe)
            return [
                types.TextContent(
                    type="text",
                    text=f"Historical OHLCV data for {symbol} ({timeframe}) on {exchange_id.upper()}:\n\n{formatted_data}"
                )
            ]

        elif name == "get-price-change":
            symbol = arguments.get("symbol", "").upper()
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)

            # Get current price
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Get historical prices
            timeframes = {
                "1h": (1, "1h"),
                "24h": (1, "1d"),
                "7d": (7, "1d"),
                "30d": (30, "1d")
            }

            changes = []
            for label, (days, tf) in timeframes.items():
                since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                ohlcv = await exchange.fetch_ohlcv(symbol, tf, since=since, limit=1)
                if ohlcv:
                    start_price = ohlcv[0][1]  # Open price
                    change_pct = ((current_price - start_price) / start_price) * 100
                    changes.append(f"{label} change: {change_pct:+.2f}%")

            return [
                types.TextContent(
                    type="text",
                    text=f"Price changes for {symbol} on {exchange_id.upper()}:\n\n" + "\n".join(changes)
                )
            ]

        elif name == "get-volume-history":
            symbol = arguments.get("symbol", "").upper()
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)
            days = min(int(arguments.get("days", 7)), 30)

            # Get daily volume data
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            ohlcv = await exchange.fetch_ohlcv(symbol, "1d", since=since)

            volume_data = []
            for candle in ohlcv:
                timestamp, _, _, _, _, volume = candle
                dt = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
                volume_data.append(f"{dt}: {volume:,.2f}")

            return [
                types.TextContent(
                    type="text",
                    text=f"Daily trading volume history for {symbol} on {exchange_id.upper()}:\n\n" +
                         "\n".join(volume_data)
                )
            ]

        elif name == "get-orderbook-analysis":
            symbol = arguments.get("symbol", "").upper()
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)
            orderbook = await exchange.fetch_order_book(symbol)

            metrics = await _calculate_orderbook_metrics(orderbook)
            bid_distribution = _calculate_depth_distribution(orderbook['bids'])
            ask_distribution = _calculate_depth_distribution(orderbook['asks'])
            bid_walls = _identify_liquidity_walls(orderbook['bids'])
            ask_walls = _identify_liquidity_walls(orderbook['asks'])
            bid_cumulative_distribution = _calculate_cumulative_volume_distribution(orderbook['bids'])
            ask_cumulative_distribution = _calculate_cumulative_volume_distribution(orderbook['asks'])
            microstructure = _detect_microstructure_patterns(orderbook)

            analysis = {
                **metrics,
                "bid_distribution": bid_distribution,
                "ask_distribution": ask_distribution,
                "bid_walls": bid_walls,
                "ask_walls": ask_walls,
                "bid_cumulative_distribution": bid_cumulative_distribution,
                "ask_cumulative_distribution": ask_cumulative_distribution,
                "microstructure": microstructure,
            }

            # Use the new compact format for orderbook analysis
            formatted_data = await format_compact_orderbook_analysis(analysis, symbol)
            
            # Save detailed analysis to file for reference if needed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            result_file = os.path.join(results_dir, f"{symbol.replace('/', '_').replace(':', '_')}_orderbook_analysis_{timestamp}.json")
            
            with open(result_file, 'w') as f:
                import json
                json.dump(analysis, f, indent=2, default=str)
            
            return [
                types.TextContent(
                    type="text",
                    text=formatted_data
                )
            ]
            
        elif name == "get-technical-analysis":
            symbol = arguments.get("symbol", "").upper()
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)
            timeframe = arguments.get("timeframe", "1h")
            days_back = min(int(arguments.get("days_back", 14)), 30)
            
            # Parse requested indicators (default to all)
            indicators = arguments.get("indicators", "sma,ema,rsi,bollinger,macd,atr,stoch,obv,patterns,support_resistance")
            indicators = [ind.strip() for ind in indicators.split(",")]
            
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
            
            # Calculate requested indicators
            results = {"timestamps": timestamps, "open": opens, "high": highs, 
                      "low": lows, "close": closes, "volume": volumes}
            
            if "sma" in indicators:
                results["sma"] = ta.moving_average(closes)
                
            if "ema" in indicators:
                results["ema"] = ta.exponential_moving_average(closes)
                
            if "vwma" in indicators:
                results["vwma"] = ta.volume_weighted_ma(closes, volumes)
                
            if "rsi" in indicators:
                results["rsi"] = ta.relative_strength_index(closes)
                
            if "bollinger" in indicators:
                results["bollinger"] = ta.bollinger_bands(closes)
                
            if "macd" in indicators:
                results["macd"] = ta.macd(closes)
                
            if "atr" in indicators:
                results["atr"] = ta.average_true_range(highs, lows, closes)
                
            if "stoch" in indicators:
                results["stoch"] = ta.stochastic_oscillator(highs, lows, closes)
                
            if "obv" in indicators:
                results["obv"] = ta.on_balance_volume(closes, volumes)
                
            if "patterns" in indicators:
                results["patterns"] = ta.detect_candlestick_patterns(opens, highs, lows, closes)
                
            if "support_resistance" in indicators:
                results["support_resistance"] = ta.identify_support_resistance(highs, lows)
            
            # Format the results using the compact format
            formatted_data = await format_compact_technical_analysis(results, symbol)
            
            # Save detailed analysis to file for reference if needed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            result_file = os.path.join(results_dir, f"{symbol.replace('/', '_').replace(':', '_')}_ta_analysis_{timestamp}.json")
            
            with open(result_file, 'w') as f:
                import json
                json.dump(results, f, indent=2, default=str)
            
            return [
                types.TextContent(
                    type="text",
                    text=formatted_data
                )
            ]

        elif name == "get-token-dashboard":
            symbol = arguments.get("symbol", "BTC/USDT").upper()
            exchange_id = arguments.get("exchange", "binanceusdm")
            timeframe = arguments.get("timeframe", "1h")
            days_back = min(int(arguments.get("days_back", 7)), 30)
            detail_level = arguments.get("detail_level", "medium")
            
            # Normalize the symbol for futures exchanges
            symbol = normalize_symbol(symbol, exchange_id)
            
            # Get exchange instance
            exchange = await get_exchange(exchange_id)
            
            try:
                # Get the token dashboard data
                dashboard_data = await get_token_dashboard(
                    exchange=exchange,
                    exchange_id=exchange_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    days_back=days_back,
                    detail_level=detail_level
                )
                
                # Format the dashboard data as text
                return [
                    types.TextContent(
                        type="text",
                        text=dashboard_data["text"]
                    )
                ]
            except Exception as e:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error generating token dashboard: {str(e)}"
                    )
                ]
        else:
            raise ValueError(f"Unknown tool: {name}")

    except ccxt.BaseError as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error accessing cryptocurrency data: {str(e)}"
            )
        ]
    finally:
        # Clean up exchange connections
        for instance in exchange_instances.values():
            await instance.close()
        exchange_instances.clear()


async def main():
    """Run the server using stdin/stdout streams."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="crypto-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def run_server():
    """Wrapper to run the async main function"""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
