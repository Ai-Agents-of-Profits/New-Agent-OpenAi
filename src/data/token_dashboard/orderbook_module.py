"""
Orderbook Module

This module handles fetching and analyzing order book data for tokens.
"""

from typing import Any, Dict, List
import ccxt.async_support as ccxt
import numpy as np


async def get_orderbook_analysis(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive order book analysis.
    
    Args:
        exchange: The exchange instance
        symbol: The trading pair symbol (e.g., 'BTC/USDT')
    
    Returns:
        Dictionary with order book analysis data
    """
    # Fetch the order book data
    orderbook = await exchange.fetch_order_book(symbol)
    
    # Perform analysis on the data
    metrics = await calculate_orderbook_metrics(orderbook)
    bid_distribution = calculate_depth_distribution(orderbook['bids'])
    ask_distribution = calculate_depth_distribution(orderbook['asks'])
    bid_walls = identify_liquidity_walls(orderbook['bids'])
    ask_walls = identify_liquidity_walls(orderbook['asks'])
    bid_cumulative_distribution = calculate_cumulative_volume_distribution(orderbook['bids'])
    ask_cumulative_distribution = calculate_cumulative_volume_distribution(orderbook['asks'])
    microstructure = detect_microstructure_patterns(orderbook)
    
    # Combine all analysis
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
    
    return analysis


async def calculate_orderbook_metrics(orderbook: Dict[str, List]) -> Dict[str, Any]:
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


def calculate_depth_distribution(orders: List, bins: int = 5) -> Dict[str, float]:
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


def calculate_cumulative_volume_distribution(orders: List, price_levels: int = 10) -> Dict[str, float]:
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
            
        threshold_percent = (i + 1) * step_size
        if orders[0][0] > orders[-1][0]:  # Bids (buy orders) - descending prices
            price_threshold = best_price * (1 - threshold_percent / 100)
            included_orders = [order for order in orders if order[0] >= price_threshold]
        else:  # Asks (sell orders) - ascending prices
            price_threshold = best_price * (1 + threshold_percent / 100)
            included_orders = [order for order in orders if order[0] <= price_threshold]
        
        level_volume = sum(amount for _, amount in included_orders)
        cumulative_distribution[f"{threshold_percent:.2f}%"] = level_volume
    
    return cumulative_distribution


def identify_liquidity_walls(orders: List, threshold_factor: float = 3.0) -> List[Dict[str, Any]]:
    """Identify price levels with significantly higher liquidity than average."""
    if not orders or len(orders) < 3:
        return []
    
    # Group orders by price (since the same price may appear multiple times)
    grouped = {}
    for price, amount in orders:
        if price in grouped:
            grouped[price] += amount
        else:
            grouped[price] = amount
    
    # Convert back to list and sort by price
    processed_orders = [[price, amount] for price, amount in grouped.items()]
    if orders[0][0] > orders[-1][0]:  # If descending (bids)
        processed_orders.sort(key=lambda x: x[0], reverse=True)
    else:  # If ascending (asks)
        processed_orders.sort(key=lambda x: x[0])
    
    # Calculate average volume per level
    volumes = [amount for _, amount in processed_orders]
    avg_volume = sum(volumes) / len(volumes)
    std_volume = np.std(volumes) if len(volumes) > 1 else 0
    
    # Identify walls (unusually large liquidity)
    walls = []
    for price, amount in processed_orders:
        # Consider both absolute size and deviation from mean
        if amount > avg_volume * threshold_factor or (std_volume > 0 and amount > avg_volume + std_volume * 2):
            walls.append({
                "price": price,
                "volume": amount,
                "ratio_to_avg": amount / avg_volume if avg_volume > 0 else float('inf')
            })
    
    # Sort walls by size (largest first)
    walls.sort(key=lambda x: x['volume'], reverse=True)
    
    # Return only top 5 walls
    return walls[:5]


def detect_microstructure_patterns(orderbook: Dict[str, List]) -> Dict[str, Any]:
    """Detect market microstructure patterns in order book data."""
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {"error": "Invalid order book data"}
    
    bids = orderbook['bids']
    asks = orderbook['asks']
    
    # Placeholder for detected patterns
    patterns = {}
    
    # 1. Detect iceberg orders (multiple orders at same price level)
    bid_prices = [price for price, _ in bids]
    ask_prices = [price for price, _ in asks]
    
    # Count duplicates
    bid_duplicates = {}
    ask_duplicates = {}
    for price in bid_prices:
        if price in bid_duplicates:
            bid_duplicates[price] += 1
        else:
            bid_duplicates[price] = 1
    
    for price in ask_prices:
        if price in ask_duplicates:
            ask_duplicates[price] += 1
        else:
            ask_duplicates[price] = 1
    
    # Filter to only levels with multiple orders
    bid_icebergs = {price: count for price, count in bid_duplicates.items() if count > 3}
    ask_icebergs = {price: count for price, count in ask_duplicates.items() if count > 3}
    
    patterns["possible_iceberg_orders"] = {
        "bids": bid_icebergs,
        "asks": ask_icebergs
    }
    
    # 2. Detect price clustering (orders clustered at specific price levels)
    # For example, round numbers or psychological levels
    bid_clustering = []
    ask_clustering = []
    
    # Check for round numbers
    for price, _ in bids:
        is_round = False
        price_str = str(price)
        if price_str.endswith('00'):
            is_round = True
        elif price_str.endswith('50') or price_str.endswith('000'):
            is_round = True
            
        if is_round:
            bid_clustering.append(price)
    
    for price, _ in asks:
        is_round = False
        price_str = str(price)
        if price_str.endswith('00'):
            is_round = True
        elif price_str.endswith('50') or price_str.endswith('000'):
            is_round = True
            
        if is_round:
            ask_clustering.append(price)
    
    patterns["price_clustering"] = {
        "bids": bid_clustering[:5],  # Top 5 only
        "asks": ask_clustering[:5]   # Top 5 only
    }
    
    # 3. Detect spoofing patterns (large orders at edges of book)
    # This is just a heuristic, real spoofing detection needs time-series data
    if len(bids) > 5:
        far_bids = bids[-5:]
        potential_spoof_bids = [
            {"price": price, "amount": amount} 
            for price, amount in far_bids 
            if amount > sum(a for _, a in bids[:5]) / 5
        ]
    else:
        potential_spoof_bids = []
        
    if len(asks) > 5:
        far_asks = asks[-5:]
        potential_spoof_asks = [
            {"price": price, "amount": amount} 
            for price, amount in far_asks 
            if amount > sum(a for _, a in asks[:5]) / 5
        ]
    else:
        potential_spoof_asks = []
    
    patterns["potential_large_edge_orders"] = {
        "bids": potential_spoof_bids,
        "asks": potential_spoof_asks
    }
    
    return patterns
