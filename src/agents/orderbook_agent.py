"""
Orderbook analysis agent for analyzing market depth and liquidity.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import asyncio
import numpy as np
from collections import defaultdict

from agents import Agent, Tool
from pydantic import BaseModel, Field

from .base_agent import BaseMarketAgent
from ..exchange.connector import ExchangeConnector, OrderbookData
from ..data.processor import DataProcessor

logger = logging.getLogger(__name__)


class OrderbookAnalysisRequest(BaseModel):
    """Model for orderbook analysis requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: Optional[str] = Field(None, description="Exchange ID (default: system's default exchange)")
    depth: Optional[int] = Field(100, description="Depth of orderbook to analyze")


class OrderbookAnalysisAgent(BaseMarketAgent):
    """
    Specialized agent for analyzing orderbook data.
    Focuses on market depth, liquidity distribution, and identifying support/resistance levels.
    """
    
    def __init__(self):
        """Initialize the orderbook analysis agent."""
        super().__init__(
            name="Orderbook Analysis Agent",
            description="A specialized agent for analyzing market depth, liquidity, and orderbook patterns."
        )
        
        # Register tools
        self._register_tools()
    
    def _get_instructions(self) -> str:
        """
        Get specific instructions for the orderbook analysis agent.
        
        Returns:
            String with agent instructions
        """
        return """
        You are the Orderbook Analysis Agent, a specialized agent for analyzing cryptocurrency market orderbooks,
        liquidity, and market depth.
        
        Your primary responsibility is to analyze orderbook data to provide insights about:
        
        1. **Market Depth Analysis**: Evaluate the depth and structure of order books
        2. **Liquidity Assessment**: Assess available liquidity at different price levels
        3. **Support and Resistance**: Identify key price levels with significant order density
        4. **Order Imbalances**: Detect imbalances between buy and sell orders
        5. **Market Microstructure**: Analyze patterns in order placement and market maker behavior
        6. **Liquidity Walls**: Find areas with significant order accumulation that may act as price barriers
        7. **Distribution Analysis**: Understand how orders are distributed throughout the price range
        8. **Market Maker Activity**: Detect patterns indicating market maker behavior
        
        ### Your Capabilities:
        
        - Analyze real-time orderbook data from various exchanges
        - Identify key support and resistance levels based on order clusters
        - Detect liquidity walls and gaps that may affect price movement
        - Calculate buy/sell ratios and order imbalances
        - Provide visualizations of orderbook structure when helpful
        - Analyze cumulative volume distribution to understand market depth
        - Identify potential market manipulation patterns
        - Recognize microstructure patterns for short-term price predictions
        
        ### Communication Guidelines:
        
        1. Present numerical data with appropriate precision
        2. Use markdown formatting for clarity
        3. Explain the significance of findings for traders
        4. Highlight unusual patterns or anomalies
        5. Be specific about timeframes and market conditions
        
        Use your tools to fetch and analyze orderbook data from various exchanges.
        Always specify which trading pair and exchange you're analyzing.
        """
    
    def _register_tools(self) -> None:
        """Register tools for the orderbook analysis agent."""
        # Tool for fetching and analyzing orderbook data
        self.add_tool(
            Tool(
                name="analyze_orderbook",
                description="Fetch and analyze orderbook data for a specific trading pair",
                function=self._analyze_orderbook,
                parameters=[
                    OrderbookAnalysisRequest
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
    
    def _calculate_depth_distribution(self, orders: List[List[float]], num_bins: int = 10) -> Dict[str, float]:
        """
        Calculate the distribution of orders across price ranges.
        
        Args:
            orders: List of [price, size] pairs
            num_bins: Number of bins to divide the price range into
            
        Returns:
            Dictionary mapping price range names to percentage of total volume
        """
        if not orders:
            return {}
            
        # Extract prices and sizes
        prices = np.array([order[0] for order in orders])
        sizes = np.array([order[1] for order in orders])
        
        # Calculate total volume
        total_volume = np.sum(sizes)
        if total_volume == 0:
            return {}
            
        # Create bins based on price range
        min_price = np.min(prices)
        max_price = np.max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:  # All orders at same price
            return {"single_price": 100.0}
            
        # Create bins
        bins = np.linspace(min_price, max_price, num_bins + 1)
        
        # Distribute orders into bins
        indices = np.digitize(prices, bins) - 1
        
        # Calculate volume in each bin
        bin_volumes = np.zeros(num_bins)
        for i, size in enumerate(sizes):
            if indices[i] < num_bins:  # Ensure we don't go out of bounds
                bin_volumes[indices[i]] += size
        
        # Convert to percentages
        bin_percentages = (bin_volumes / total_volume) * 100
        
        # Format result
        result = {}
        for i in range(num_bins):
            bin_name = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
            result[bin_name] = round(bin_percentages[i], 2)
            
        return result
    
    def _identify_liquidity_walls(self, orders: List[List[float]], threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
        """
        Identify liquidity walls (significant order clusters).
        
        Args:
            orders: List of [price, size] pairs
            threshold_factor: Factor above average to consider a wall
            
        Returns:
            List of dictionaries describing liquidity walls
        """
        if not orders or len(orders) < 3:
            return []
            
        sizes = np.array([order[1] for order in orders])
        prices = np.array([order[0] for order in orders])
        
        # Calculate moving average of sizes (3 points)
        moving_avg = np.convolve(sizes, np.ones(3)/3, mode='valid')
        
        # Find average size
        avg_size = np.mean(sizes)
        
        # Find walls (points where size exceeds average by threshold_factor)
        wall_indices = []
        for i in range(len(moving_avg)):
            if moving_avg[i] > avg_size * threshold_factor:
                wall_indices.append(i + 1)  # +1 due to valid mode truncating first and last points
        
        # Group adjacent wall indices
        walls = []
        if wall_indices:
            current_wall = [wall_indices[0]]
            
            for i in range(1, len(wall_indices)):
                if wall_indices[i] == wall_indices[i-1] + 1:
                    current_wall.append(wall_indices[i])
                else:
                    # Process completed wall
                    start_idx = current_wall[0]
                    end_idx = current_wall[-1]
                    wall_volume = sum(sizes[start_idx:end_idx+1])
                    wall_price_start = prices[start_idx]
                    wall_price_end = prices[end_idx]
                    
                    walls.append({
                        "price_range": [round(wall_price_start, 4), round(wall_price_end, 4)],
                        "average_price": round(np.mean([wall_price_start, wall_price_end]), 4),
                        "total_volume": round(wall_volume, 4),
                        "percentage_of_visible": round((wall_volume / np.sum(sizes)) * 100, 2)
                    })
                    
                    # Start new wall
                    current_wall = [wall_indices[i]]
            
            # Process the last wall
            start_idx = current_wall[0]
            end_idx = current_wall[-1]
            wall_volume = sum(sizes[start_idx:end_idx+1])
            wall_price_start = prices[start_idx]
            wall_price_end = prices[end_idx]
            
            walls.append({
                "price_range": [round(wall_price_start, 4), round(wall_price_end, 4)],
                "average_price": round(np.mean([wall_price_start, wall_price_end]), 4),
                "total_volume": round(wall_volume, 4),
                "percentage_of_visible": round((wall_volume / np.sum(sizes)) * 100, 2)
            })
        
        return walls
    
    def _calculate_cumulative_volume_distribution(self, orders: List[List[float]], num_points: int = 10) -> List[Dict[str, float]]:
        """
        Calculate the cumulative volume distribution as price moves away from the best.
        
        Args:
            orders: List of [price, size] pairs
            num_points: Number of points in the distribution curve
            
        Returns:
            List of dictionaries with price distances and cumulative percentages
        """
        if not orders:
            return []
            
        # Extract prices and sizes
        prices = np.array([order[0] for order in orders])
        sizes = np.array([order[1] for order in orders])
        
        # Calculate total volume
        total_volume = np.sum(sizes)
        if total_volume == 0:
            return []
            
        # Reference price (best bid/ask)
        reference_price = prices[0]
        
        # Calculate price distances
        price_distances = np.abs(prices - reference_price) / reference_price * 100  # percentage distance
        
        # Sort by distance
        sorted_indices = np.argsort(price_distances)
        sorted_distances = price_distances[sorted_indices]
        sorted_sizes = sizes[sorted_indices]
        
        # Calculate cumulative volumes
        cumulative_sizes = np.cumsum(sorted_sizes)
        cumulative_percentages = (cumulative_sizes / total_volume) * 100
        
        # Sample at regular intervals
        result = []
        max_distance = sorted_distances[-1] if len(sorted_distances) > 0 else 0
        step = max_distance / num_points if max_distance > 0 else 0
        
        if step > 0:
            for i in range(num_points + 1):
                distance = i * step
                
                # Find the closest point in our sorted array
                idx = np.searchsorted(sorted_distances, distance)
                if idx >= len(cumulative_percentages):
                    cum_pct = 100.0
                else:
                    cum_pct = cumulative_percentages[idx]
                
                result.append({
                    "distance_pct": round(distance, 2),
                    "cumulative_pct": round(cum_pct, 2)
                })
        
        return result
    
    def _detect_microstructure_patterns(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect market microstructure patterns in the orderbook.
        
        Args:
            orderbook: Dictionary with 'bids' and 'asks'
            
        Returns:
            Dictionary with detected patterns
        """
        bids = orderbook.bids if hasattr(orderbook, 'bids') else orderbook.get('bids', [])
        asks = orderbook.asks if hasattr(orderbook, 'asks') else orderbook.get('asks', [])
        
        if not bids or not asks:
            return {"patterns": [], "summary": "Insufficient data for microstructure analysis"}
            
        # Extract the first few levels
        top_bids = bids[:5] if len(bids) >= 5 else bids
        top_asks = asks[:5] if len(asks) >= 5 else asks
        
        patterns = []
        
        # 1. Check for iceberg orders (large orders hidden behind small orders)
        bid_sizes = [order[1] for order in top_bids]
        ask_sizes = [order[1] for order in top_asks]
        
        if len(bid_sizes) >= 2 and bid_sizes[1] > bid_sizes[0] * 2:
            patterns.append("Potential iceberg buy orders detected")
            
        if len(ask_sizes) >= 2 and ask_sizes[1] > ask_sizes[0] * 2:
            patterns.append("Potential iceberg sell orders detected")
        
        # 2. Check for spoofing (large orders that may be removed quickly)
        if len(bid_sizes) >= 1 and bid_sizes[0] > sum(bid_sizes[1:]) * 2 if len(bid_sizes) > 1 else True:
            patterns.append("Potential spoofing on buy side (large bid may be removed)")
            
        if len(ask_sizes) >= 1 and ask_sizes[0] > sum(ask_sizes[1:]) * 2 if len(ask_sizes) > 1 else True:
            patterns.append("Potential spoofing on sell side (large ask may be removed)")
        
        # 3. Check for layering (multiple orders at different price levels)
        bid_price_gaps = [abs(top_bids[i][0] - top_bids[i+1][0]) for i in range(len(top_bids)-1)] if len(top_bids) > 1 else []
        ask_price_gaps = [abs(top_asks[i][0] - top_asks[i+1][0]) for i in range(len(top_asks)-1)] if len(top_asks) > 1 else []
        
        if bid_price_gaps and all(gap < np.mean(bid_price_gaps) * 0.5 for gap in bid_price_gaps):
            patterns.append("Layering pattern detected on buy side (evenly spaced bids)")
            
        if ask_price_gaps and all(gap < np.mean(ask_price_gaps) * 0.5 for gap in ask_price_gaps):
            patterns.append("Layering pattern detected on sell side (evenly spaced asks)")
        
        # 4. Check for order book imbalance
        total_bid_volume = sum(order[1] for order in top_bids)
        total_ask_volume = sum(order[1] for order in top_asks)
        
        imbalance_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
        
        if imbalance_ratio > 3:
            patterns.append(f"Strong buy imbalance detected (bid/ask ratio: {round(imbalance_ratio, 2)})")
        elif imbalance_ratio < 0.33:
            patterns.append(f"Strong sell imbalance detected (bid/ask ratio: {round(imbalance_ratio, 2)})")
        
        # Summary
        if patterns:
            summary = "Multiple microstructure patterns detected suggesting potential price action"
        else:
            summary = "No significant microstructure patterns detected"
        
        return {
            "patterns": patterns,
            "summary": summary,
            "metrics": {
                "bid_ask_imbalance": round(imbalance_ratio, 4) if imbalance_ratio != float('inf') else None,
                "avg_bid_gap": round(np.mean(bid_price_gaps), 6) if bid_price_gaps else None,
                "avg_ask_gap": round(np.mean(ask_price_gaps), 6) if ask_price_gaps else None
            }
        }
    
    async def _analyze_orderbook(self, params: OrderbookAnalysisRequest) -> Dict[str, Any]:
        """
        Fetch and analyze orderbook data with enhanced features.
        
        Args:
            params: Parameters for the analysis
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            # Get exchange connector
            async with ExchangeConnector(params.exchange) as connector:
                # Fetch orderbook data
                orderbook = await connector.fetch_orderbook(params.symbol, params.depth)
                
                # Process orderbook data
                analysis = DataProcessor.analyze_orderbook(orderbook)
                
                # Calculate enhanced metrics
                bid_distribution = self._calculate_depth_distribution(orderbook.bids)
                ask_distribution = self._calculate_depth_distribution(orderbook.asks)
                bid_walls = self._identify_liquidity_walls(orderbook.bids)
                ask_walls = self._identify_liquidity_walls(orderbook.asks)
                bid_cumulative_distribution = self._calculate_cumulative_volume_distribution(orderbook.bids)
                ask_cumulative_distribution = self._calculate_cumulative_volume_distribution(orderbook.asks)
                microstructure = self._detect_microstructure_patterns(orderbook)
                
                # Format the response
                result = {
                    "symbol": orderbook.symbol,
                    "exchange": orderbook.exchange,
                    "timestamp": orderbook.timestamp,
                    "datetime": orderbook.datetime,
                    "analysis": {
                        "buy_sell_ratio": round(analysis.buy_sell_ratio, 4),
                        "liquidity_distribution": {
                            k: round(v * 100, 2) for k, v in analysis.liquidity_distribution.items()
                        },
                        "support_levels": [round(level, 4) for level in analysis.support_levels],
                        "resistance_levels": [round(level, 4) for level in analysis.resistance_levels],
                        "imbalance_points": analysis.imbalance_points
                    },
                    "market_structure": {
                        "bid_count": len(orderbook.bids),
                        "ask_count": len(orderbook.asks),
                        "bid_total_size": round(sum(size for _, size in orderbook.bids), 8),
                        "ask_total_size": round(sum(size for _, size in orderbook.asks), 8),
                        "spread": round(orderbook.asks[0][0] - orderbook.bids[0][0], 8) if orderbook.asks and orderbook.bids else None,
                        "spread_percentage": round(((orderbook.asks[0][0] - orderbook.bids[0][0]) / orderbook.bids[0][0]) * 100, 4) if orderbook.asks and orderbook.bids else None,
                    },
                    "best_levels": {
                        "best_bid": {"price": round(orderbook.bids[0][0], 8), "size": round(orderbook.bids[0][1], 8)} if orderbook.bids else None,
                        "best_ask": {"price": round(orderbook.asks[0][0], 8), "size": round(orderbook.asks[0][1], 8)} if orderbook.asks else None,
                    },
                    "depth_analysis": {
                        "bid_distribution": bid_distribution,
                        "ask_distribution": ask_distribution,
                        "bid_cumulative_distribution": bid_cumulative_distribution,
                        "ask_cumulative_distribution": ask_cumulative_distribution,
                    },
                    "liquidity_walls": {
                        "bid_walls": bid_walls,
                        "ask_walls": ask_walls
                    },
                    "microstructure_patterns": microstructure
                }
                
                # Add interpretation
                result["interpretation"] = self._interpret_orderbook_analysis(result)
                
                # Calculate current price
                if orderbook.bids and orderbook.asks:
                    result["current_price"] = round((orderbook.bids[0][0] + orderbook.asks[0][0]) / 2, 4)
                    
                # Calculate max buy/sell impact (price impact if top X% of orders executed)
                if orderbook.bids and orderbook.asks:
                    bid_total = sum(size for _, size in orderbook.bids)
                    ask_total = sum(size for _, size in orderbook.asks)
                    
                    # Calculate 5% impact
                    bid_5pct = bid_total * 0.05
                    ask_5pct = ask_total * 0.05
                    
                    bid_5pct_impact = self._calculate_price_impact(orderbook.bids, bid_5pct)
                    ask_5pct_impact = self._calculate_price_impact(orderbook.asks, ask_5pct)
                    
                    result["price_impact"] = {
                        "buy_5pct": {
                            "size": round(ask_5pct, 8),
                            "price_change": round(ask_5pct_impact, 4) if ask_5pct_impact is not None else None,
                            "percentage": round((ask_5pct_impact / orderbook.asks[0][0]) * 100, 4) if ask_5pct_impact is not None else None
                        },
                        "sell_5pct": {
                            "size": round(bid_5pct, 8),
                            "price_change": round(bid_5pct_impact, 4) if bid_5pct_impact is not None else None,
                            "percentage": round((bid_5pct_impact / orderbook.bids[0][0]) * 100, 4) if bid_5pct_impact is not None else None
                        }
                    }
                
                # Add overall market quality score (0-100)
                quality_score = self._calculate_market_quality_score(result)
                result["market_quality"] = {
                    "score": quality_score,
                    "rating": "Excellent" if quality_score >= 80 else
                            "Good" if quality_score >= 60 else
                            "Moderate" if quality_score >= 40 else
                            "Poor" if quality_score >= 20 else "Very Poor",
                    "liquidity": "High" if quality_score >= 70 else "Medium" if quality_score >= 40 else "Low"
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error analyzing orderbook: {e}")
            return {
                "error": str(e),
                "symbol": params.symbol,
                "exchange": params.exchange
            }
    
    def _calculate_price_impact(self, orders: List[List[float]], target_volume: float) -> Optional[float]:
        """
        Calculate price impact of executing orders up to target volume.
        
        Args:
            orders: List of [price, size] pairs
            target_volume: Target volume to execute
            
        Returns:
            Price impact (difference from best price) or None if not enough volume
        """
        if not orders or target_volume <= 0:
            return None
            
        best_price = orders[0][0]
        cumulative_volume = 0
        
        for price, size in orders:
            cumulative_volume += size
            if cumulative_volume >= target_volume:
                return abs(price - best_price)
                
        return None
    
    def _calculate_market_quality_score(self, analysis: Dict[str, Any]) -> int:
        """
        Calculate an overall market quality score (0-100).
        
        Args:
            analysis: Dictionary with analysis results
            
        Returns:
            Market quality score (0-100)
        """
        score = 50  # Start with neutral score
        
        # Check spread
        if "market_structure" in analysis and "spread_percentage" in analysis["market_structure"]:
            spread_pct = analysis["market_structure"]["spread_percentage"]
            if spread_pct is not None:
                if spread_pct < 0.1:
                    score += 15
                elif spread_pct < 0.3:
                    score += 10
                elif spread_pct < 0.5:
                    score += 5
                elif spread_pct > 2:
                    score -= 15
                elif spread_pct > 1:
                    score -= 10
                elif spread_pct > 0.7:
                    score -= 5
        
        # Check market depth (bid/ask totals)
        if "market_structure" in analysis:
            bid_total = analysis["market_structure"].get("bid_total_size", 0)
            ask_total = analysis["market_structure"].get("ask_total_size", 0)
            
            # Higher volumes generally indicate better liquidity
            total_volume = bid_total + ask_total
            if total_volume > 1000:
                score += 15
            elif total_volume > 500:
                score += 10
            elif total_volume > 100:
                score += 5
            elif total_volume < 10:
                score -= 15
            elif total_volume < 50:
                score -= 10
        
        # Check bid/ask ratio
        if "analysis" in analysis and "buy_sell_ratio" in analysis["analysis"]:
            ratio = analysis["analysis"]["buy_sell_ratio"]
            # Balanced order books (ratio close to 1) are generally healthier
            if 0.8 <= ratio <= 1.2:
                score += 10
            elif ratio < 0.5 or ratio > 2:
                score -= 10
        
        # Check for manipulation patterns
        if "microstructure_patterns" in analysis:
            patterns = analysis["microstructure_patterns"].get("patterns", [])
            if any("spoofing" in p.lower() for p in patterns):
                score -= 15
            if any("layering" in p.lower() for p in patterns):
                score -= 10
        
        # Cap the score between 0 and 100
        score = max(0, min(100, score))
        
        return score
    
    def _interpret_orderbook_analysis(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret orderbook analysis results.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Dictionary with interpretations
        """
        interpretations = {}
        
        # Interpret buy/sell ratio
        buy_sell_ratio = analysis["analysis"]["buy_sell_ratio"]
        if buy_sell_ratio > 1.2:
            interpretations["buy_sell_ratio"] = "Strong buying pressure. Significantly more buy orders than sell orders."
        elif buy_sell_ratio > 1.05:
            interpretations["buy_sell_ratio"] = "Moderate buying pressure. More buy orders than sell orders."
        elif buy_sell_ratio < 0.8:
            interpretations["buy_sell_ratio"] = "Strong selling pressure. Significantly more sell orders than buy orders."
        elif buy_sell_ratio < 0.95:
            interpretations["buy_sell_ratio"] = "Moderate selling pressure. More sell orders than buy orders."
        else:
            interpretations["buy_sell_ratio"] = "Balanced order book. Roughly equal buy and sell orders."
        
        # Interpret liquidity distribution
        liquidity_dist = analysis["analysis"]["liquidity_distribution"]
        top_quartile = max(liquidity_dist.items(), key=lambda x: x[1])
        if top_quartile[1] > 50:
            interpretations["liquidity_distribution"] = f"Liquidity concentration in {top_quartile[0]} quartile ({top_quartile[1]}% of total liquidity)."
        else:
            interpretations["liquidity_distribution"] = "Relatively balanced liquidity distribution across price ranges."
        
        # Interpret support and resistance levels
        support_levels = analysis["analysis"]["support_levels"]
        resistance_levels = analysis["analysis"]["resistance_levels"]
        
        if support_levels:
            interpretations["support_levels"] = f"Key support levels identified at: {', '.join([str(round(level, 4)) for level in support_levels])}."
        else:
            interpretations["support_levels"] = "No significant support levels identified."
            
        if resistance_levels:
            interpretations["resistance_levels"] = f"Key resistance levels identified at: {', '.join([str(round(level, 4)) for level in resistance_levels])}."
        else:
            interpretations["resistance_levels"] = "No significant resistance levels identified."
        
        # Interpret spread
        if "spread_percentage" in analysis["market_structure"] and analysis["market_structure"]["spread_percentage"] is not None:
            spread_pct = analysis["market_structure"]["spread_percentage"]
            if spread_pct > 1.0:
                interpretations["spread"] = f"Wide spread ({spread_pct}%) indicating low liquidity or high volatility."
            elif spread_pct > 0.5:
                interpretations["spread"] = f"Moderate spread ({spread_pct}%)."
            else:
                interpretations["spread"] = f"Tight spread ({spread_pct}%) indicating high liquidity or market maker competition."
        
        # Interpret imbalance points
        imbalance_points = analysis["analysis"]["imbalance_points"]
        if imbalance_points:
            interpretations["imbalance_points"] = f"Found {len(imbalance_points)} significant order imbalances that may create price action volatility."
        else:
            interpretations["imbalance_points"] = "No significant order imbalances detected."
        
        # Interpret liquidity walls
        if "liquidity_walls" in analysis and analysis["liquidity_walls"]:
            walls = analysis["liquidity_walls"]
            wall_count = len(walls)
            
            largest_wall = max(walls, key=lambda x: x["total_volume"])
            largest_wall_price = largest_wall["average_price"]
            largest_wall_pct = largest_wall["percentage_of_visible"]
            
            interpretations["liquidity_walls"] = f"Detected {wall_count} significant liquidity wall(s). The largest wall is at price {largest_wall_price} with {largest_wall_pct}% of visible liquidity, which may act as a strong support/resistance level."
        else:
            interpretations["liquidity_walls"] = "No significant liquidity walls detected."
        
        # Interpret depth distribution
        if "depth_distribution" in analysis and analysis["depth_distribution"]:
            dist = analysis["depth_distribution"]
            # Find the price range with the highest concentration
            if dist:
                max_range = max(dist.items(), key=lambda x: x[1])
                interpretations["depth_distribution"] = f"Highest order concentration ({max_range[1]}%) is in the price range {max_range[0]}."
            else:
                interpretations["depth_distribution"] = "Order book shows uniform distribution with no significant concentration."
        
        # Interpret cumulative volume distribution
        if "cumulative_volume_distribution" in analysis and analysis["cumulative_volume_distribution"]:
            cvd = analysis["cumulative_volume_distribution"]
            # Find the point where we reach close to 50% of volume
            fifty_pct_point = None
            for point in cvd:
                if point["cumulative_pct"] >= 50:
                    fifty_pct_point = point
                    break
                    
            if fifty_pct_point:
                interpretations["cumulative_volume"] = f"50% of total volume is within {fifty_pct_point['distance_pct']}% of the best price, indicating {'good' if fifty_pct_point['distance_pct'] < 1.0 else 'moderate' if fifty_pct_point['distance_pct'] < 3.0 else 'poor'} market depth."
            else:
                interpretations["cumulative_volume"] = "Unable to determine the price distance for 50% cumulative volume."
                
        # Interpret microstructure patterns
        if "microstructure_patterns" in analysis and analysis["microstructure_patterns"]:
            patterns = analysis["microstructure_patterns"]
            
            if patterns.get("patterns"):
                pattern_count = len(patterns["patterns"])
                pattern_list = ", ".join(patterns["patterns"])
                
                interpretations["microstructure"] = f"Detected {pattern_count} microstructure pattern(s): {pattern_list}. {patterns.get('summary', '')}"
            else:
                interpretations["microstructure"] = patterns.get("summary", "No significant microstructure patterns detected.")
                
            # Add metrics-based interpretation
            metrics = patterns.get("metrics", {})
            if metrics.get("bid_ask_imbalance") is not None:
                imbalance = metrics["bid_ask_imbalance"]
                if imbalance > 1.5:
                    interpretations["bid_ask_imbalance"] = f"Strong buying pressure with bid/ask imbalance of {imbalance}."
                elif imbalance < 0.67:
                    interpretations["bid_ask_imbalance"] = f"Strong selling pressure with bid/ask imbalance of {imbalance}."
        
        # Overall market assessment
        overall_assessment = []
        
        # Assess based on buy/sell ratio
        if buy_sell_ratio > 1.1:
            overall_assessment.append("buying pressure")
        elif buy_sell_ratio < 0.9:
            overall_assessment.append("selling pressure")
        else:
            overall_assessment.append("balanced order flow")
            
        # Assess based on liquidity walls
        if "liquidity_walls" in analysis and analysis["liquidity_walls"]:
            walls = analysis["liquidity_walls"]
            if walls:
                # Check if walls are mostly above or below current price
                current_price = (analysis["best_levels"]["best_bid"]["price"] + analysis["best_levels"]["best_ask"]["price"]) / 2 if analysis["best_levels"]["best_bid"] and analysis["best_levels"]["best_ask"] else None
                
                if current_price:
                    walls_above = sum(1 for wall in walls if wall["average_price"] > current_price)
                    walls_below = sum(1 for wall in walls if wall["average_price"] < current_price)
                    
                    if walls_above > walls_below * 2:
                        overall_assessment.append("strong resistance overhead")
                    elif walls_below > walls_above * 2:
                        overall_assessment.append("strong support below")
        
        # Assess based on microstructure
        if "microstructure_patterns" in analysis and analysis["microstructure_patterns"] and analysis["microstructure_patterns"].get("patterns"):
            patterns = analysis["microstructure_patterns"]["patterns"]
            if any("spoofing" in p.lower() for p in patterns):
                overall_assessment.append("potential market manipulation")
                
            if any("iceberg" in p.lower() for p in patterns):
                overall_assessment.append("hidden liquidity")
        
        interpretations["overall_assessment"] = f"Market shows {', '.join(overall_assessment)}."
        
        return interpretations
