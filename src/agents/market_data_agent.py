"""
Market Data Agent for fetching and processing cryptocurrency market summary data.
"""
import logging
from typing import Dict, List, Any
import asyncio
from datetime import datetime

from agents import Agent, Tool
from pydantic import BaseModel, Field

from .base_agent import BaseMarketAgent
from ..exchange.connector import ExchangeConnector
from ..models.market_data import MarketSummary

logger = logging.getLogger(__name__)


class MarketDataRequest(BaseModel):
    """Model for market data requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: str = Field(None, description="Exchange ID (default: binanceusdm)")


class MarketDataAgent(BaseMarketAgent):
    """
    Specialized agent for retrieving and analyzing market summary data.
    Focuses on current prices, trading volumes, and basic market metrics.
    """
    
    def __init__(self, connector=None):
        """
        Initialize the market data agent.
        
        Args:
            connector: Optional ExchangeConnector instance. If provided, will use this
                       instead of creating a new one for each request.
        """
        super().__init__(
            name="Market Data Agent",
            description="A specialized agent for retrieving and analyzing cryptocurrency market data."
        )
        
        # Store connector if provided
        self.connector = connector
        
        # Register tools
        self._register_tools()
    
    def _get_instructions(self) -> str:
        """
        Get specific instructions for the market data agent.
        
        Returns:
            String with agent instructions
        """
        return """
        You are the Market Data Agent, a specialized agent for retrieving and analyzing cryptocurrency market data.
        
        Your primary responsibility is to provide market summary information including:
        
        1. **Current Prices**: Latest price information for trading pairs
        2. **Trading Metrics**: Bid, ask, spread, and 24-hour trading statistics
        3. **Volume Analysis**: Trading volume information and patterns
        4. **Market Overview**: Comprehensive market summaries for specific pairs
        
        ### Your Capabilities:
        
        - Retrieve real-time market data from various exchanges
        - Report current prices and price changes
        - Analyze trading volumes and market activity
        - Provide detailed market summaries with key metrics
        
        ### Communication Guidelines:
        
        1. Present numerical data with appropriate precision
        2. Use markdown formatting for clarity
        3. Highlight significant market movements or anomalies
        4. Be specific about timeframes and market conditions
        5. Format prices according to asset precision standards
        
        Use your tools to fetch market data from various exchanges.
        Always specify which trading pair and exchange you're analyzing.
        """
    
    def _register_tools(self) -> None:
        """Register tools for the market data agent."""
        # Tool for fetching market summary data
        self.add_tool(
            Tool(
                name="get_market_summary",
                description="Get a detailed market summary for a cryptocurrency pair",
                function=self._get_market_summary,
                parameters=[
                    MarketDataRequest
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
    
    async def _get_market_summary(self, params: MarketDataRequest) -> Dict[str, Any]:
        """
        Get a detailed market summary for a cryptocurrency pair.
        
        Args:
            params: Parameters for the request
            
        Returns:
            Dictionary with market summary information
        """
        try:
            # Set default exchange if not provided
            exchange = params.exchange or "binanceusdm"
            
            # Use existing connector or create a new one
            if self.connector:
                # Use the existing connector
                ticker = await self.connector.fetch_ticker(params.symbol)
                exchange_id = self.connector.exchange_id
            else:
                # Create a new connector
                async with ExchangeConnector(exchange) as connector:
                    ticker = await connector.fetch_ticker(params.symbol)
                    exchange_id = connector.exchange_id
            
            # Safely extract values with defaults for missing fields
            last_price = float(ticker.get('last', 0.0))
            bid = float(ticker.get('bid', 0.0))
            ask = float(ticker.get('ask', 0.0))
            volume_24h = float(ticker.get('quoteVolume', ticker.get('volume', 0.0)))
            percent_change = float(ticker.get('percentage', ticker.get('change', 0.0)))
            high_24h = float(ticker.get('high', 0.0))
            low_24h = float(ticker.get('low', 0.0))
            timestamp = ticker.get('timestamp', int(datetime.now().timestamp() * 1000))
            
            # Format the response
            result = {
                "symbol": params.symbol,
                "exchange": exchange_id,
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp / 1000).isoformat(),
                "price_data": {
                    "last_price": round(last_price, 8),
                    "bid": round(bid, 8),
                    "ask": round(ask, 8),
                    "spread": round(ask - bid, 8) if ask > 0 and bid > 0 else 0.0,
                    "spread_percentage": round((ask - bid) / bid * 100, 4) if ask > 0 and bid > 0 else 0.0,
                },
                "24h_stats": {
                    "high": round(high_24h, 8),
                    "low": round(low_24h, 8),
                    "volume": round(volume_24h, 2),
                    "percent_change": round(percent_change, 2),
                    "range_percentage": round((high_24h - low_24h) / low_24h * 100, 2) if low_24h > 0 else 0.0,
                }
            }
                
            return result
                
        except Exception as e:
            logger.error(f"Error getting market summary for {params.symbol}: {e}")
            return {
                "error": f"Failed to get market summary: {str(e)}",
                "symbol": params.symbol,
                "exchange": params.exchange
            }
            
    def build_agent(self) -> Agent:
        """
        Build and configure the agent instance.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            tools=self.tools,
            instructions=self._get_instructions()
        )
        
        self.agent_instance = agent
        return agent
