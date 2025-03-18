"""
Token Dashboard agent for providing comprehensive token analysis.
"""
import logging
from typing import Dict, List, Any
import asyncio

from pydantic import BaseModel, Field
from agents import Tool

from .base_agent import BaseMarketAgent
from ..exchange.connector import ExchangeConnector
from ..data.token_dashboard.dashboard_controller import get_token_dashboard

logger = logging.getLogger(__name__)


class TokenDashboardRequest(BaseModel):
    """Model for token dashboard analysis requests."""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    exchange: str = Field(None, description="Exchange ID (default: binanceusdm)")
    timeframe: str = Field(None, description="Timeframe for analysis (e.g., '1h', '4h', '1d')")
    days_back: int = Field(None, description="Number of days of historical data to analyze (max 30)")
    detail_level: str = Field(None, description="Level of detail ('low', 'medium', 'high')")


class TokenDashboardAgent(BaseMarketAgent):
    """
    Specialized agent for providing comprehensive token analysis dashboards.
    Integrates market data, orderbook analysis, technical analysis, and futures data.
    """
    
    def __init__(self):
        """Initialize the token dashboard agent."""
        super().__init__(
            name="Token Dashboard Agent",
            description="A specialized agent for providing comprehensive token analysis including pricing, liquidity, technical indicators, and market structure."
        )
        
        # Register tools
        self._register_tools()
    
    def _get_instructions(self) -> str:
        """
        Get specific instructions for the token dashboard agent.
        
        Returns:
            String with agent instructions
        """
        return """
You are a specialized Token Dashboard Agent that provides comprehensive analysis for cryptocurrencies.
You can generate detailed dashboards that include:

1. Current market data and price information
2. Historical price analysis and trends
3. Volume analysis and unusual volume patterns
4. Orderbook depth and liquidity analysis
5. Technical indicator analysis and trading signals
6. Futures-specific data (funding rates, open interest) when available

Use the get-token-dashboard tool to analyze tokens and provide multi-faceted insights.

When responding to users:
- Focus on the most important insights from each section of the dashboard
- Highlight significant patterns, anomalies, or potential trading opportunities
- Explain technical concepts in an accessible way
- Provide context about why certain indicators or metrics are important
- Avoid making specific price predictions, but discuss likely scenarios based on the data
- Prioritize actionable insights over raw data when possible

Example request: "Generate a dashboard for ETH/USDT with a 4-hour timeframe, looking back 14 days."
"""
    
    def _register_tools(self) -> None:
        """Register the tools for the token dashboard agent."""
        self.add_tool(
            Tool(
                name="get-token-dashboard",
                description="Generate a comprehensive token dashboard with market data, technical analysis, and orderbook analysis",
                params={
                    "symbol": {"type": "string", "description": "Trading pair symbol (e.g., 'BTC/USDT')"},
                    "exchange": {"type": "string", "description": "Exchange ID (default: binanceusdm)", "required": False},
                    "timeframe": {"type": "string", "description": "Timeframe for analysis (e.g., '1h', '4h', '1d')", "required": False},
                    "days_back": {"type": "integer", "description": "Number of days of historical data to analyze (max 30)", "required": False},
                    "detail_level": {"type": "string", "description": "Level of detail to include ('low', 'medium', 'high')", "required": False}
                },
                function=self._generate_token_dashboard
            )
        )
    
    async def _generate_token_dashboard(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive token dashboard.
        
        Args:
            params: Parameters for the dashboard
            
        Returns:
            Comprehensive token analysis dashboard
        """
        try:
            # Validate and process parameters
            symbol = params.get("symbol")
            if not symbol:
                raise ValueError("Symbol is required")
                
            exchange_id = params.get("exchange") or "binanceusdm"
            timeframe = params.get("timeframe") or "1h"
            days_back = min(params.get("days_back", 7), 30)  # Cap at 30 days
            detail_level = params.get("detail_level") or "medium"
            
            # Get exchange connector
            async with ExchangeConnector(exchange_id) as connector:
                # Get the exchange instance
                exchange = connector.exchange_instance
                
                # Generate the dashboard
                dashboard_data = await get_token_dashboard(
                    exchange=exchange,
                    exchange_id=exchange_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    days_back=days_back,
                    detail_level=detail_level
                )
                
                # Structure the response
                result = {
                    "symbol": symbol,
                    "exchange": exchange_id,
                    "timeframe": timeframe,
                    "days_analyzed": days_back,
                    "detail_level": detail_level,
                    "dashboard": dashboard_data
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error generating token dashboard: {e}")
            return {
                "error": str(e),
                "symbol": params.get("symbol"),
                "exchange": params.get("exchange")
            }
    
    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        if tool_name == "get-token-dashboard":
            # Convert dict to request model
            request = TokenDashboardRequest(**params)
            return await self._generate_token_dashboard(params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
