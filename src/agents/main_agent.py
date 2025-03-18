"""
Main orchestration agent for coordinating the crypto market analysis system.
"""
import logging
from typing import Dict, List, Any, Optional
import json
import asyncio

from agents import Agent, Tool, Runner, handoff
from pydantic import BaseModel, Field

from .base_agent import BaseMarketAgent
from ..config import get_config
from .orderbook_agent import OrderbookAnalysisAgent
from .technical_agent import TechnicalAnalysisAgent
from .token_dashboard_agent import TokenDashboardAgent
from .execution_agent import ExecutionAgent
from .market_data_agent import MarketDataAgent

logger = logging.getLogger(__name__)


class MainOrchestrationAgent(BaseMarketAgent):
    """
    Main orchestration agent that coordinates all specialized agents.
    Handles initial user requests and routes to the appropriate specialist.
    """
    
    def __init__(self):
        """Initialize the main orchestration agent."""
        super().__init__(
            name="Crypto Market Analysis Orchestrator",
            description="An orchestration agent that coordinates specialized market analysis agents."
        )
        
        # Store specialized agents for potential handoffs
        self.specialized_agents = []
        
        # Register tools
        self._register_tools()
        
    def _get_instructions(self) -> str:
        """
        Get specific instructions for the main orchestration agent.
        
        Returns:
            String with agent instructions
        """
        return """
        You are the Crypto Market Analysis Orchestrator, responsible for coordinating specialized analysis agents.
        
        You can delegate tasks to the following specialized agents:
        
        1. Market Data Agent: For retrieving current prices, market summaries, and volume information
        2. Orderbook Analysis Agent: For analyzing market depth, liquidity, and order book patterns
        3. Technical Analysis Agent: For performing technical analysis using indicators and chart patterns
        4. Token Dashboard Agent: For comprehensive token analysis combining all data sources
        5. Execution Agent: For executing trades, managing positions, and implementing risk management
        
        Based on the user's question, determine which specialized agent(s) would be most appropriate to handle it,
        and delegate accordingly. Synthesize the information from multiple agents when necessary to provide
        comprehensive answers.
        
        When users ask for comprehensive analysis or a "full picture" of a token, use the Token Dashboard Agent
        which provides integrated insights from multiple data sources.
        
        For trading-related requests, use the Execution Agent to:
        - Execute trades with proper risk management
        - Monitor and close positions
        - Set and adjust leverage
        - Implement trailing stops and other advanced order types
        - Generate trading signals based on various strategies
        
        Always maintain a helpful, informative tone and explain complex concepts in clear terms.
        """
    
    def _register_tools(self) -> None:
        """Register tools for the agent."""
        # This agent doesn't need additional tools as it mainly coordinates other agents
        pass
    
    async def setup_crypto_agents(self):
        """Setup and initialize all crypto market analysis agents."""
        try:
            logger.info("Setting up specialized crypto market analysis agents...")
            
            # Initialize specialized agents
            orderbook_agent = OrderbookAnalysisAgent()
            technical_agent = TechnicalAnalysisAgent()
            market_data_agent = MarketDataAgent()
            token_dashboard_agent = TokenDashboardAgent()
            execution_agent = ExecutionAgent()
            
            # Register tools
            self.tools = [
                market_data_agent.as_tool(
                    tool_name="get_price",
                    tool_description="Get the current price of a cryptocurrency pair"
                ),
                market_data_agent.as_tool(
                    tool_name="get_market_summary",
                    tool_description="Get a detailed market summary with price, volume, and 24h statistics"
                ),
                orderbook_agent.as_tool(
                    tool_name="analyze_orderbook_data",
                    tool_description="Analyze market depth, liquidity, and order book patterns"
                ),
                technical_agent.as_tool(
                    tool_name="perform_technical_analysis",
                    tool_description="Perform technical analysis using indicators and chart patterns"
                ),
                token_dashboard_agent.as_tool(
                    tool_name="get_token_dashboard",
                    tool_description="Generate a comprehensive token dashboard with market data, technical analysis, and orderbook analysis"
                ),
                execution_agent.as_tool(
                    tool_name="execute_trade",
                    tool_description="Execute a trade on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="close_position",
                    tool_description="Close an open position on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="get_positions",
                    tool_description="Get all open positions on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="get_open_orders",
                    tool_description="Get all open orders on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="set_leverage",
                    tool_description="Set leverage for a trading pair on a cryptocurrency exchange"
                ),
                execution_agent.as_tool(
                    tool_name="set_trailing_stop",
                    tool_description="Set a trailing stop for an open position"
                ),
                execution_agent.as_tool(
                    tool_name="get_balance",
                    tool_description="Get the current balance of a cryptocurrency exchange account"
                )
            ]
            
            # Register specialized agents
            self.register_specialized_agent(orderbook_agent, "Analyze market depth, liquidity, and order book patterns")
            self.register_specialized_agent(technical_agent, "Perform technical analysis using indicators and chart patterns")
            self.register_specialized_agent(market_data_agent, "Retrieve current prices, market summaries, and volume information")
            self.register_specialized_agent(token_dashboard_agent, "Generate a comprehensive token dashboard with market data, technical analysis, and orderbook analysis")
            self.register_specialized_agent(execution_agent, "Execute trades, manage positions, and implement risk management")
            
            logger.info("Specialized crypto market analysis agents set up successfully")
        
        except Exception as e:
            logger.error(f"Error setting up specialized crypto market analysis agents: {e}")
    
    def create_agent(self) -> Agent:
        """
        Create the OpenAI Agent instance with handoffs to specialized agents.
        
        Returns:
            Configured Agent instance
        """
        self.agent_instance = Agent(
            name=self.name,
            instructions=self._get_instructions(),
            tools=self.tools
        )
        
        return self.agent_instance
    
    def register_specialized_agent(self, agent_instance: Agent, intent_description: str) -> None:
        """
        Register a specialized agent for potential handoff.
        
        Args:
            agent_instance: The specialized agent instance
            intent_description: Description of when to use this agent
        """
        handoff_obj = handoff(
            agent=agent_instance,
            tool_description_override=intent_description
        )
        
        self.specialized_agents.append({
            "agent": agent_instance,
            "description": intent_description
        })
        
        # Add the agent as a handoff to the main agent
        if self.agent_instance:
            if not hasattr(self.agent_instance, "handoffs"):
                self.agent_instance.handoffs = []
            self.agent_instance.handoffs.append(handoff_obj)
        
        logger.info(f"Registered specialized agent: {agent_instance.name}")

    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user request by coordinating specialized agents.
        
        Args:
            user_input: User's input string
            
        Returns:
            Dictionary with the response
        """
        try:
            logger.info(f"Processing request: '{user_input}'")
            
            # Create the agent if it doesn't exist
            agent = self.create_agent()
            
            # Run the agent
            result = await Runner.run(agent, user_input)
            
            # Extract the response
            response = {
                "response": result.final_output,
                "success": True
            }
            
            logger.info(f"Request processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "response": f"An error occurred while processing your request: {str(e)}",
                "success": False
            }

    def add_tool(self, tool):
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool instance to add
        """
        self.tools.append(tool)
