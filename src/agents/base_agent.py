"""
Base agent module defining common functionality for all agents in the system.
"""
import logging
from typing import Dict, List, Optional, Any

from agents import Agent, Tool, Runner
from pydantic import BaseModel

from ..config import get_config

logger = logging.getLogger(__name__)


class BaseMarketAgent:
    """
    Base class for all market analysis agents.
    Provides common functionality and interface for all specialized agents.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a base market agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's capabilities
        """
        self.name = name
        self.description = description
        self.config = get_config()
        self.agent_instance = None
        self.tools = []
        
    def _get_instructions(self) -> str:
        """
        Get the default instructions for the agent.
        Should be overridden by subclasses to provide specific instructions.
        
        Returns:
            String with agent instructions
        """
        return f"""
        You are {self.name}, {self.description}
        
        Follow these guidelines:
        1. Provide clear, concise responses based on data analysis
        2. When uncertain, acknowledge limitations rather than making assumptions
        3. Always explain your reasoning and analysis process
        4. Format numerical data appropriately (e.g., percentages, decimal places)
        5. Use markdown formatting for clarity when presenting complex information
        """
    
    def create_agent(self) -> Agent:
        """
        Create the OpenAI Agent instance.
        
        Returns:
            Configured Agent instance
        """
        # Create the agent if it doesn't exist
        if not self.agent_instance:
            self.agent_instance = Agent(
                name=self.name,
                instructions=self._get_instructions(),
                tools=self.tools,
            )
        
        return self.agent_instance
    
    def add_tool(self, tool):
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool instance to add
        """
        self.tools.append(tool)
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        Run the agent with user input.
        
        Args:
            user_input: User input string
            
        Returns:
            Dictionary with agent response
        """
        agent = self.create_agent()
        # Run the agent
        result = await Runner.run(agent, user_input)
        return result
