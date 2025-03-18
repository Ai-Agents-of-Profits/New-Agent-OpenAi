"""
Configuration module for the crypto market analysis system.
Handles loading of settings, API keys, and defaults.
"""
import os
import logging
from typing import Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class ExchangeConfig(BaseModel):
    """Configuration for a specific exchange."""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timeout: int = 30000  # 30 seconds in milliseconds
    rate_limit: bool = True
    testnet: bool = False  # Important: Set to False to use real mainnet data


class Config:
    """Main configuration class for the market analysis system."""
    
    def __init__(self):
        """Initialize the configuration."""
        self.default_exchange = "binanceusdm"
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.__post_init__()
    
    def __post_init__(self):
        """Load exchange-specific configurations."""
        # Add default exchanges
        for exchange_name in ["binance", "binanceusdm", "binancecoinm", "coinbase", "kraken", "kucoin"]:
            env_prefix = exchange_name.upper()
            
            # Get API credentials from environment variables
            api_key = os.getenv(f"{env_prefix}_API_KEY")
            api_secret = os.getenv(f"{env_prefix}_API_SECRET")
            
            # Create exchange config
            exchange_config = ExchangeConfig(
                name=exchange_name,
                api_key=api_key,
                api_secret=api_secret,
                testnet=False  # Explicitly set testnet to False for mainnet
            )
            
            self.exchanges[exchange_name] = exchange_config
            
        logger.info(f"Loaded configurations for {len(self.exchanges)} exchanges")


# Singleton configuration instance
_config_instance = None


def get_config() -> Config:
    """
    Get the configuration singleton.
    
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
