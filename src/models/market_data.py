"""
Market data models for cryptocurrency data.
"""
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime

class MarketSummary(BaseModel):
    """
    Market summary data for a cryptocurrency pair.
    """
    symbol: str
    exchange: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    percent_change_24h: float = 0.0  # Default to 0.0 instead of None
    high_24h: float
    low_24h: float
    timestamp: int
    
    @property
    def spread(self) -> float:
        """Calculate the bid-ask spread percentage."""
        return (self.ask - self.bid) / self.bid * 100
    
    @property
    def datetime(self) -> str:
        """Convert timestamp to datetime string."""
        dt = datetime.fromtimestamp(self.timestamp / 1000)  # Convert ms to seconds
        return dt.isoformat()

class VolumeData(BaseModel):
    """
    Volume data for a cryptocurrency pair.
    """
    symbol: str
    exchange: str
    volume: float
    price: float
    timestamp: int
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)  # Convert ms to seconds

class MarketData(BaseModel):
    """
    OHLCV data for a cryptocurrency pair.
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)  # Convert ms to seconds
    
    @property
    def range(self) -> float:
        """Calculate the price range percentage."""
        return (self.high - self.low) / self.low * 100

class OrderbookData(BaseModel):
    """
    Orderbook data for a cryptocurrency pair.
    """
    symbol: str
    exchange: str
    timestamp: int
    bids: List[List[float]]  # [price, amount]
    asks: List[List[float]]  # [price, amount]
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)  # Convert ms to seconds
    
    @property
    def best_bid(self) -> float:
        """Get the best bid price."""
        return self.bids[0][0] if self.bids else 0
    
    @property
    def best_ask(self) -> float:
        """Get the best ask price."""
        return self.asks[0][0] if self.asks else 0
    
    @property
    def spread(self) -> float:
        """Calculate the bid-ask spread percentage."""
        if not self.bids or not self.asks:
            return 0
        return (self.best_ask - self.best_bid) / self.best_bid * 100
