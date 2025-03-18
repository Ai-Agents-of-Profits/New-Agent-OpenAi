"""
Exchange connector module for interacting with cryptocurrency exchanges via CCXT.
"""
import asyncio
import logging
from typing import Dict, List, Any

import ccxt.async_support as ccxt
import pandas as pd
from pydantic import BaseModel

from ..config import ExchangeConfig, get_config

logger = logging.getLogger(__name__)

# Define supported exchanges and their mappings
SUPPORTED_EXCHANGES = {
    'binance': ccxt.binance,
    'binanceusdm': ccxt.binanceusdm,  # USD-Margined futures (USDT/BUSD settled)
    'binancecoinm': ccxt.binancecoinm,  # COIN-Margined futures
    'coinbase': ccxt.coinbase,
    'kraken': ccxt.kraken,
    'kucoin': ccxt.kucoin,
    'bybit': ccxt.bybit,
    'okx': ccxt.okx,
}


class MarketData(BaseModel):
    """Model for standardized market data across exchanges."""
    symbol: str
    exchange: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


class OrderbookData(BaseModel):
    """Model for standardized orderbook data across exchanges."""
    symbol: str
    exchange: str
    timestamp: int
    datetime: str
    bids: List[List[float]]  # [[price, amount], ...]
    asks: List[List[float]]  # [[price, amount], ...]
    metadata: Dict = None


class ExchangeConnector:
    """
    Connector for cryptocurrency exchanges using CCXT.
    Handles connection management, data fetching, and standardization.
    """
    
    def __init__(self, exchange_id: str = None):
        """
        Initialize the exchange connector.
        
        Args:
            exchange_id: ID of the exchange to connect to (default: config's default_exchange)
        """
        config = get_config()
        self.exchange_id = exchange_id or config.default_exchange
        
        if self.exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange: {self.exchange_id}")
        
        exchange_config = config.exchanges.get(self.exchange_id)
        if not exchange_config:
            exchange_config = ExchangeConfig(name=self.exchange_id)
        
        self.exchange_instance = None
        self.exchange_class = SUPPORTED_EXCHANGES[self.exchange_id]
        self.exchange_config = exchange_config
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize the CCXT exchange instance with configuration settings."""
        options = {}
        
        # Configure settings based on the exchange
        if self.exchange_id in ['binanceusdm', 'binance', 'binancecoinm']:
            if self.exchange_config.testnet:
                options['defaultType'] = 'future'
                options['testnet'] = False  # Ensure we're using mainnet, not testnet
        
        # Create common parameters for all exchanges
        exchange_params = {
            'apiKey': self.exchange_config.api_key,
            'secret': self.exchange_config.api_secret,
            'timeout': self.exchange_config.timeout,
            'enableRateLimit': self.exchange_config.rate_limit,
            'options': options
        }
        
        # Add exchange-specific parameters
        if self.exchange_id in ['binanceusdm', 'binance', 'binancecoinm']:
            # Add a larger recvWindow to prevent timestamp errors
            exchange_params['recvWindow'] = 60000  # 60 seconds (default is 5000ms)
        
        self.exchange_instance = self.exchange_class(exchange_params)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()
    
    async def close(self):
        """Close the exchange connection."""
        if self.exchange_instance:
            await self.exchange_instance.close()
    
    async def load_markets(self) -> Dict:
        """
        Load markets from the exchange.
        
        Returns:
            Dictionary of markets
        """
        try:
            return await self.exchange_instance.load_markets()
        except ccxt.NetworkError as e:
            logger.error(f"Network error loading markets: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error loading markets: {e}")
            raise
        except Exception as e:
            logger.error(f"Unknown error loading markets: {e}")
            raise
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        limit: int = 100, 
        since: int = None
    ) -> List[MarketData]:
        """
        Fetch OHLCV (candlestick) data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for candlesticks (e.g., '1m', '5m', '1h', '1d')
            limit: Maximum number of candles to fetch
            since: Timestamp in milliseconds to fetch data from
            
        Returns:
            List of MarketData objects
        """
        try:
            # Ensure markets are loaded
            if not self.exchange_instance.markets:
                await self.load_markets()
            
            # Fetch OHLCV data
            ohlcv = await self.exchange_instance.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convert to standardized format
            result = []
            for candle in ohlcv:
                timestamp, open_price, high, low, close, volume = candle
                result.append(MarketData(
                    symbol=symbol,
                    exchange=self.exchange_id,
                    timestamp=timestamp,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    timeframe=timeframe
                ))
            
            return result
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching OHLCV for {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching OHLCV for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unknown error fetching OHLCV for {symbol}: {e}")
            raise
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> OrderbookData:
        """
        Fetch orderbook data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Maximum number of orders to fetch
            
        Returns:
            OrderbookData object
        """
        try:
            # Ensure markets are loaded
            if not self.exchange_instance.markets:
                await self.load_markets()
            
            # Fetch orderbook
            orderbook = await self.exchange_instance.fetch_order_book(symbol, limit)
            
            # Convert to standardized format
            return OrderbookData(
                symbol=symbol,
                exchange=self.exchange_id,
                timestamp=orderbook['timestamp'],
                datetime=orderbook['datetime'],
                bids=orderbook['bids'],
                asks=orderbook['asks']
            )
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching orderbook for {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching orderbook for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unknown error fetching orderbook for {symbol}: {e}")
            raise
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with ticker data
        """
        try:
            return await self.exchange_instance.fetch_ticker(symbol)
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching ticker for {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching ticker for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unknown error fetching ticker for {symbol}: {e}")
            raise
    
    async def fetch_balance(self) -> Dict:
        """
        Fetch account balance.
        
        Returns:
            Dictionary with balance information
        """
        try:
            return await self.exchange_instance.fetch_balance()
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching balance: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching balance: {e}")
            raise
        except Exception as e:
            logger.error(f"Unknown error fetching balance: {e}")
            raise
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Set leverage for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            leverage: Leverage level (1-125)
            
        Returns:
            Dictionary with result information
        """
        try:
            logger.info(f"Setting leverage to {leverage}x for {symbol}")
            return await self.exchange_instance.set_leverage(leverage, symbol)
        except ccxt.NetworkError as e:
            logger.error(f"Network error setting leverage for {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error setting leverage for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unknown error setting leverage for {symbol}: {e}")
            raise
    
    async def create_market_order(self, symbol: str, side: str, amount: float, params: Dict = None) -> Dict:
        """
        Create a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            params: Additional parameters
            
        Returns:
            Dictionary with order information
        """
        try:
            params = params or {}
            logger.info(f"Creating market order: {side} {amount} {symbol}")
            return await self.exchange_instance.create_market_order(symbol, side, amount, None, params)
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            raise
    
    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: Dict = None) -> Dict:
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price
            params: Additional parameters
            
        Returns:
            Dictionary with order information
        """
        try:
            params = params or {}
            logger.info(f"Creating limit order: {side} {amount} {symbol} @ {price}")
            return await self.exchange_instance.create_limit_order(symbol, side, amount, price, params)
        except Exception as e:
            logger.error(f"Error creating limit order: {e}")
            raise
    
    async def create_stop_loss_order(self, symbol: str, side: str, amount: float, stop_price: float, params: Dict = None) -> Dict:
        """
        Create a stop loss order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            stop_price: Stop price level
            params: Additional parameters
            
        Returns:
            Dictionary with order information
        """
        try:
            params = params or {}
            params['stopPrice'] = stop_price
            logger.info(f"Creating stop loss order: {side} {amount} {symbol} @ {stop_price}")
            
            # Different exchanges have different ways to create stop orders
            if self.exchange_id in ['binance', 'binanceusdm', 'binancecoinm']:
                return await self.exchange_instance.create_order(
                    symbol, 'STOP_MARKET', side, amount, None, params
                )
            else:
                # Generic approach for other exchanges
                return await self.exchange_instance.create_order(
                    symbol, 'stop', side, amount, None, params
                )
        except Exception as e:
            logger.error(f"Error creating stop loss order: {e}")
            raise
    
    async def create_take_profit_order(self, symbol: str, side: str, amount: float, price: float, params: Dict = None) -> Dict:
        """
        Create a take profit order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Take profit price level
            params: Additional parameters
            
        Returns:
            Dictionary with order information
        """
        try:
            params = params or {}
            logger.info(f"Creating take profit order: {side} {amount} {symbol} @ {price}")
            
            # Different exchanges have different ways to create take profit orders
            if self.exchange_id in ['binance', 'binanceusdm', 'binancecoinm']:
                params['stopPrice'] = price
                return await self.exchange_instance.create_order(
                    symbol, 'TAKE_PROFIT_MARKET', side, amount, None, params
                )
            else:
                # Generic approach for other exchanges
                return await self.exchange_instance.create_order(
                    symbol, 'take_profit', side, amount, price, params
                )
        except Exception as e:
            logger.error(f"Error creating take profit order: {e}")
            raise
    
    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None, params: Dict = None) -> Dict:
        """
        Create a generic order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            type: Order type
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (optional)
            params: Additional parameters
            
        Returns:
            Dictionary with order information
        """
        try:
            params = params or {}
            logger.info(f"Creating {type} order: {side} {amount} {symbol}" + (f" @ {price}" if price else ""))
            return await self.exchange_instance.create_order(symbol, type, side, amount, price, params)
        except Exception as e:
            logger.error(f"Error creating {type} order: {e}")
            raise
    
    async def fetch_positions(self) -> List[Dict]:
        """
        Fetch open positions.
        
        Returns:
            List of dictionaries with position information
        """
        try:
            logger.info("Fetching positions")
            if hasattr(self.exchange_instance, 'fetch_positions'):
                return await self.exchange_instance.fetch_positions()
            else:
                # Fallback for exchanges that don't have this method
                logger.warning(f"Exchange {self.exchange_id} doesn't support fetch_positions")
                return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            # Return empty list instead of raising to prevent crashes
            return []
    
    async def fetch_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Fetch open orders.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of dictionaries with order information
        """
        try:
            logger.info(f"Fetching open orders" + (f" for {symbol}" if symbol else ""))
            return await self.exchange_instance.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            # Return empty list instead of raising to prevent crashes
            return []
    
    @staticmethod
    def ohlcv_to_dataframe(ohlcv_data: List[MarketData]) -> pd.DataFrame:
        """
        Convert OHLCV data to a pandas DataFrame.
        
        Args:
            ohlcv_data: List of MarketData objects
            
        Returns:
            Pandas DataFrame with OHLCV data
        """
        data = [
            {
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume
            }
            for item in ohlcv_data
        ]
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df


async def get_connector(exchange_id: str = None) -> ExchangeConnector:
    """
    Factory function to get an exchange connector.
    
    Args:
        exchange_id: ID of the exchange to connect to
        
    Returns:
        ExchangeConnector instance
    """
    connector = ExchangeConnector(exchange_id)
    await connector.load_markets()
    return connector
