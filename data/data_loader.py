import pandas as pd
import requests
from typing import Tuple, Dict, Optional, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlphaVantageAPI:
    """
    Class to handle data fetching from Alpha Vantage API
    """
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str):
        """
        Initialize with API key
        
        Args:
            api_key (str): Alpha Vantage API key
        """
        self.api_key = api_key
        
    def fetch_time_series_daily(self, symbol: str, outputsize: str = "compact") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch daily time series data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            outputsize (str): 'compact' for latest 100 data points, 'full' for up to 20 years
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (DataFrame with data, error message if any)
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        try:
            logger.info(f"Fetching daily time series data for {symbol}")
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            # Check if there's an error message in the response
            if "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None, data["Error Message"]
            
            # Check for API limit message
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
            
            # Extract time series data
            time_series = data.get("Time Series (Daily)", {})
            
            if not time_series:
                logger.error("No data available for the selected ticker.")
                return None, "No data available for the selected ticker."
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Rename columns
            df.columns = ["open", "high", "low", "close", "volume"]
            
            # Add date as index
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            logger.info(f"Successfully fetched data with {len(df)} rows")
            return df, None
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None, str(e)
    
    def fetch_intraday(self, symbol: str, interval: str = "5min", outputsize: str = "compact") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch intraday time series data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            interval (str): Time interval between data points (1min, 5min, 15min, 30min, 60min)
            outputsize (str): 'compact' for latest 100 data points, 'full' for extended history
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (DataFrame with data, error message if any)
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        try:
            logger.info(f"Fetching intraday data for {symbol} with interval {interval}")
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            # Check if there's an error message in the response
            if "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None, data["Error Message"]
            
            # Check for API limit message
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
            
            # Extract time series data
            time_series_key = f"Time Series ({interval})"
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                logger.error("No data available for the selected ticker and interval.")
                return None, "No data available for the selected ticker and interval."
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Rename columns
            df.columns = ["open", "high", "low", "close", "volume"]
            
            # Add date as index
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            logger.info(f"Successfully fetched intraday data with {len(df)} rows")
            return df, None
            
        except Exception as e:
            logger.error(f"Error fetching intraday data: {str(e)}")
            return None, str(e)
    
    def fetch_crypto(self, symbol: str, market: str = "USD", interval: str = "daily") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch cryptocurrency data for a given symbol
        
        Args:
            symbol (str): Crypto symbol (e.g., 'BTC', 'ETH')
            market (str): Market currency (e.g., 'USD', 'EUR')
            interval (str): 'daily', 'weekly', 'monthly'
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (DataFrame with data, error message if any)
        """
        function_map = {
            "daily": "DIGITAL_CURRENCY_DAILY",
            "weekly": "DIGITAL_CURRENCY_WEEKLY",
            "monthly": "DIGITAL_CURRENCY_MONTHLY"
        }
        
        if interval not in function_map:
            return None, f"Invalid interval: {interval}. Must be one of: daily, weekly, monthly"
        
        params = {
            "function": function_map[interval],
            "symbol": symbol,
            "market": market,
            "apikey": self.api_key
        }
        
        try:
            logger.info(f"Fetching {interval} crypto data for {symbol}/{market}")
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            # Check if there's an error message in the response
            if "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None, data["Error Message"]
            
            # Check for API limit message
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
            
            # Extract time series data
            time_series_key = f"Time Series (Digital Currency {interval.capitalize()})"
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                logger.error("No data available for the selected crypto and interval.")
                return None, "No data available for the selected crypto and interval."
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Filter for the market columns
            market_cols = [col for col in df.columns if market in col]
            df = df[market_cols]
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Rename columns - remove market from column names
            df.columns = [col.replace(f" ({market})", "") for col in df.columns]
            
            # Add date as index
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            logger.info(f"Successfully fetched crypto data with {len(df)} rows")
            return df, None
            
        except Exception as e:
            logger.error(f"Error fetching crypto data: {str(e)}")
            return None, str(e)
    
    def fetch_forex(self, from_currency: str, to_currency: str, interval: str = "daily") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch forex data for a given currency pair
        
        Args:
            from_currency (str): From currency (e.g., 'USD', 'EUR')
            to_currency (str): To currency (e.g., 'JPY', 'GBP')
            interval (str): 'daily', 'weekly', 'monthly'
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: (DataFrame with data, error message if any)
        """
        function_map = {
            "daily": "FX_DAILY",
            "weekly": "FX_WEEKLY",
            "monthly": "FX_MONTHLY"
        }
        
        if interval not in function_map:
            return None, f"Invalid interval: {interval}. Must be one of: daily, weekly, monthly"
        
        params = {
            "function": function_map[interval],
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "apikey": self.api_key
        }
        
        try:
            logger.info(f"Fetching {interval} forex data for {from_currency}/{to_currency}")
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            # Check if there's an error message in the response
            if "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None, data["Error Message"]
            
            # Check for API limit message
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
            
            # Extract time series data
            time_series_key = f"Time Series FX ({interval.capitalize()})"
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                logger.error("No data available for the selected forex pair and interval.")
                return None, "No data available for the selected forex pair and interval."
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Rename columns
            df.columns = ["open", "high", "low", "close"]
            
            # Add date as index
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            logger.info(f"Successfully fetched forex data with {len(df)} rows")
            return df, None
            
        except Exception as e:
            logger.error(f"Error fetching forex data: {str(e)}")
            return None, str(e)


def filter_data_by_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Filter dataframe based on selected timeframe
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        timeframe (str): Timeframe to filter ('1 Month', '3 Months', '6 Months', '1 Year', '5 Years')
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if timeframe == "1 Month":
        return df.last('30D')
    elif timeframe == "3 Months":
        return df.last('90D')
    elif timeframe == "6 Months":
        return df.last('180D')
    elif timeframe == "1 Year":
        return df.last('365D')
    elif timeframe == "5 Years":
        return df.last('1825D')
    elif timeframe == "All":
        return df
    else:
        # Default to last 30 days if timeframe not recognized
        return df.last('30D')