import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

def simple_moving_average(df: pd.DataFrame, column: str = 'close', window: Union[int, List[int]] = 20) -> pd.DataFrame:
    """
    Calculate Simple Moving Average(s)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate SMA for
        window (Union[int, List[int]]): Window size(s) for SMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional SMA column(s)
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If window is a single integer, convert to list
    if isinstance(window, int):
        window = [window]
    
    # Calculate SMA for each window size
    for w in window:
        result_df[f'sma_{w}'] = result_df[column].rolling(window=w).mean()
    
    return result_df

def exponential_moving_average(df: pd.DataFrame, column: str = 'close', window: Union[int, List[int]] = 20) -> pd.DataFrame:
    """
    Calculate Exponential Moving Average(s)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate EMA for
        window (Union[int, List[int]]): Window size(s) for EMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional EMA column(s)
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If window is a single integer, convert to list
    if isinstance(window, int):
        window = [window]
    
    # Calculate EMA for each window size
    for w in window:
        result_df[f'ema_{w}'] = result_df[column].ewm(span=w, adjust=False).mean()
    
    return result_df

def weighted_moving_average(df: pd.DataFrame, column: str = 'close', window: Union[int, List[int]] = 20) -> pd.DataFrame:
    """
    Calculate Weighted Moving Average(s)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate WMA for
        window (Union[int, List[int]]): Window size(s) for WMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional WMA column(s)
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If window is a single integer, convert to list
    if isinstance(window, int):
        window = [window]
    
    # Calculate WMA for each window size
    for w in window:
        weights = np.arange(1, w + 1)
        result_df[f'wma_{w}'] = result_df[column].rolling(window=w).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
    
    return result_df

def hull_moving_average(df: pd.DataFrame, column: str = 'close', window: Union[int, List[int]] = 20) -> pd.DataFrame:
    """
    Calculate Hull Moving Average(s)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate HMA for
        window (Union[int, List[int]]): Window size(s) for HMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional HMA column(s)
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If window is a single integer, convert to list
    if isinstance(window, int):
        window = [window]
    
    # Calculate HMA for each window size
    for w in window:
        # Calculate WMA with period window/2
        half_window = int(w/2)
        wma_half = result_df[column].rolling(window=half_window).apply(
            lambda x: np.sum(np.arange(1, half_window + 1) * x) / np.sum(np.arange(1, half_window + 1)), 
            raw=True
        )
        
        # Calculate WMA with period window
        wma_full = result_df[column].rolling(window=w).apply(
            lambda x: np.sum(np.arange(1, w + 1) * x) / np.sum(np.arange(1, w + 1)), 
            raw=True
        )
        
        # Calculate 2 * WMA(n/2) - WMA(n)
        temp = 2 * wma_half - wma_full
        
        # Calculate WMA with period sqrt(n) on the resulting values
        sqrt_window = int(np.sqrt(w))
        result_df[f'hma_{w}'] = temp.rolling(window=sqrt_window).apply(
            lambda x: np.sum(np.arange(1, sqrt_window + 1) * x) / np.sum(np.arange(1, sqrt_window + 1)), 
            raw=True
        )
    
    return result_df

def moving_average_convergence_divergence(df: pd.DataFrame, column: str = 'close', 
                                        fast_period: int = 12, slow_period: int = 26, 
                                        signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate MACD for
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal EMA period
        
    Returns:
        pd.DataFrame: DataFrame with additional MACD columns
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate fast and slow EMAs
    fast_ema = result_df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = result_df[column].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    result_df['macd_line'] = fast_ema - slow_ema
    
    # Calculate signal line
    result_df['macd_signal'] = result_df['macd_line'].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram/divergence
    result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']
    
    return result_df

def bollinger_bands(df: pd.DataFrame, column: str = 'close', 
                   window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate Bollinger Bands for
        window (int): Window size for moving average
        num_std (float): Number of standard deviations for bands
        
    Returns:
        pd.DataFrame: DataFrame with additional Bollinger Bands columns
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate middle band (SMA)
    result_df['bb_middle'] = result_df[column].rolling(window=window).mean()
    
    # Calculate standard deviation
    result_df['bb_std'] = result_df[column].rolling(window=window).std()
    
    # Calculate upper and lower bands
    result_df['bb_upper'] = result_df['bb_middle'] + (result_df['bb_std'] * num_std)
    result_df['bb_lower'] = result_df['bb_middle'] - (result_df['bb_std'] * num_std)
    
    # Calculate bandwidth and %B
    result_df['bb_bandwidth'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
    result_df['bb_percent_b'] = (result_df[column] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
    
    return result_df

def keltner_channel(df: pd.DataFrame, column: str = 'close', 
                   window: int = 20, atr_window: int = 10, 
                   atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Calculate Keltner Channel
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate Keltner Channel for
        window (int): Window size for EMA
        atr_window (int): Window size for Average True Range
        atr_multiplier (float): Multiplier for ATR
        
    Returns:
        pd.DataFrame: DataFrame with additional Keltner Channel columns
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate True Range
    result_df['tr'] = np.maximum(
        np.maximum(
            result_df['high'] - result_df['low'],
            np.abs(result_df['high'] - result_df['close'].shift(1))
        ),
        np.abs(result_df['low'] - result_df['close'].shift(1))
    )
    
    # Calculate Average True Range (ATR)
    result_df['atr'] = result_df['tr'].rolling(window=atr_window).mean()
    
    # Calculate middle line (EMA)
    result_df['kc_middle'] = result_df[column].ewm(span=window, adjust=False).mean()
    
    # Calculate upper and lower bands
    result_df['kc_upper'] = result_df['kc_middle'] + (result_df['atr'] * atr_multiplier)
    result_df['kc_lower'] = result_df['kc_middle'] - (result_df['atr'] * atr_multiplier)
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['tr'])
    
    return result_df