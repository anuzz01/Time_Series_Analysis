import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

def relative_strength_index(df: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        column (str): Column to calculate RSI for
        window (int): Window size for RSI calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional RSI column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate price changes
    delta = result_df[column].diff()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    result_df['rsi'] = 100 - (100 / (1 + rs))
    
    return result_df

def stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low, close)
        k_window (int): Window size for %K calculation
        d_window (int): Window size for %D calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional Stochastic Oscillator columns
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate %K
    # Formula: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    low_min = result_df['low'].rolling(window=k_window).min()
    high_max = result_df['high'].rolling(window=k_window).max()
    
    result_df['stoch_k'] = 100 * ((result_df['close'] - low_min) / (high_max - low_min))
    
    # Calculate %D (3-day SMA of %K)
    result_df['stoch_d'] = result_df['stoch_k'].rolling(window=d_window).mean()
    
    return result_df

def williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Williams %R
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low, close)
        window (int): Window size for calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional Williams %R column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate Williams %R
    # Formula: %R = (Highest High - Current Close) / (Highest High - Lowest Low) * -100
    high_max = result_df['high'].rolling(window=window).max()
    low_min = result_df['low'].rolling(window=window).min()
    
    result_df['williams_r'] = -100 * ((high_max - result_df['close']) / (high_max - low_min))
    
    return result_df

def commodity_channel_index(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Commodity Channel Index (CCI)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low, close)
        window (int): Window size for calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional CCI column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate typical price
    result_df['typical_price'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # Calculate simple moving average of typical price
    sma_tp = result_df['typical_price'].rolling(window=window).mean()
    
    # Calculate mean deviation
    # First calculate absolute deviation from SMA for each period
    mad = result_df['typical_price'].rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    
    # Calculate CCI
    # Formula: CCI = (Typical Price - SMA of Typical Price) / (0.015 * Mean Deviation)
    result_df['cci'] = (result_df['typical_price'] - sma_tp) / (0.015 * mad)
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['typical_price'])
    
    return result_df

def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Money Flow Index (MFI)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low, close, volume)
        window (int): Window size for calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional MFI column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate typical price
    result_df['typical_price'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # Calculate raw money flow
    result_df['money_flow'] = result_df['typical_price'] * result_df['volume']
    
    # Calculate direction
    result_df['direction'] = np.where(result_df['typical_price'] > result_df['typical_price'].shift(1), 1, -1)
    
    # Calculate positive and negative money flow
    result_df['positive_flow'] = np.where(result_df['direction'] > 0, result_df['money_flow'], 0)
    result_df['negative_flow'] = np.where(result_df['direction'] < 0, result_df['money_flow'], 0)
    
    # Calculate positive and negative money flow sums
    positive_flow_sum = result_df['positive_flow'].rolling(window=window).sum()
    negative_flow_sum = result_df['negative_flow'].rolling(window=window).sum()
    
    # Calculate money flow ratio
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    
    # Calculate MFI
    result_df['mfi'] = 100 - (100 / (1 + money_flow_ratio))
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['typical_price', 'money_flow', 'direction', 'positive_flow', 'negative_flow'])
    
    return result_df

def awesome_oscillator(df: pd.DataFrame, fast_window: int = 5, slow_window: int = 34) -> pd.DataFrame:
    """
    Calculate Awesome Oscillator (AO)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low)
        fast_window (int): Window size for fast SMA
        slow_window (int): Window size for slow SMA
        
    Returns:
        pd.DataFrame: DataFrame with additional AO column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate median price
    result_df['median_price'] = (result_df['high'] + result_df['low']) / 2
    
    # Calculate fast SMA
    fast_sma = result_df['median_price'].rolling(window=fast_window).mean()
    
    # Calculate slow SMA
    slow_sma = result_df['median_price'].rolling(window=slow_window).mean()
    
    # Calculate Awesome Oscillator
    result_df['ao'] = fast_sma - slow_sma
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['median_price'])
    
    return result_df