import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

def on_balance_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain close, volume)
        
    Returns:
        pd.DataFrame: DataFrame with additional OBV column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate price direction
    result_df['direction'] = np.where(
        result_df['close'] > result_df['close'].shift(1), 1,
        np.where(result_df['close'] < result_df['close'].shift(1), -1, 0)
    )
    
    # Calculate OBV
    result_df['obv'] = (result_df['direction'] * result_df['volume']).cumsum()
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['direction'])
    
    return result_df

def volume_weighted_average_price(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """
    Calculate Volume-Weighted Average Price (VWAP)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low, close, volume)
        window (int, optional): Window size for calculation. If None, calculates from start of dataframe.
        
    Returns:
        pd.DataFrame: DataFrame with additional VWAP column
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
    
    # Calculate price * volume
    result_df['pv'] = result_df['typical_price'] * result_df['volume']
    
    if window is None:
        # Calculate cumulative values
        result_df['cum_pv'] = result_df['pv'].cumsum()
        result_df['cum_volume'] = result_df['volume'].cumsum()
        
        # Calculate VWAP
        result_df['vwap'] = result_df['cum_pv'] / result_df['cum_volume']
        
        # Drop temporary columns
        result_df = result_df.drop(columns=['typical_price', 'pv', 'cum_pv', 'cum_volume'])
    else:
        # Calculate rolling values
        result_df['rolling_pv'] = result_df['pv'].rolling(window=window).sum()
        result_df['rolling_volume'] = result_df['volume'].rolling(window=window).sum()
        
        # Calculate VWAP
        result_df['vwap'] = result_df['rolling_pv'] / result_df['rolling_volume']
        
        # Drop temporary columns
        result_df = result_df.drop(columns=['typical_price', 'pv', 'rolling_pv', 'rolling_volume'])
    
    return result_df

def accumulation_distribution_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Accumulation/Distribution Line
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low, close, volume)
        
    Returns:
        pd.DataFrame: DataFrame with additional A/D Line column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate money flow multiplier
    # Formula: ((Close - Low) - (High - Close)) / (High - Low)
    result_df['mfm'] = ((result_df['close'] - result_df['low']) - 
                        (result_df['high'] - result_df['close'])) / \
                        (result_df['high'] - result_df['low'])
    
    # Handle division by zero (when high = low)
    result_df['mfm'] = result_df['mfm'].replace([np.inf, -np.inf], 0)
    result_df['mfm'] = result_df['mfm'].fillna(0)
    
    # Calculate money flow volume
    result_df['mfv'] = result_df['mfm'] * result_df['volume']
    
    # Calculate A/D Line (cumulative sum of money flow volume)
    result_df['ad_line'] = result_df['mfv'].cumsum()
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['mfm', 'mfv'])
    
    return result_df

def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Chaikin Money Flow (CMF)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain high, low, close, volume)
        window (int): Window size for calculation
        
    Returns:
        pd.DataFrame: DataFrame with additional CMF column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate money flow multiplier
    # Formula: ((Close - Low) - (High - Close)) / (High - Low)
    result_df['mfm'] = ((result_df['close'] - result_df['low']) - 
                        (result_df['high'] - result_df['close'])) / \
                        (result_df['high'] - result_df['low'])
    
    # Handle division by zero (when high = low)
    result_df['mfm'] = result_df['mfm'].replace([np.inf, -np.inf], 0)
    result_df['mfm'] = result_df['mfm'].fillna(0)
    
    # Calculate money flow volume
    result_df['mfv'] = result_df['mfm'] * result_df['volume']
    
    # Calculate Chaikin Money Flow
    result_df['cmf'] = result_df['mfv'].rolling(window=window).sum() / \
                      result_df['volume'].rolling(window=window).sum()
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['mfm', 'mfv'])
    
    return result_df

def volume_price_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Price Trend (VPT)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain close, volume)
        
    Returns:
        pd.DataFrame: DataFrame with additional VPT column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate price change percentage
    result_df['price_change_pct'] = result_df['close'].pct_change()
    
    # Calculate VPT
    result_df['vpt'] = (result_df['volume'] * result_df['price_change_pct']).cumsum()
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['price_change_pct'])
    
    return result_df

def negative_volume_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Negative Volume Index (NVI)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain close, volume)
        
    Returns:
        pd.DataFrame: DataFrame with additional NVI column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate price change percentage
    result_df['price_change_pct'] = result_df['close'].pct_change()
    
    # Calculate volume change
    result_df['volume_change'] = result_df['volume'].pct_change()
    
    # Initialize NVI column (start with 1000)
    result_df['nvi'] = 1000.0
    
    # Calculate NVI
    for i in range(1, len(result_df)):
        if result_df.iloc[i]['volume'] < result_df.iloc[i-1]['volume']:
            result_df.iloc[i, result_df.columns.get_loc('nvi')] = \
                result_df.iloc[i-1]['nvi'] * (1.0 + result_df.iloc[i]['price_change_pct'])
        else:
            result_df.iloc[i, result_df.columns.get_loc('nvi')] = result_df.iloc[i-1]['nvi']
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['price_change_pct', 'volume_change'])
    
    return result_df

def positive_volume_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Positive Volume Index (PVI)
    
    Args:
        df (pd.DataFrame): DataFrame with time series data (must contain close, volume)
        
    Returns:
        pd.DataFrame: DataFrame with additional PVI column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Calculate price change percentage
    result_df['price_change_pct'] = result_df['close'].pct_change()
    
    # Calculate volume change
    result_df['volume_change'] = result_df['volume'].pct_change()
    
    # Initialize PVI column (start with 1000)
    result_df['pvi'] = 1000.0
    
    # Calculate PVI
    for i in range(1, len(result_df)):
        if result_df.iloc[i]['volume'] > result_df.iloc[i-1]['volume']:
            result_df.iloc[i, result_df.columns.get_loc('pvi')] = \
                result_df.iloc[i-1]['pvi'] * (1.0 + result_df.iloc[i]['price_change_pct'])
        else:
            result_df.iloc[i, result_df.columns.get_loc('pvi')] = result_df.iloc[i-1]['pvi']
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['price_change_pct', 'volume_change'])
    
    return result_df