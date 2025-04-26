import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from typing import Dict, Any, Tuple, List, Optional, Union

def decompose_time_series(
    series: pd.Series,
    model: str = 'additive',
    period: Optional[int] = None,
    extrapolate_trend: Optional[int] = 0
) -> Dict[str, pd.Series]:
    """
    Decompose a time series into trend, seasonal, and residual components
    
    Args:
        series (pd.Series): Time series to decompose
        model (str): Type of decomposition ('additive' or 'multiplicative')
        period (Optional[int]): Period of the seasonality. If None, it will be estimated
        extrapolate_trend (Optional[int]): If 'extrapolate_trend' is 'freq', uses the entire series to estimate trend.
                                         If extrapolate_trend is an integer, uses that many points to fit trend at start and end.
    
    Returns:
        Dict[str, pd.Series]: Dictionary with decomposed components
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    # Estimate period if not provided
    if period is None:
        # Check if we have at least 2 full years of data
        if len(series_clean) >= 2 * 252:  # Assuming 252 trading days in a year
            period = 252  # Annual cycle for daily data
        elif len(series_clean) >= 2 * 52:  # At least 2 years of weekly data
            period = 52   # Annual cycle for weekly data
        elif len(series_clean) >= 2 * 12:  # At least 2 years of monthly data
            period = 12   # Annual cycle for monthly data
        else:
            # For shorter series, use a smaller period or default value
            period = min(len(series_clean) // 4, 10)  # Use a quarter of the data length or 10, whichever is smaller
    
    # Perform decomposition
    result = seasonal_decompose(
        series_clean,
        model=model,
        period=period,
        extrapolate_trend=extrapolate_trend
    )
    
    # Return components
    return {
        'observed': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid,
        'period': period,
        'model': model
    }

def adf_test(series: pd.Series, regression: str = 'c') -> Dict[str, Any]:
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Args:
        series (pd.Series): Time series to test
        regression (str): Regression type ('c' for constant, 'ct' for constant and trend, 'ctt' for constant, trend, and quadratic trend)
        
    Returns:
        Dict[str, Any]: Dictionary with test results
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    # Perform ADF test
    result = adfuller(series_clean, regression=regression)
    
    # Extract results
    adf_statistic = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]
    
    # Determine if stationary
    is_stationary = p_value < 0.05
    
    # Return results
    return {
        'stationary': is_stationary,
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'lags_used': lags_used,
        'num_observations': n_obs,
        'critical_values': critical_values
    }

def kpss_test(series: pd.Series, regression: str = 'c', nlags: str = 'auto') -> Dict[str, Any]:
    """
    Perform KPSS test for stationarity
    
    Args:
        series (pd.Series): Time series to test
        regression (str): Regression type ('c' for constant, 'ct' for constant and trend)
        nlags (str): Number of lags to use
        
    Returns:
        Dict[str, Any]: Dictionary with test results
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    # Perform KPSS test
    result = kpss(series_clean, regression=regression, nlags=nlags)
    
    # Extract results
    kpss_statistic = result[0]
    p_value = result[1]
    lags_used = result[2]
    critical_values = result[3]
    
    # Determine if stationary (for KPSS, null hypothesis is that series is stationary)
    is_stationary = p_value >= 0.05
    
    # Return results
    return {
        'stationary': is_stationary,
        'kpss_statistic': kpss_statistic,
        'p_value': p_value,
        'lags_used': lags_used,
        'critical_values': critical_values
    }

def calculate_acf_pacf(
    series: pd.Series, 
    nlags: int = 40, 
    alpha: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Calculate AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF)
    
    Args:
        series (pd.Series): Time series to analyze
        nlags (int): Number of lags to calculate
        alpha (float): Significance level for confidence intervals
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with ACF and PACF results
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    # Calculate ACF and confidence intervals
    acf_values, acf_confint = acf(series_clean, nlags=nlags, alpha=alpha, fft=True)
    
    # Calculate PACF and confidence intervals
    pacf_values, pacf_confint = pacf(series_clean, nlags=nlags, alpha=alpha)
    
    # Return results
    return {
        'acf_values': acf_values,
        'acf_confint': acf_confint,
        'pacf_values': pacf_values,
        'pacf_confint': pacf_confint,
        'lags': np.arange(len(acf_values))
    }

def rolling_statistics(
    series: pd.Series, 
    window: int = 20
) -> Dict[str, pd.Series]:
    """
    Calculate rolling statistics (mean, std, variance)
    
    Args:
        series (pd.Series): Time series to analyze
        window (int): Window size for rolling calculations
        
    Returns:
        Dict[str, pd.Series]: Dictionary with rolling statistics
    """
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    rolling_var = series.rolling(window=window).var()
    
    # Return results
    return {
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'rolling_var': rolling_var
    }

def check_time_series_properties(series: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive check of time series properties
    
    Args:
        series (pd.Series): Time series to analyze
        
    Returns:
        Dict[str, Any]: Dictionary with time series properties
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    # Basic statistics
    basic_stats = {
        'mean': series_clean.mean(),
        'median': series_clean.median(),
        'min': series_clean.min(),
        'max': series_clean.max(),
        'std': series_clean.std(),
        'skewness': series_clean.skew(),
        'kurtosis': series_clean.kurtosis()
    }
    
    # Check for stationarity
    adf_results = adf_test(series_clean)
    kpss_results = kpss_test(series_clean)
    
    # Check for autocorrelation
    acf_pacf_results = calculate_acf_pacf(series_clean, nlags=min(40, len(series_clean) // 5))
    
    # Calculate if series has strong autocorrelation
    significant_acf = np.abs(acf_pacf_results['acf_values'][1:]) > (1.96 / np.sqrt(len(series_clean)))
    has_autocorrelation = np.any(significant_acf)
    
    # Simple seasonality check (look at ACF for peaks at regular intervals)
    acf_values = acf_pacf_results['acf_values'][1:]  # Skip lag 0
    
    # Look for local maxima in ACF
    peaks = []
    for i in range(1, len(acf_values) - 1):
        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1] and acf_values[i] > 0.2:
            peaks.append(i + 1)  # +1 because we skipped lag 0
    
    # Check if peaks form a pattern
    seasonal_period = None
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        if np.allclose(intervals, intervals[0], rtol=0.3):  # Allow 30% tolerance
            seasonal_period = intervals[0]
    
    # Check for trend (using simple regression)
    x = np.arange(len(series_clean))
    y = series_clean.values
    regression = np.polyfit(x, y, 1)
    trend_slope = regression[0]
    has_trend = abs(trend_slope) > 0.01 * (series_clean.max() - series_clean.min()) / len(series_clean)
    
    # Return comprehensive results
    return {
        'basic_stats': basic_stats,
        'stationarity': {
            'adf_test': adf_results,
            'kpss_test': kpss_results,
            'is_stationary': adf_results['stationary'] and kpss_results['stationary']
        },
        'autocorrelation': {
            'has_autocorrelation': has_autocorrelation,
            'significant_lags': np.where(significant_acf)[0] + 1  # +1 because we skipped lag 0
        },
        'seasonality': {
            'potential_seasonal_period': seasonal_period,
            'acf_peaks': peaks
        },
        'trend': {
            'has_trend': has_trend,
            'trend_slope': trend_slope
        }
    }