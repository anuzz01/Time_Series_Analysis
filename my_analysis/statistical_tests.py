import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, Any, Tuple, List, Optional, Union

def adf_test_summary(series: pd.Series, regression: str = 'c') -> Dict[str, Any]:
    """
    Perform Augmented Dickey-Fuller test for stationarity and generate a text summary
    
    Args:
        series (pd.Series): Time series to test
        regression (str): Regression type ('c' for constant, 'ct' for constant and trend, 
                         'ctt' for constant, trend, and quadratic trend, 'n' for no regression)
        
    Returns:
        Dict[str, Any]: Dictionary with test results and summary text
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
    
    # Create summary text
    summary = []
    summary.append(f"ADF Statistic: {adf_statistic:.4f}")
    summary.append(f"p-value: {p_value:.4f}")
    summary.append(f"Lags Used: {lags_used}")
    summary.append(f"Number of Observations: {n_obs}")
    summary.append("Critical Values:")
    for key, value in critical_values.items():
        summary.append(f"    {key}: {value:.4f}")
    
    if is_stationary:
        summary.append("Result: The series is stationary (reject the null hypothesis of a unit root)")
    else:
        summary.append("Result: The series is non-stationary (fail to reject the null hypothesis of a unit root)")
    
    # Return results
    return {
        'stationary': is_stationary,
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'lags_used': lags_used,
        'num_observations': n_obs,
        'critical_values': critical_values,
        'summary': '\n'.join(summary)
    }

def kpss_test_summary(series: pd.Series, regression: str = 'c', nlags: str = 'auto') -> Dict[str, Any]:
    """
    Perform KPSS test for stationarity and generate a text summary
    
    Args:
        series (pd.Series): Time series to test
        regression (str): Regression type ('c' for constant, 'ct' for constant and trend)
        nlags (str): Number of lags to use
        
    Returns:
        Dict[str, Any]: Dictionary with test results and summary text
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
    
    # Create summary text
    summary = []
    summary.append(f"KPSS Statistic: {kpss_statistic:.4f}")
    summary.append(f"p-value: {p_value:.4f}")
    summary.append(f"Lags Used: {lags_used}")
    summary.append("Critical Values:")
    for key, value in critical_values.items():
        summary.append(f"    {key}: {value:.4f}")
    
    if is_stationary:
        summary.append("Result: The series is stationary (fail to reject the null hypothesis of stationarity)")
    else:
        summary.append("Result: The series is non-stationary (reject the null hypothesis of stationarity)")
    
    # Return results
    return {
        'stationary': is_stationary,
        'kpss_statistic': kpss_statistic,
        'p_value': p_value,
        'lags_used': lags_used,
        'critical_values': critical_values,
        'summary': '\n'.join(summary)
    }

def ljung_box_test(series: pd.Series, lags: int = 10) -> Dict[str, Any]:
    """
    Perform Ljung-Box test for autocorrelation
    
    Args:
        series (pd.Series): Time series to test
        lags (int): Number of lags to test
        
    Returns:
        Dict[str, Any]: Dictionary with test results and summary
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    # Perform Ljung-Box test
    result = acorr_ljungbox(series_clean, lags=lags)
    
    # Extract results
    lb_statistics = result[0]
    p_values = result[1]
    
    # Determine if series has autocorrelation
    has_autocorrelation = np.any(p_values < 0.05)
    
    # Create summary text
    summary = []
    summary.append("Ljung-Box Test for Autocorrelation")
    summary.append("Lag | Test Statistic | p-value | Significant")
    for i, (stat, pval) in enumerate(zip(lb_statistics, p_values), 1):
        is_significant = pval < 0.05
        significance = "Yes" if is_significant else "No"
        summary.append(f"{i} | {stat:.4f} | {pval:.4f} | {significance}")
    
    if has_autocorrelation:
        summary.append("\nResult: The series has significant autocorrelation at some lags")
    else:
        summary.append("\nResult: The series does not have significant autocorrelation at tested lags")
    
    # Return results
    return {
        'has_autocorrelation': has_autocorrelation,
        'lb_statistics': lb_statistics,
        'p_values': p_values,
        'significant_lags': np.where(p_values < 0.05)[0] + 1,  # +1 because lags are 1-indexed
        'summary': '\n'.join(summary)
    }

def granger_causality_test(
    x: pd.Series, 
    y: pd.Series, 
    maxlag: int = 5, 
    test: str = 'ssr_chi2test'
) -> Dict[str, Any]:
    """
    Perform Granger causality test to check if x Granger-causes y
    
    Args:
        x (pd.Series): Potential causal series
        y (pd.Series): Effect series
        maxlag (int): Maximum number of lags to test
        test (str): Test to use ('ssr_chi2test', 'ssr_ftest', 'ssr_chi2test', 'lrtest')
        
    Returns:
        Dict[str, Any]: Dictionary with test results and summary
    """
    # Ensure both series have the same index
    if not x.index.equals(y.index):
        raise ValueError("Both series must have the same index")
    
    # Combine series into DataFrame
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Drop NaN values
    df = df.dropna()
    
    # Perform Granger causality test
    result = grangercausalitytests(df[['y', 'x']], maxlag=maxlag, verbose=False)
    
    # Extract results for the specified test
    test_results = {}
    causes_at_lags = []
    
    for lag in range(1, maxlag + 1):
        test_stat = result[lag][0][test][0]
        p_value = result[lag][0][test][1]
        test_results[lag] = {
            'test_statistic': test_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        if p_value < 0.05:
            causes_at_lags.append(lag)
    
    # Determine overall result
    causes = len(causes_at_lags) > 0
    
    # Create summary text
    summary = []
    summary.append(f"Granger Causality Test: X → Y (using {test})")
    summary.append("Lag | Test Statistic | p-value | Significant")
    for lag, res in test_results.items():
        significance = "Yes" if res['significant'] else "No"
        summary.append(f"{lag} | {res['test_statistic']:.4f} | {res['p_value']:.4f} | {significance}")
    
    if causes:
        summary.append(f"\nResult: X Granger-causes Y at lag(s): {causes_at_lags}")
    else:
        summary.append("\nResult: X does not Granger-cause Y at any of the tested lags")
    
    # Return results
    return {
        'causes': causes,
        'causes_at_lags': causes_at_lags,
        'test_results': test_results,
        'summary': '\n'.join(summary)
    }

def suggest_arima_orders(series: pd.Series, max_order: int = 5) -> Dict[str, Any]:
    """
    Suggest potential ARIMA orders based on ACF and PACF plots
    
    Args:
        series (pd.Series): Time series to analyze
        max_order (int): Maximum order to consider
        
    Returns:
        Dict[str, Any]: Dictionary with suggested orders and explanation
    """
    # Drop NaN values
    series_clean = series.dropna()
    
    # Check stationarity
    adf_result = adf_test_summary(series_clean)
    
    # If not stationary, difference the series
    if not adf_result['stationary']:
        diff_series = series_clean.diff().dropna()
        diff_adf_result = adf_test_summary(diff_series)
        
        if not diff_adf_result['stationary']:
            diff2_series = diff_series.diff().dropna()
            diff2_adf_result = adf_test_summary(diff2_series)
            
            if diff2_adf_result['stationary']:
                d = 2
                series_to_analyze = diff2_series
            else:
                d = 1
                series_to_analyze = diff_series
        else:
            d = 1
            series_to_analyze = diff_series
    else:
        d = 0
        series_to_analyze = series_clean
    
    # Calculate ACF and PACF
    acf_values = acf(series_to_analyze, nlags=max_order, fft=True)
    pacf_values = pacf(series_to_analyze, nlags=max_order)
    
    # Determine significance threshold
    significance_threshold = 1.96 / np.sqrt(len(series_to_analyze))
    
    # Count significant lags in ACF and PACF
    significant_acf = np.where(np.abs(acf_values[1:]) > significance_threshold)[0] + 1
    significant_pacf = np.where(np.abs(pacf_values[1:]) > significance_threshold)[0] + 1
    
    # Suggest possible models
    models = []
    explanations = []
    
    # Check for AR pattern (PACF cuts off, ACF tails off)
    if len(significant_pacf) > 0 and (len(significant_acf) > len(significant_pacf) or len(significant_acf) >= max_order // 2):
        p = max(significant_pacf)
        if p <= max_order:
            models.append((p, d, 0))
            explanations.append(f"AR({p}): PACF cuts off after lag {p}, ACF tails off gradually")
    
    # Check for MA pattern (ACF cuts off, PACF tails off)
    if len(significant_acf) > 0 and (len(significant_pacf) > len(significant_acf) or len(significant_pacf) >= max_order // 2):
        q = max(significant_acf)
        if q <= max_order:
            models.append((0, d, q))
            explanations.append(f"MA({q}): ACF cuts off after lag {q}, PACF tails off gradually")
    
    # Check for ARMA pattern (both ACF and PACF tail off)
    if len(significant_acf) > 0 and len(significant_pacf) > 0:
        p = min(3, len(significant_pacf))
        q = min(3, len(significant_acf))
        if p > 0 and q > 0 and p <= max_order and q <= max_order:
            models.append((p, d, q))
            explanations.append(f"ARMA({p},{q}): Both ACF and PACF tail off gradually")
    
    # If no patterns detected, suggest simple models
    if not models:
        models.append((1, d, 1))
        explanations.append("ARIMA(1,d,1): Simple model as starting point")
        models.append((1, d, 0))
        explanations.append("ARIMA(1,d,0): Simple AR model as starting point")
        models.append((0, d, 1))
        explanations.append("ARIMA(0,d,1): Simple MA model as starting point")
    
    # Create summary
    summary = []
    summary.append(f"Suggested differencing order (d): {d}")
    summary.append("\nSuggested ARIMA models:")
    for i, ((p, d, q), explanation) in enumerate(zip(models, explanations), 1):
        summary.append(f"{i}. ARIMA({p},{d},{q}): {explanation}")
    
    summary.append("\nNote: These are suggestions based on ACF/PACF patterns. Multiple models should be compared using AIC, BIC, or other criteria.")
    
    # Return results
    return {
        'suggested_models': models,
        'differencing_order': d,
        'explanations': explanations,
        'significant_acf_lags': significant_acf.tolist(),
        'significant_pacf_lags': significant_pacf.tolist(),
        'summary': '\n'.join(summary)
    }