import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple, Union

def plot_ohlc(
    df: pd.DataFrame,
    title: str = "Price Chart",
    volume: bool = True,
    ma_periods: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    date_format: str = '%Y-%m-%d',
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create an OHLC (Open-High-Low-Close) price chart with Plotly
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data (must contain 'open', 'high', 'low', 'close')
        title (str): Chart title
        volume (bool): Whether to include volume subplot
        ma_periods (Optional[List[int]]): List of periods for moving averages
        figsize (Tuple[int, int]): Figure size (width, height)
        date_format (str): Date format for x-axis
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Verify required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Check if volume is available
    has_volume = 'volume' in df.columns and volume
    
    # Create subplot structure
    if has_volume:
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(title, "Volume"),
            row_heights=[0.7, 0.3]
        )
    else:
        fig = make_subplots(
            rows=1, 
            cols=1,
            subplot_titles=(title,)
        )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add moving averages if requested
    if ma_periods is not None:
        for period in ma_periods:
            ma = df['close'].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma,
                    name=f"MA ({period})",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
    
    # Add volume chart if available
    if has_volume:
        # Color volume bars based on price change
        colors = ['green' if close >= open else 'red' 
                  for open, close in zip(df['open'], df['close'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name="Volume",
                marker=dict(color=colors, opacity=0.7)
            ),
            row=2, col=1
        )
    
    # Update layout for better appearance
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis date format
    fig.update_xaxes(
        type='category' if not isinstance(df.index, pd.DatetimeIndex) else 'date',
        tickformat=date_format
    )
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None


def plot_line_chart(
    series: Union[pd.Series, Dict[str, pd.Series]],
    title: str = "Price Chart",
    y_label: str = "Price",
    figsize: Tuple[int, int] = (12, 6),
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create a line chart with Plotly
    
    Args:
        series (Union[pd.Series, Dict[str, pd.Series]]): Series or dict of series to plot
        title (str): Chart title
        y_label (str): Y-axis label
        figsize (Tuple[int, int]): Figure size (width, height)
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Create figure
    fig = go.Figure()
    
    # Handle different input types
    if isinstance(series, pd.Series):
        # Single series
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=series.name if series.name is not None else 'Series'
            )
        )
    elif isinstance(series, dict):
        # Multiple series
        for name, s in series.items():
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode='lines',
                    name=name
                )
            )
    else:
        raise ValueError("series must be a pandas Series or a dict of Series")
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date" if isinstance(
            series.index if isinstance(series, pd.Series) else list(series.values())[0].index, 
            pd.DatetimeIndex
        ) else "Time",
        yaxis_title=y_label,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None


def plot_forecast(
    train: pd.Series,
    forecast: pd.Series,
    test: Optional[pd.Series] = None,
    title: str = "Forecast",
    y_label: str = "Value",
    confidence_intervals: Optional[Dict[str, pd.Series]] = None,
    figsize: Tuple[int, int] = (12, 6),
    include_train: bool = True,
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create a forecast chart with Plotly
    
    Args:
        train (pd.Series): Training data
        forecast (pd.Series): Forecast data
        test (Optional[pd.Series]): Test data for comparison
        title (str): Chart title
        y_label (str): Y-axis label
        confidence_intervals (Optional[Dict[str, pd.Series]]): Dictionary with 'lower' and 'upper' confidence bounds
        figsize (Tuple[int, int]): Figure size (width, height)
        include_train (bool): Whether to include training data in the plot
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Create figure
    fig = go.Figure()
    
    # Add training data if requested
    if include_train:
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train.values,
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            )
        )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        )
    )
    
    # Add test data if provided
    if test is not None:
        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=test.values,
                mode='lines',
                name='Actual',
                line=dict(color='green')
            )
        )
    
    # Add confidence intervals if provided
    if confidence_intervals is not None and 'lower' in confidence_intervals and 'upper' in confidence_intervals:
        lower = confidence_intervals['lower']
        upper = confidence_intervals['upper']
        
        # Add confidence interval as a filled area
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast.index, forecast.index[::-1]]),
                y=pd.concat([lower, upper[::-1]]),
                fill='toself',
                fillcolor='rgba(231,107,243,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date" if isinstance(train.index, pd.DatetimeIndex) else "Time",
        yaxis_title=y_label,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None


def plot_decomposition(
    decomposition: Dict[str, pd.Series],
    title: str = "Time Series Decomposition",
    figsize: Tuple[int, int] = (12, 10),
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Plot time series decomposition with Plotly
    
    Args:
        decomposition (Dict[str, pd.Series]): Dictionary with decomposition components
                                             (must contain 'observed', 'trend', 'seasonal', 'residual')
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Verify required components exist
    required_components = ['observed', 'trend', 'seasonal', 'residual']
    for component in required_components:
        if component not in decomposition:
            raise ValueError(f"Required component '{component}' not found in decomposition")
    
    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
    )
    
    # Add traces for each component
    components = [
        ('observed', 1, 'blue'),
        ('trend', 2, 'red'),
        ('seasonal', 3, 'green'),
        ('residual', 4, 'purple')
    ]
    
    for component, row, color in components:
        series = decomposition[component].dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=component.capitalize(),
                line=dict(color=color)
            ),
            row=row, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        showlegend=False
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)
    
    # Set x-axis title for bottom subplot
    fig.update_xaxes(title_text="Date" if isinstance(decomposition['observed'].index, pd.DatetimeIndex) else "Time", row=4, col=1)
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None


def plot_acf_pacf(
    acf_values: np.ndarray,
    pacf_values: np.ndarray,
    lags: np.ndarray,
    acf_confint: Optional[np.ndarray] = None,
    pacf_confint: Optional[np.ndarray] = None,
    title: str = "ACF and PACF",
    figsize: Tuple[int, int] = (12, 8),
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Plot ACF and PACF with Plotly
    
    Args:
        acf_values (np.ndarray): ACF values
        pacf_values (np.ndarray): PACF values
        lags (np.ndarray): Lag values
        acf_confint (Optional[np.ndarray]): ACF confidence intervals
        pacf_confint (Optional[np.ndarray]): PACF confidence intervals
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Autocorrelation Function (ACF)", "Partial Autocorrelation Function (PACF)")
    )
    
    # Add ACF trace
    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_values,
            name="ACF",
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    # Add PACF trace
    fig.add_trace(
        go.Bar(
            x=lags,
            y=pacf_values,
            name="PACF",
            marker_color='red'
        ),
        row=2, col=1
    )
    
    # Add confidence intervals if provided
    if acf_confint is not None:
        # Calculate confidence bounds
        acf_lower = acf_confint[:, 0] - acf_values
        acf_upper = acf_confint[:, 1] - acf_values
        
        # Add confidence interval as error bars
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=acf_values,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=acf_upper,
                    arrayminus=abs(acf_lower)
                ),
                mode='markers',
                marker=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    if pacf_confint is not None:
        # Calculate confidence bounds
        pacf_lower = pacf_confint[:, 0] - pacf_values
        pacf_upper = pacf_confint[:, 1] - pacf_values
        
        # Add confidence interval as error bars
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=pacf_values,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=pacf_upper,
                    arrayminus=abs(pacf_lower)
                ),
                mode='markers',
                marker=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        showlegend=False
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    
    # Set x-axis titles
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None