import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple, Union

def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'rmse',
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 6),
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create a bar chart comparing model performance on a specific metric
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary of model metrics
                                              (format: {model_name: {metric_name: value}})
        metric_name (str): Name of metric to compare
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Extract model names and metric values
    model_names = list(metrics.keys())
    metric_values = [metrics[model].get(metric_name, np.nan) for model in model_names]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=metric_values,
            marker_color='lightblue',
            text=[f"{val:.4f}" if not np.isnan(val) else "N/A" for val in metric_values],
            textposition='auto'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{title} - {metric_name.upper()}",
        xaxis_title="Model",
        yaxis_title=metric_name.upper(),
        width=figsize[0] * 80,
        height=figsize[1] * 80
    )
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None


def plot_multiple_metrics(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = ['rmse', 'mae', 'mape', 'r2'],
    title: str = "Model Metrics Comparison",
    figsize: Tuple[int, int] = (12, 10),
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create a subplot of multiple metrics for model comparison
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary of model metrics
                                              (format: {model_name: {metric_name: value}})
        metric_names (List[str]): List of metrics to compare
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Extract model names
    model_names = list(metrics.keys())
    
    # Create subplots
    fig = make_subplots(
        rows=len(metric_names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[metric.upper() for metric in metric_names]
    )
    
    # Add bar charts for each metric
    for i, metric in enumerate(metric_names, 1):
        # Extract metric values
        metric_values = [metrics[model].get(metric, np.nan) for model in model_names]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=metric_values,
                marker_color='lightblue',
                text=[f"{val:.4f}" if not np.isnan(val) else "N/A" for val in metric_values],
                textposition='auto',
                name=metric.upper()
            ),
            row=i, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        showlegend=False
    )
    
    # Set y-axis titles
    for i, metric in enumerate(metric_names, 1):
        fig.update_yaxes(title_text=metric.upper(), row=i, col=1)
    
    # Set x-axis title for bottom subplot
    fig.update_xaxes(title_text="Model", row=len(metric_names), col=1)
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None


def plot_equity_curve(
    initial_investment: float,
    returns: Union[pd.Series, Dict[str, pd.Series]],
    title: str = "Investment Performance",
    figsize: Tuple[int, int] = (12, 6),
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create an equity curve chart
    
    Args:
        initial_investment (float): Initial investment amount
        returns (Union[pd.Series, Dict[str, pd.Series]]): Series or dict of series with returns (percentage)
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Create figure
    fig = go.Figure()
    
    # Handle different input types
    if isinstance(returns, pd.Series):
        # Calculate equity curve for single series
        equity = initial_investment * (1 + returns / 100).cumprod()
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode='lines',
                name=returns.name if returns.name is not None else 'Strategy'
            )
        )
        
        # Add initial investment as a horizontal line
        fig.add_trace(
            go.Scatter(
                x=[equity.index[0], equity.index[-1]],
                y=[initial_investment, initial_investment],
                mode='lines',
                name='Initial Investment',
                line=dict(dash='dash', color='grey')
            )
        )
    elif isinstance(returns, dict):
        # Calculate equity curves for multiple series
        for name, series in returns.items():
            equity = initial_investment * (1 + series / 100).cumprod()
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    mode='lines',
                    name=name
                )
            )
        
        # Add initial investment as a horizontal line
        sample_index = list(returns.values())[0].index
        fig.add_trace(
            go.Scatter(
                x=[sample_index[0], sample_index[-1]],
                y=[initial_investment, initial_investment],
                mode='lines',
                name='Initial Investment',
                line=dict(dash='dash', color='grey')
            )
        )
    else:
        raise ValueError("returns must be a pandas Series or a dict of Series")
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date" if isinstance(
            returns.index if isinstance(returns, pd.Series) else list(returns.values())[0].index, 
            pd.DatetimeIndex
        ) else "Time",
        yaxis_title="Portfolio Value",
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
    
    # Format y-axis as currency
    fig.update_yaxes(tickprefix="$", tickformat=",.")
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None


def plot_drawdowns(
    returns: Union[pd.Series, Dict[str, pd.Series]],
    title: str = "Drawdowns",
    figsize: Tuple[int, int] = (12, 6),
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create a drawdown chart
    
    Args:
        returns (Union[pd.Series, Dict[str, pd.Series]]): Series or dict of series with returns (percentage)
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Function to calculate drawdowns
    def calculate_drawdowns(series):
        # Calculate cumulative returns
        cum_returns = (1 + series / 100).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cum_returns / running_max - 1) * 100
        
        return drawdowns
    
    # Create figure
    fig = go.Figure()
    
    # Handle different input types
    if isinstance(returns, pd.Series):
        # Calculate drawdowns for single series
        drawdowns = calculate_drawdowns(returns)
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                mode='lines',
                name=returns.name if returns.name is not None else 'Strategy',
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)'
            )
        )
    elif isinstance(returns, dict):
        # Calculate drawdowns for multiple series
        for name, series in returns.items():
            drawdowns = calculate_drawdowns(series)
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns.values,
                    mode='lines',
                    name=name
                )
            )
    else:
        raise ValueError("returns must be a pandas Series or a dict of Series")
    
    # Add zero line
    fig.add_trace(
        go.Scatter(
            x=[drawdowns.index[0], drawdowns.index[-1]] if isinstance(returns, pd.Series) 
              else [list(returns.values())[0].index[0], list(returns.values())[0].index[-1]],
            y=[0, 0],
            mode='lines',
            name='No Drawdown',
            line=dict(dash='dash', color='grey')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date" if isinstance(
            returns.index if isinstance(returns, pd.Series) else list(returns.values())[0].index, 
            pd.DatetimeIndex
        ) else "Time",
        yaxis_title="Drawdown (%)",
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


def plot_model_residuals(
    residuals: Union[pd.Series, Dict[str, pd.Series]],
    title: str = "Model Residuals",
    figsize: Tuple[int, int] = (12, 10),
    include_histogram: bool = True,
    return_fig: bool = False
) -> Optional[go.Figure]:
    """
    Create a residual analysis chart
    
    Args:
        residuals (Union[pd.Series, Dict[str, pd.Series]]): Series or dict of series with residuals
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size (width, height)
        include_histogram (bool): Whether to include residual histogram
        return_fig (bool): Whether to return the figure object
        
    Returns:
        Optional[go.Figure]: Plotly figure if return_fig is True
    """
    # Create subplots
    if include_histogram:
        fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.1,
            subplot_titles=("Residuals Over Time", "Residual Distribution"),
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    # Handle different input types
    if isinstance(residuals, pd.Series):
        # Add residuals over time
        fig.add_trace(
            go.Scatter(
                x=residuals.index,
                y=residuals.values,
                mode='markers',
                name=residuals.name if residuals.name is not None else 'Residuals'
            ),
            row=1 if include_histogram else None, col=1
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[residuals.index[0], residuals.index[-1]],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(dash='dash', color='grey')
            ),
            row=1 if include_histogram else None, col=1
        )
        
        # Add histogram
        if include_histogram:
            fig.add_trace(
                go.Histogram(
                    x=residuals.values,
                    name='Distribution',
                    marker_color='rgba(0, 0, 255, 0.5)'
                ),
                row=2, col=1
            )
    elif isinstance(residuals, dict):
        # Add residuals over time for each model
        for name, series in residuals.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='markers',
                    name=name
                ),
                row=1 if include_histogram else None, col=1
            )
        
        # Add zero line
        sample_index = list(residuals.values())[0].index
        fig.add_trace(
            go.Scatter(
                x=[sample_index[0], sample_index[-1]],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(dash='dash', color='grey')
            ),
            row=1 if include_histogram else None, col=1
        )
        
        # Add histograms
        if include_histogram:
            for name, series in residuals.items():
                fig.add_trace(
                    go.Histogram(
                        x=series.values,
                        name=f'{name} Distribution',
                        opacity=0.5
                    ),
                    row=2, col=1
                )
    else:
        raise ValueError("residuals must be a pandas Series or a dict of Series")
    
    # Update layout
    fig.update_layout(
        title=title,
        width=figsize[0] * 80,
        height=figsize[1] * 80
    )
    
    # Set y-axis title for residuals plot
    if include_histogram:
        fig.update_yaxes(title_text="Residual", row=1, col=1)
        fig.update_xaxes(title_text="Date" if isinstance(
            residuals.index if isinstance(residuals, pd.Series) else list(residuals.values())[0].index, 
            pd.DatetimeIndex
        ) else "Time", row=1, col=1)
        
        # Set x-axis title for histogram
        fig.update_xaxes(title_text="Residual Value", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Residual")
        fig.update_xaxes(title_text="Date" if isinstance(
            residuals.index if isinstance(residuals, pd.Series) else list(residuals.values())[0].index, 
            pd.DatetimeIndex
        ) else "Time")
    
    # Return figure if requested
    if return_fig:
        return fig
    
    # Show figure
    fig.show()
    return None