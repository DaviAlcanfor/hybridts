from typing import Optional, Tuple
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib")


def plot_forecast(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    y_col: str = 'y',
    ds_col: str = 'ds',
    primary_pred: Optional[pd.DataFrame] = None,
    ax: Optional[Axes] = None,
    figsize: tuple = (12, 6),
    title: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot actual values vs forecast.
    
    Args:
        actual_df: DataFrame with actual values
        forecast_df: Forecast predictions (must have 'yhat' column)
        y_col: Column name for actual values
        ds_col: Column name for dates
        primary_pred: Optional primary model predictions for comparison
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    _check_matplotlib()
    
    if 'yhat' not in forecast_df.columns:
        raise ValueError(f"forecast_df must have 'yhat' column. Found: {list(forecast_df.columns)}")
    
    if primary_pred is not None and 'yhat' not in primary_pred.columns:
        raise ValueError(f"primary_pred must have 'yhat' column. Found: {list(primary_pred.columns)}")
    
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)  # Create new one
    else:
        fig = plt.gcf()  # Get existing figure for subplotting into 1 figure
    
    # Plot: actual 
    ax.plot(
        actual_df[ds_col],
        actual_df[y_col],
        'o-',
        color='black',
        label='Actual',
        linewidth=1.5,
        markersize=3
    )
    
    # Plot: primary
    if primary_pred is not None:
        ax.plot(
            primary_pred[ds_col],
            primary_pred['yhat'],
            '--',
            color='#3498db',
            label='Primary',
            linewidth=1.5,
            alpha=0.7
        )
    
    # Plot: forecast 
    ax.plot(
        forecast_df[ds_col],
        forecast_df['yhat'],
        'o-',
        color='#e74c3c',
        label='Forecast',
        linewidth=2,
        markersize=4
    )
    
    # Confidence interval
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        ax.fill_between(
            forecast_df[ds_col],
            forecast_df['yhat_lower'],
            forecast_df['yhat_upper'],
            alpha=0.2,
            color='#e74c3c'
        )
    
    # Labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(ds_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax