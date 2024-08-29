import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import churn.config as cfg
from typing import List
import logging


# Configurar el registro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_boxplots(data: pd.DataFrame, numeric_variables: pd.DataFrame, target_column: str = 'churn') -> None:
    """
    Generate boxplot for each numeric column against the target column using Plotly and arrange them in rows of 4.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the numeric columns and the target column.
    numeric_variables (pd.DataFrame): The sub-dataset of numeric columns to plot.
    target_column (str): The target column to compare against. Default is 'churn'.
    
    Returns:
    None
    """
    numeric_columns = numeric_variables.columns.tolist()
    num_plots = len(numeric_columns)
    rows = (num_plots + 3) // 4  # Calculate the number of rows needed (4 plots per row)
    fig = make_subplots(rows=rows, cols=4, subplot_titles=[f'Distribución de {col}' for col in numeric_columns],
                        vertical_spacing=0.1, horizontal_spacing=0.05)

    for i, num_col in enumerate(numeric_columns):
        row = (i // 4) + 1
        col = (i % 4) + 1
        
        # Boxplot with color differentiation
        boxplot = px.box(data, x=target_column, y=num_col, 
                         labels={target_column: target_column, num_col: num_col},
                         category_orders={target_column: [0, 1]},
                         color=target_column, 
                         color_discrete_map={0: 'green', 1: 'lightcoral'})
        
        for trace in boxplot['data']:
            fig.add_trace(trace, row=row, col=col)
        
    # Update layout to show legend only once and set black background
    fig.update_layout(
        height=400 * rows, 
        width=1800, 
        title_text=f'Variables numéricas y {target_column}', 
        showlegend=False,
        **cfg.PLOTLY_LAYOUT_CONFIG)

    fig.show()


def plot_barcharts(data: pd.DataFrame, categorical_variables: pd.DataFrame, target_column: str = 'churn') -> None:
    """
    Generate bar charts for each categorical column against the target column using Plotly and arrange them in rows of 4.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the categorical columns and the target column.
    categorical_variables (pd.DataFrame): The sub-dataset of categorical columns to plot.
    target_column (str): The target column to compare against. Default is 'churn'.
    
    Returns:
    None
    """
    categorical_columns = categorical_variables.columns.tolist()
    num_plots = len(categorical_columns)
    rows = (num_plots + 3) // 4  # Calculate the number of rows needed (4 plots per row)
    fig = make_subplots(rows=rows, cols=4, subplot_titles=[f'Distribución de {col} por {target_column}' for col in categorical_columns if col != target_column],
                        vertical_spacing=0.1, horizontal_spacing=0.05)

    plot_index = 0
    for cat_col in categorical_columns:
        if cat_col == target_column:
            continue
        
        row = (plot_index // 4) + 1
        col = (plot_index % 4) + 1
        
        # Bar chart with color differentiation and transparency
        bar_chart = px.histogram(data, x=cat_col, color=target_column, barmode='group', 
                                 labels={cat_col: cat_col.capitalize(), 'count': 'Conteo'},
                                 category_orders={target_column: [0, 1]},
                                 color_discrete_map={0: 'green', 1: 'lightcoral'},
                                 opacity=0.6)  # Set opacity for transparency
        
        for trace in bar_chart['data']:
            trace.update(marker=dict(line=dict(color=trace.marker.color, width=2)))  # Set the line color to match the bar color
            fig.add_trace(trace, row=row, col=col)
        
        plot_index += 1
        
    # Update layout using the configuration dictionary
    fig.update_layout(
        height=400 * rows, 
        width=1800, 
        title_text=f'Variables categóricas y {target_column}', 
        showlegend=False,
        **cfg.PLOTLY_LAYOUT_CONFIG  # Apply the layout configuration from the dictionary
    )

    fig.show()