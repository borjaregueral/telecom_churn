"""
Module for plotting charts and results.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

import churn.config as cfg

# Configurar el registro
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_boxplots(
    data: pd.DataFrame, numeric_variables: pd.DataFrame, target_column: str = "churn"
) -> None:
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
    fig = make_subplots(
        rows=rows,
        cols=4,
        subplot_titles=[f"Distribución de {col}" for col in numeric_columns],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    for i, num_col in enumerate(numeric_columns):
        row = (i // 4) + 1
        col = (i % 4) + 1

        # Boxplot with color differentiation
        boxplot = px.box(
            data,
            x=target_column,
            y=num_col,
            labels={target_column: target_column, num_col: num_col},
            category_orders={target_column: [0, 1]},
            color=target_column,
            color_discrete_map={0: "green", 1: "lightcoral"},
        )

        for trace in boxplot["data"]:
            fig.add_trace(trace, row=row, col=col)

    # Update layout to show legend only once and set black background
    fig.update_layout(
        height=400 * rows,
        width=1800,
        title_text=f"Variables numéricas y {target_column}",
        showlegend=False,
        **cfg.PLOTLY_LAYOUT_CONFIG,
    )

    fig.show()


def plot_barcharts(
    data: pd.DataFrame,
    categorical_variables: pd.DataFrame,
    target_column: str = "churn",
) -> None:
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
    fig = make_subplots(
        rows=rows,
        cols=4,
        subplot_titles=[
            f"Distribución de {col} por {target_column}"
            for col in categorical_columns
            if col != target_column
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    plot_index = 0
    for cat_col in categorical_columns:
        if cat_col == target_column:
            continue

        row = (plot_index // 4) + 1
        col = (plot_index % 4) + 1

        # Bar chart with color differentiation and transparency
        bar_chart = px.histogram(
            data,
            x=cat_col,
            color=target_column,
            barmode="group",
            labels={cat_col: cat_col.capitalize(), "count": "Conteo"},
            category_orders={target_column: [0, 1]},
            color_discrete_map={0: "green", 1: "lightcoral"},
            opacity=0.6,
        )  # Set opacity for transparency

        for trace in bar_chart["data"]:
            trace.update(
                marker=dict(line=dict(color=trace.marker.color, width=2))
            )  # Set the line color to match the bar color
            fig.add_trace(trace, row=row, col=col)

        plot_index += 1

    # Update layout using the configuration dictionary
    fig.update_layout(
        height=400 * rows,
        width=1800,
        title_text=f"Variables categóricas y {target_column}",
        showlegend=False,
        **cfg.PLOTLY_LAYOUT_CONFIG,  # Apply the layout configuration from the dictionary
    )

    fig.show()


def draw_roc_curve(y_true: pd.Series, y_pred_proba: pd.Series, title: str):
    """Draw ROC curve using true labels and predicted probabilities."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC curve (area = {roc_auc:.2f})",
            line=dict(color="darkorange", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="navy", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        cfg.PLOTLY_LAYOUT_CONFIG,
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        # width=cfg.FIG_SIZE[0],  # Convert inches to pixels
        # height=cfg.FIG_SIZE[1],  # Convert inches to pixels
    )
    return fig


def plot_ecdf_plots(
    data: pd.DataFrame, variables: pd.DataFrame, target_column: str = "churn"
) -> None:
    """
    Generate ECDF plot for each column against the target column using Plotly and arrange them in rows of 4.

    Parameters:
    data (pd.DataFrame): The dataset containing the columns and the target column.
    variables (pd.DataFrame): The sub-dataset of  columns to plot.
    target_column (str): The target column to compare against. Default is 'churn'.

    Returns:
    None
    """
    columns = variables.columns.tolist()
    num_plots = len(columns)
    rows = (num_plots + 3) // 4  # Calculate the number of rows needed (4 plots per row)
    fig = make_subplots(
        rows=rows,
        cols=4,
        subplot_titles=[f"Distribución de {col}" for col in columns],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    for i, num_col in enumerate(columns):
        row = (i // 4) + 1
        col = (i % 4) + 1

        # ECDF plot
        ecdf_plot = px.ecdf(
            data,
            x=num_col,
            color=target_column,
            labels={target_column: target_column, num_col: num_col},
        )

        for trace in ecdf_plot["data"]:
            fig.add_trace(trace, row=row, col=col)

    # Update layout to show legend only once and set black background
    fig.update_layout(
        height=400 * rows,
        width=1800,
        showlegend=False,
        **cfg.PLOTLY_LAYOUT_CONFIG,
    )

    fig.show()


def create_histogram(
    treatment_effects_flat: np.ndarray,
    nbins: int = 50,
    title: str = "Distribución de efectos individuales del tratamiento",
) -> None:
    """
    Creates and displays a histogram of treatment effects with customized layout settings.

    Parameters:
    - treatment_effects_flat: Flattened array of treatment effects.
    - nbins: Number of bins for the histogram.
    - title: Title of the histogram.
    """
    # Create the histogram
    fig = px.histogram(treatment_effects_flat, nbins=nbins, title=title)

    # Extract font settings from the config if they exist
    font_settings = cfg.PLOTLY_LAYOUT_CONFIG.get("font", {})
    title_font_settings = cfg.PLOTLY_LAYOUT_CONFIG.get("title", {}).get("font", {})

    # Customize the layout to center the plot at zero and apply configuration from config.py
    fig.update_layout(
        xaxis_title="Treatment Effect",
        yaxis_title="Frequency",
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            tickfont=dict(size=8),  # Reduce the x-axis tick font size
            title=dict(font=dict(size=12)),  # Reduce the x-axis title font size
        ),
        yaxis=dict(
            tickfont=dict(size=8),  # Reduce the y-axis tick font size
            title=dict(font=dict(size=12)),  # Reduce the y-axis title font size
        ),
        bargap=0.1,
        width=cfg.FIG_SIZE[0]
        * 75,  # Convert inches to pixels (assuming 100 pixels per inch)
        height=cfg.FIG_SIZE[1]
        * 75,  # Convert inches to pixels (assuming 100 pixels per inch)
        font={**font_settings, "size": 10},  # Reduce the overall font size
        title=dict(
            font={**title_font_settings, "size": 14}
        ),  # Reduce the title font size
        **{
            k: v
            for k, v in cfg.PLOTLY_LAYOUT_CONFIG.items()
            if k not in ["font", "title"]
        },
    )

    # Show the plot
    fig.show()
