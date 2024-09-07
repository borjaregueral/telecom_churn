"""
Module for loading data.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the raw data.

    Args:
    file_path: Path to the raw data.

    Returns:
    pd.DataFrame: Raw data.
    """
    # Open and read the raw data
    with file_path.open("rb") as file:
        raw = pd.read_parquet(file)
    logging.info("Data loaded from %s", file_path)
    return raw
