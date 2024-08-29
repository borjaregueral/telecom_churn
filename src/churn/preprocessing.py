import logging
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the raw data.
    
    Args:
    file_path: Path to the raw data.
    
    Returns:
    pd.DataFrame: Raw data.
    """
    # Open and read the raw data
    with file_path.open('rb') as file:
        raw = pd.read_parquet(file)
    logging.info("Data loaded from %s", file_path)
    return raw

# class CategoricalTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, columns=None):
#         self.columns = columns

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         for column in self.columns:
#             X[column] = X[column].astype('category')
#         return X
    
# class CategoricalTransformerTarget(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def fit(self, y, X=None):
#         return self

#     def transform(self, y):
#         y = y.copy()
#         return y.astype('category')
    
