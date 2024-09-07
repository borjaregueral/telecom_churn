import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Replicability
SEED = 123
# Test size for splitting the data
TEST_SIZE = 0.25

# Number of estimators
ESTIMATORS = 50
DEPTH = 6

# Number of features to be selected
NUM_FEATURES = 7

# figure size and default configuration for graphs
FIG_SIZE = (10, 8)  # Default figure size for plots
PLOTLY_LAYOUT_CONFIG = {
    "plot_bgcolor": "rgba(0,0,0,0)",  # Transparent background
    "paper_bgcolor": "rgba(0,0,0,0)",  # Transparent background
    "font": {"color": "white"},
}

# Discretization parameters
BIN_SIZES = range(2, 11)
OPT_STEP = 0.05
RELATIONSHIP_THRESHOLD = 0.10

# Modelling parameters
BERNOULLI_LOWER_BOUND = 0.1
BERNOULLI_UPPER_BOUND = 10
SVC_C_LOWER_BOUND = 0.1
SCV_C_UPPER_BOUND = 100

# Number of cross-validation folds and repetitions
N_SPLITS = 5
N_REPEATS = 2

# Number of trials for hyperparameter tunning Optuna
N_TRIALS = 50

# Percentile for the confidence interval
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
