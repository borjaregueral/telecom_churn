"""
Module for training, tuning, and evaluating models.
"""

import functools
import logging
from functools import partial
from typing import Any, Callable, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from IPython.display import HTML, display
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

import churn.config as cfg

# Set up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Optuna logging level to WARNING to reduce verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Initialize Plotly to work in offline mode
pyo.init_notebook_mode(connected=True)


def objective(
    trial: optuna.trial.Trial,
    data: pd.DataFrame,
    model_class: Callable,
    cv: Callable,
    param_distributions: Dict[str, Any],
) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    params = {
        param_name: param_func(trial)
        for param_name, param_func in param_distributions.items()
    }
    model = model_class()

    if isinstance(model, Pipeline):
        model.set_params(**params)
    else:
        model = model_class(**params)

    roc_auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    scores = cross_val_score(
        model, data.drop("churn", axis=1), data["churn"], cv=cv, scoring=roc_auc_scorer
    )
    return scores.mean()


def get_class_name(model_class: Callable, custom_name: str = None) -> str:
    if custom_name:
        return custom_name
    if isinstance(model_class, partial):
        if isinstance(model_class.func, Pipeline):
            steps = [step[0] for step in model_class.keywords["steps"]]
            return " -> ".join(steps)
        return model_class.func.__name__
    elif isinstance(model_class, Pipeline):
        steps = [step[0] for step in model_class.steps]
        return " -> ".join(steps)
    else:
        return model_class.__name__


def optimize_hyperparameters(
    data: pd.DataFrame,
    model_class: Callable,
    param_distributions: Dict[str, Any],
    n_trials: int,
    cv: Callable,
) -> Dict[str, Any]:
    """Optimize hyperparameters for a given model using Optuna."""
    model_name = get_class_name(model_class)
    logger.info(f"Starting hyperparameter optimization for {model_name}...")
    study = optuna.create_study(direction="maximize")

    for _ in tqdm(range(n_trials), desc=f"Optimizing {model_name}"):
        study.optimize(
            lambda trial: objective(trial, data, model_class, cv, param_distributions),
            n_trials=1,
            catch=(Exception,),
        )

    best_params = study.best_params
    logger.info(f"Hyperparameter optimization for {model_name} completed.")
    return best_params


def train_and_evaluate_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_class: Callable,
    best_params: Dict[str, Any],
    cv: Callable,
    model_name: str,
) -> Tuple[float, float, float, Any, Any, Any, float]:
    """Train and evaluate the model using cross-validation and test data."""
    model = model_class()

    if isinstance(model, Pipeline):
        model.set_params(**best_params)
    else:
        model = model_class(**best_params)

    # Cross-validation on training data
    roc_auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    cv_scores = cross_val_score(
        model,
        train_data.drop("churn", axis=1),
        train_data["churn"],
        cv=cv,
        scoring=roc_auc_scorer,
    )
    roc_auc_cv = cv_scores.mean()
    # Fit the model on the entire training data
    model.fit(train_data.drop("churn", axis=1), train_data["churn"])
    # Evaluate on training data
    predictions_train_proba = model.predict_proba(train_data.drop("churn", axis=1))[
        :, 1
    ]
    predictions_train = model.predict(train_data.drop("churn", axis=1))
    roc_auc_train = roc_auc_score(train_data["churn"], predictions_train_proba)
    # Evaluate on test data
    predictions_test_proba = model.predict_proba(test_data.drop("churn", axis=1))[:, 1]
    predictions_test = model.predict(test_data.drop("churn", axis=1))
    roc_auc_test = roc_auc_score(test_data["churn"], predictions_test_proba)
    # Adjust the decision threshold to balance precision and recall
    precision, recall, thresholds = precision_recall_curve(
        test_data["churn"], predictions_test_proba
    )
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = f1_scores.argmax()
    correction_factor = 0.75
    best_threshold = thresholds[best_threshold_index] * correction_factor
    # Make predictions using the best threshold
    predictions_test_adjusted = (predictions_test_proba >= best_threshold).astype(int)
    return (
        roc_auc_cv,
        roc_auc_train,
        roc_auc_test,
        model,
        predictions_train_proba,
        predictions_train,
        predictions_test_proba,
        predictions_test,
        predictions_test_adjusted,
        best_threshold,
    )


def split_features_and_label(
    df: pd.DataFrame, label_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features and label.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): The name of the label column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame and the label Series.
    """
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    return X, y


def train_tune_evaluate(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    models: Dict[str, Tuple[Callable, Dict[str, Any]]],
    cv: Callable,
    n_trials: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """
    Main function to optimize hyperparameters, train, and evaluate models.

    Parameters:
    - train_data: pd.DataFrame - Training data including features and target 'churn'.
    - test_data: pd.DataFrame - Test data including features and target 'churn'.
    - models: Dict[str, Tuple[Callable, Dict[str, Any]]] - Dictionary: models and hyperparameter.
    - n_trials: int - Number of trials for hyperparameter optimization.
    - cv: Callable - Cross-validation strategy.

    Returns:
    - results: Dict[str, Dict[str, Any]] - Dictionary: best parameters, ROC AUC, and trained model.
    """
    results = {}
    for model_key, (model_class, param_distributions) in models.items():
        # Optimize hyperparameters
        best_params = optimize_hyperparameters(
            train_data, model_class, param_distributions, n_trials, cv
        )
        # Train and evaluate the model
        (
            roc_auc_cv,
            roc_auc_train,
            roc_auc_test,
            model,
            predictions_train_proba,
            predictions_train,
            predictions_test_proba,
            predictions_test,
            predictions_test_adjusted,
            best_threshold,
        ) = train_and_evaluate_model(
            train_data, test_data, model_class, best_params, cv, model_key
        )
        # Store results
        results[model_key] = {
            "best_params": best_params,
            "roc_auc_cv": roc_auc_cv,
            "roc_auc_train": roc_auc_train,
            "roc_auc_test": roc_auc_test,
            "model": model,
            "predictions_train": predictions_train,
            "predictions_test": predictions_test,
            "predictions_train_proba": predictions_train_proba,
            "predictions_test_proba": predictions_test_proba,
            "predictions_test_adjusted": predictions_test_adjusted,
            "threshold": best_threshold,
        }
    return results


def calculate_classification_metrics(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    predictions_train: pd.Series,
    predictions_test: pd.Series,
) -> Dict[str, Any]:
    """Calculate classification reports and confusion matrices for training and test data."""
    metrics = {}
    # Classification Report (Train)
    report_train = classification_report(
        train_data["churn"], predictions_train, output_dict=True
    )
    df_report_train = pd.DataFrame(report_train).transpose()
    df_report_train = df_report_train.map(
        lambda x: f"{x:.2f}" if isinstance(x, float) else x
    )
    metrics["report_train"] = df_report_train
    # Confusion Matrix (Train)
    cm_train = confusion_matrix(train_data["churn"], predictions_train)
    df_cm_train = pd.DataFrame(
        cm_train,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    metrics["cm_train"] = df_cm_train
    # Classification Report (Test)
    report_test = classification_report(
        test_data["churn"], predictions_test, output_dict=True
    )
    df_report_test = pd.DataFrame(report_test).transpose()
    df_report_test = df_report_test.map(
        lambda x: f"{x:.2f}" if isinstance(x, float) else x
    )
    metrics["report_test"] = df_report_test
    # Confusion Matrix (Test)
    cm_test = confusion_matrix(test_data["churn"], predictions_test)
    df_cm_test = pd.DataFrame(
        cm_test,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    metrics["cm_test"] = df_cm_test

    return metrics


def display_classification_results(metrics: Dict[str, Any], model_name: str):
    """Display classification reports and confusion matrices for training and test data."""
    # Display Results
    display(HTML(f"<h2>Model: {model_name}</h2>"))
    display(
        HTML(
            """
    <div style="display: flex; justify-content: space-between;">
        <div style="flex: 1; margin-right: 10px;">
            <h3>Classification Report (Train):</h3>
            {report_train}
        </div>
        <div style="flex: 1; margin-left: 10px;">
            <h3>Confusion Matrix (Train):</h3>
            {cm_train}
        </div>
    </div>
    """.format(
                report_train=metrics["report_train"].to_html(),
                cm_train=metrics["cm_train"].to_html(),
            )
        )
    )
    display(
        HTML(
            """
    <div style="display: flex; justify-content: space-between;">
        <div style="flex: 1; margin-right: 10px;">
            <h3>Classification Report (Test):</h3>
            {report_test}
        </div>
        <div style="flex: 1; margin-left: 10px;">
            <h3>Confusion Matrix (Test):</h3>
            {cm_test}
        </div>
    </div>
    """.format(
                report_test=metrics["report_test"].to_html(),
                cm_test=metrics["cm_test"].to_html(),
            )
        )
    )
    display(HTML("<br>"))


def custom_f1_scorer(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """
    Computes the best F1 score and corresponding threshold.

    Parameters:
    - y_true: True labels.
    - y_probs: Predicted probabilities.

    Returns:
    A dictionary with the best F1 score and the best threshold.
    """
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_f1 = 0
    best_threshold = 0.5

    # Tune the decision threshold
    for threshold in thresholds:
        y_pred_threshold = (y_probs[:, 1] >= threshold).astype(
            int
        )  # Use positive class probabilities
        f1 = f1_score(y_true, y_pred_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    results = {"best_f1": best_f1, "best_threshold": best_threshold}
    return results


class ThresholdedSVC(BaseEstimator, ClassifierMixin):
    """
    A custom SVC classifier with a tunable decision threshold.

    Parameters:
    - base_model: The base SVC model.
    - threshold: The decision threshold for predicting the positive class.
    """

    def __init__(self, base_model: Any, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdedSVC":
        """
        Fit the base model.

        Parameters:
        - X: Training data.
        - y: Target values.

        Returns:
        The fitted instance of the class.
        """
        self.base_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters:
        - X: Input data.

        Returns:
        Predicted class probabilities.
        """
        return self.base_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the decision threshold.

        Parameters:
        - X: Input data.

        Returns:
        Predicted class labels.
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the F1 score.

        Parameters:
        - X: Input data.
        - y: True labels.

        Returns:
        The F1 score.
        """
        y_pred = self.predict(X)
        return f1_score(y, y_pred)


def eval_model_performance(
    model: Any, X_test: Any, y_test: Any, subgroup_name: str = "Subgroup"
) -> Dict[str, Any]:
    """
    Evaluate model performance for a given subgroup, returning Precision, Recall, F1-score, PR AUC, and Confusion Matrix.

    Parameters:
    - model: Trained model (must have `predict` and `predict_proba` methods).
    - X_test: Features for the test set (for the specific subgroup).
    - y_test: Ground truth labels for the test set (for the specific subgroup).
    - subgroup_name: Name of the subgroup for logging purposes (optional).

    Returns:
    A dictionary with Precision, Recall, F1-score, PR AUC, and Confusion Matrix.
    """
    try:
        # Make binary predictions and probability predictions for positive class
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate precision, recall, and F1-score for the positive class (churn)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        # Calculate Precision-Recall curve and PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test, y_pred_proba, pos_label=1
        )
        pr_auc = auc(recall_curve, precision_curve)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Print the results for this subgroup
        print(f"--- {subgroup_name} ---")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"PR AUC: {pr_auc}")
        print(f"Confusion Matrix:\n{cm}")

        # Return results as a dictionary
        results = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "PR AUC": pr_auc,
            "Confusion Matrix": cm,
        }
        return results
    except Exception as e:
        print(f"Error evaluating model performance for {subgroup_name}: {e}")
        return {
            "Precision": None,
            "Recall": None,
            "F1-Score": None,
            "PR AUC": None,
            "Confusion Matrix": None,
        }


def objective_causal(
    trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    """
    Objective function for Optuna optimization.

    Parameters:
    - trial: A single trial object for hyperparameter optimization.
    - X_train: Training feature set.
    - y_train: Training target set.

    Returns:
    The mean F1 score across all cross-validation folds.
    """
    # Suggest hyperparameters for SVC (optimizing only C)
    C = trial.suggest_float("C", cfg.SVC_C_LOWER_BOUND, cfg.SCV_C_UPPER_BOUND, log=True)

    # Initialize SVC with class weights and the suggested C hyperparameter
    svc_model = SVC(C=C, kernel="linear", class_weight="balanced", probability=True)

    # Define the cross-validation strategy
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS, random_state=cfg.SEED
    )

    f1_scores = []
    best_thresholds = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = (
            X_train.iloc[train_idx].values,
            X_train.iloc[val_idx].values,
        )
        y_train_fold, y_val_fold = (
            y_train.iloc[train_idx].values,
            y_train.iloc[val_idx].values,
        )

        # Handle class imbalance with SMOTEENN
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy="minority"), enn=EditedNearestNeighbours()
        )
        X_train_res, y_train_res = smote_enn.fit_resample(X_train_fold, y_train_fold)

        # Fit the model on the resampled data
        svc_model.fit(X_train_res, y_train_res)

        # Get the predicted probabilities for the validation fold
        y_probs = svc_model.predict_proba(X_val_fold)

        # Compute the F1 score by tuning the threshold
        results = custom_f1_scorer(y_val_fold, y_probs)
        f1_scores.append(results["best_f1"])
        best_thresholds.append(results["best_threshold"])

    # Store the best threshold for later use
    best_threshold = np.mean(best_thresholds)

    # Return the mean F1 score across all folds
    return np.mean(f1_scores)


def bootstrap_cate(
    treatment_effects: np.ndarray, n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Computes the Conditional Average Treatment Effect (CATE) and its 95% confidence interval using bootstrap resampling.

    Parameters:
    - treatment_effects: Array of treatment effects.
    - n_bootstrap: Number of bootstrap samples.

    Returns:
    A tuple containing the mean CATE, lower confidence interval, and upper confidence interval.
    """
    cate_bootstrap = [np.mean(resample(treatment_effects)) for _ in range(n_bootstrap)]

    # Get 95% confidence intervals
    lower_ci = np.percentile(cate_bootstrap, cfg.LOWER_PERCENTILE)
    upper_ci = np.percentile(cate_bootstrap, cfg.UPPER_PERCENTILE)

    return np.mean(cate_bootstrap), lower_ci, upper_ci


def split_and_subset(X, y, column, threshold):
    """
    Splits the DataFrame X based on the threshold of the specified column and subsets y accordingly.

    Parameters:
    - X: DataFrame to split.
    - y: Series to subset.
    - column: Column name to split on.
    - threshold: Threshold value to split the column.

    Returns:
    - high_group: Subset of X where column values are above the threshold.
    - low_group: Subset of X where column values are below or equal to the threshold.
    - y_high: Subset of y corresponding to high_group.
    - y_low: Subset of y corresponding to low_group.
    """
    high_group = X[X[column] > threshold]
    low_group = X[X[column] <= threshold]
    y_high = y.loc[high_group.index]
    y_low = y.loc[low_group.index]
    return high_group, low_group, y_high, y_low
