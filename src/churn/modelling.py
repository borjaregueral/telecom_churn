"""
Module for training, tuning, and evaluating models.
"""

import functools
import logging
from typing import Any, Callable, Dict, Tuple

import optuna
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from IPython.display import HTML, display
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

import churn.config as cfg

# Configure logging
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
    model = model_class(**params)
    roc_auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    scores = cross_val_score(
        model, data.drop("churn", axis=1), data["churn"], cv=cv, scoring=roc_auc_scorer
    )
    return scores.mean()


def get_class_name(model_class):
    """Helper function to get the class name, even if using functools.partial."""
    if isinstance(model_class, functools.partial):
        return model_class.func.__name__
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
) -> Tuple[float, float, float, Any, Any, Any, float]:
    """Train and evaluate the model using cross-validation and test data."""
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
    for model_name, (model_class, param_distributions) in models.items():
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
            train_data, test_data, model_class, best_params, cv
        )
        # Store results
        results[model_name] = {
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
