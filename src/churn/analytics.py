"""
Module for analyzing and discretizing the dataset variables.
"""

import logging
from typing import Any, Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from feature_engine.discretisation import (
    DecisionTreeDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
)
from IPython.display import HTML, display
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import churn.config as cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple = (10, 8),
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".3f",
    annot_size: int = 8,
    return_matrix: bool = False,
):
    """
    Plots the correlation matrix for the given DataFrame and optionally returns it.

    Parameters:
    df (pd.DataFrame): DataFrame containing the variables.
    method (str): Method of correlation ('pearson', 'kendall', 'spearman'). Default is 'pearson'.
    figsize (tuple): Size of the figure. Default is (10, 8).
    cmap (str): Colormap for the heatmap. Default is 'coolwarm'.
    annot (bool): Whether to annotate the heatmap. Default is True.
    fmt (str): String formatting code to use when adding annotations. Default is '.3f'.
    annot_size (int): Font size for the annotations. Default is 8.
    return_matrix (bool): Whether to return the correlation matrix. Default is False.

    Returns:
    pd.DataFrame: Correlation matrix if return_matrix is True.
    """

    # Calculate the correlation matrix
    correlation_matrix = df.corr(method=method)
    logging.info("Correlation matrix calculated using method: %s", method)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.color_palette(cmap, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=annot,
        fmt=fmt,
        annot_kws={"size": annot_size},
    )

    # Adjust the size of the variable names
    ax.tick_params(axis="both", which="major", labelsize=annot_size)

    title = f"{method.capitalize()} Correlation Matrix"
    plt.title(title)

    plt.show()

    if return_matrix:
        return correlation_matrix


def cramers_v_for_unique_pairs(data: pd.DataFrame) -> dict:
    """
    Calculate Cramér's V for each unique pair of categorical columns in the dataset.

    Parameters:
    data (pd.DataFrame): The dataset containing the columns.

    Returns:
    dict: A dictionary with Cramér's V values for each unique pair of columns.
    """
    # Ensure the categorical columns are correctly identified
    categorical_columns = data.select_dtypes(include=["category", "object"]).columns

    logging.info(f"Categorical columns: {categorical_columns}")

    # Set to keep track of processed pairs
    processed_pairs = set()

    # Dictionary comprehension to calculate Cramér's V for each unique pair of columns
    cramer_dict = {
        f"{col_1}_vs_{col_2}": calculate_cramers_v(data, col_1, col_2)
        for col_1 in categorical_columns
        for col_2 in categorical_columns
        if col_1 != col_2
        and (col_2, col_1) not in processed_pairs
        and not processed_pairs.add((col_1, col_2))
    }

    return cramer_dict


def calculate_cramers_v(data: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate Cramér's V statistic for categorical-categorical association.

    Parameters:
    data (pd.DataFrame): The dataset containing the columns.
    col1 (str): The name of the first categorical column.
    col2 (str): The name of the second categorical column.

    Returns:
    float: Cramér's V statistic.
    """
    confusion_matrix = pd.crosstab(data[col1], data[col2])
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n if n != 0 else 0
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else 0
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else 0
    result = (
        np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        if min((kcorr - 1), (rcorr - 1)) > 0
        else 0
    )
    return result


def relationships_cat_vs_num(
    data: pd.DataFrame, categorical_columns: List[str], numerical_columns: List[str]
) -> pd.DataFrame:
    """
    Apply the best test (ANOVA or Kruskal-Wallis) to all combinations of
    categorical and numerical variables in the dataset.

    Parameters:
    - data: DataFrame containing the dataset.
    - categorical_columns: List of categorical column names.
    - numerical_columns: List of numerical column names.

    Returns:
    - A DataFrame with the results for each combination.
    """
    tests = [
        {
            "Categorical Variable": cat_col,
            "Numerical Variable": num_col,
            "Test Used": test_used,
            "P-Value": p_value,
        }
        for cat_col in categorical_columns
        for num_col in numerical_columns
        for p_value, test_used in [best_test_for_categorical(data, num_col, cat_col)]
    ]

    results = pd.DataFrame(tests)
    return results


def best_test_for_categorical(
    data: pd.DataFrame, numerical_col: str, categorical_col: str
) -> Tuple[float, str]:
    """
    Determine whether to use ANOVA or Kruskal-Wallis test based on the assumptions
    of normality and homogeneity of variances.

    Parameters:
    - data: DataFrame containing the dataset.
    - numerical_col: Name of the numerical column to test.
    - categorical_col: Name of the categorical column to group by.

    Returns:
    - Tuple containing the p-value from the appropriate test and the name of the test used.
    """

    # # Check Normality with Anderson-Darling test
    # normality_pvalues = [
    #     np.exp(-1.2337141 * stats.anderson(data[data[categorical_col] == group][numerical_col])[0])
    #     for group in data[categorical_col].unique()
    # ]

    # Check Normality with Anderson-Darling test
    normality_pvalues = [
        stats.anderson(data[data[categorical_col] == group][numerical_col]).statistic
        < stats.anderson(
            data[data[categorical_col] == group][numerical_col]
        ).critical_values[2]
        for group in data[categorical_col].unique()
    ]

    # Convert boolean results to p-values-like structure
    normality_pvalues = [1.0 if result else 0.0 for result in normality_pvalues]

    # Check Homogeneity of Variances with Levene's test
    grouped_data = [
        data[data[categorical_col] == group][numerical_col]
        for group in data[categorical_col].unique()
    ]
    levene_stat, levene_pvalue = stats.levene(*grouped_data)

    # Determine the test based on p-values
    if all(p > 0.05 for p in normality_pvalues) and levene_pvalue > 0.05:
        # Use ANOVA if normality and homogeneity of variances are satisfied
        anova_stat, anova_pvalue = stats.f_oneway(*grouped_data)
        return anova_pvalue, "ANOVA"
    else:
        # Use Kruskal-Wallis if assumptions are violated
        kruskal_stat, kruskal_pvalue = stats.kruskal(*grouped_data)
        return kruskal_pvalue, "Kruskal-Wallis"


def discretize_and_calculate_cramers_v(
    data: pd.DataFrame, feature: str, target: str, bins: int, strategy: str
) -> Tuple[float, Any]:
    """
    Discretize a feature and calculate Cramér's V with the target variable.

    Parameters:
    data (pd.DataFrame): The dataset containing the columns.
    feature (str): The name of the feature to discretize.
    target (str): The name of the target variable.
    bins (int): The number of bins to use for discretization.
    strategy (str): The discretization strategy ('equal_frequency', 'equal_width', 'decision_tree', 'geometric_width').

    Returns:
    Tuple[float, Any]: The Cramér's V statistic and the discretizer used.
    """
    discretizer = None
    if strategy == "equal_frequency":
        discretizer = EqualFrequencyDiscretiser(q=bins, variables=[feature])
        discretized_feature = discretizer.fit_transform(data[[feature]])
    elif strategy == "equal_width":
        discretizer = EqualWidthDiscretiser(bins=bins, variables=[feature])
        discretized_feature = discretizer.fit_transform(data[[feature]])
    elif strategy == "decision_tree":
        discretizer = DecisionTreeDiscretiser(
            cv=RepeatedStratifiedKFold(
                n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS, random_state=cfg.SEED
            ),
            scoring="balanced_accuracy",
            variables=[feature],
            regression=False,
            param_grid={"max_depth": [2, 3, 4, 5, 6]},
            random_state=cfg.SEED,
            bin_output="boundaries",
        )
        discretized_feature = discretizer.fit_transform(data[[feature]], data[target])
    elif strategy == "geometric_width":
        discretizer = GeometricWidthDiscretiser(bins=bins, variables=[feature])
        discretized_feature = discretizer.fit_transform(data[[feature]])
    else:
        return 0.0, None

    cramers_v = calculate_cramers_v(
        pd.concat([discretized_feature, data[target]], axis=1), feature, target
    )
    return cramers_v, discretizer


def find_best_discretization_strategy(
    data: pd.DataFrame, feature: str, target: str, bin_range: range, threshold: float
) -> Dict[str, Any]:
    """
    Find the best discretization strategy for a feature based on Cramér's V.

    Parameters:
    data (pd.DataFrame): The dataset containing the columns.
    feature (str): The name of the feature to discretize.
    target (str): The name of the target variable.
    bin_range (range): The range of bins to try.
    threshold (float): The threshold for improvement.

    Returns:
    Dict[str, Any]: The best result for the feature.
    """
    best_cramers_v = 0
    best_bins = 0
    best_strategy = ""
    best_discretizer = None

    strategies = ["decision_tree", "equal_frequency", "equal_width", "geometric_width"]

    for bins in bin_range:
        for strategy in strategies:
            try:
                current_cramers_v, discretizer = discretize_and_calculate_cramers_v(
                    data, feature, target, bins, strategy
                )
                if current_cramers_v > best_cramers_v * (1 + threshold):
                    best_cramers_v = current_cramers_v
                    best_bins = bins
                    best_strategy = strategy
                    best_discretizer = discretizer
            except ValueError:
                continue

    return {
        "best_cramers_v": best_cramers_v,
        "best_bins": best_bins,
        "best_strategy": best_strategy,
        "best_discretizer": best_discretizer,
    }


def analyze_features(
    data: pd.DataFrame, target: str, bin_range: range, threshold: float
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all features in the dataset to find the best discretization strategy based on Cramér's V.

    Parameters:
    data (pd.DataFrame): The dataset containing the columns.
    target (str): The name of the target variable.
    bin_range (range): The range of bins to try.
    threshold (float): The threshold for improvement.

    Returns:
    Dict[str, Dict[str, Any]]: The best results for each feature.
    """
    best_results = {}

    for feature in data.columns:
        if feature == target:
            continue
        best_results[feature] = find_best_discretization_strategy(
            data, feature, target, bin_range, threshold
        )

    return best_results


def print_best_results(best_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print the best results for the most relevant features based on Cramér's V.

    Parameters:
    best_results (Dict[str, Dict[str, Any]]): The best results for each feature.
    """
    sorted_features = sorted(
        best_results.items(), key=lambda x: x[1]["best_cramers_v"], reverse=True
    )
    for feature, result in sorted_features:
        print(
            f'Feature: {feature}, Best Cramér\'s V: {result["best_cramers_v"]:.4f} with {result["best_bins"]} bins using {result["best_strategy"]} strategy'
        )


def fit_best_discretizers(
    data: pd.DataFrame, best_results: Dict[str, Dict[str, Any]], target: str
) -> Dict[str, Any]:
    """
    Fit the best discretizers for each feature based on the best results.

    Parameters:
    data (pd.DataFrame): The dataset containing the columns.
    best_results (Dict[str, Dict[str, Any]]): The best results for each feature.
    target (str): The name of the target variable.

    Returns:
    Dict[str, Any]: The fitted discretizers for each feature.
    """
    fitted_discretizers = {}

    for feature, result in best_results.items():
        strategy = result["best_strategy"]
        bins = result["best_bins"]
        if strategy == "equal_frequency":
            discretizer = EqualFrequencyDiscretiser(q=bins, variables=[feature])
            discretizer.fit(data[[feature]])
        elif strategy == "equal_width":
            discretizer = EqualWidthDiscretiser(bins=bins, variables=[feature])
            discretizer.fit(data[[feature]])
        elif strategy == "decision_tree":
            discretizer = DecisionTreeDiscretiser(
                cv=5,
                scoring="balanced_accuracy",
                variables=[feature],
                regression=False,
                random_state=cfg.SEED,
            )
            discretizer.fit(data[[feature]], data[target])
        elif strategy == "geometric_width":
            discretizer = GeometricWidthDiscretiser(bins=bins, variables=[feature])
            discretizer.fit(data[[feature]])
        fitted_discretizers[feature] = discretizer

    return fitted_discretizers


def create_binned_dataset(
    data: pd.DataFrame, fitted_discretizers: Dict[str, Any], target: str
) -> pd.DataFrame:
    """
    Create a new dataset with the binned features and the target variable.

    Parameters:
    data (pd.DataFrame): The original dataset.
    fitted_discretizers (Dict[str, Any]): The fitted discretizers for each feature.
    target (str): The name of the target variable.

    Returns:
    pd.DataFrame: The new dataset with binned features and the target variable.
    """
    binned_data = data[[target]].copy()

    for feature, discretizer in fitted_discretizers.items():
        binned_feature = discretizer.transform(data[[feature]])
        # binned_feature.columns = [f'binned_{feature}']
        binned_data = pd.concat([binned_data, binned_feature], axis=1)

    return binned_data


def compare_variances(
    best_results: dict, train_features: pd.DataFrame, column_to_discretize: str
) -> pd.DataFrame:
    """
    Compare the variances of the original column and within each bin after discretization.

    Parameters:
    - best_results: dict, contains the best discretizer for the column
    - train_features: pd.DataFrame, the dataset containing the column to be discretized
    - column_to_discretize: str, the name of the column to be discretized

    Returns:
    - pd.DataFrame, comparison of variances including bin edges
    """
    best_discretizer = best_results[column_to_discretize]["best_discretizer"]

    # Retrieve bin edges from best_discretizer
    bin_edges = best_discretizer.binner_dict_[column_to_discretize]

    # Discretize the chosen column into bins using the bin edges and calculate variances
    comparison = (
        train_features.assign(
            binned_column=pd.cut(
                train_features[column_to_discretize],
                bins=bin_edges,
                labels=False,
                include_lowest=True,
            )
        )
        .groupby("binned_column")[column_to_discretize]
        .var()
        .reset_index(name="Within Bin Variance")
        .assign(
            Bin=lambda df: df["binned_column"],
            Bin_Edges=lambda df: df["binned_column"].apply(
                lambda x: f"{bin_edges[x]} - {bin_edges[x+1]}"
            ),
            Original_Variance=train_features[column_to_discretize].var(),
        )
        .drop(columns=["binned_column"])
        .reindex(
            columns=["Bin", "Bin_Edges", "Within Bin Variance", "Original_Variance"]
        )
    )

    return comparison


def aggregate_by_variable(
    df: pd.DataFrame, variables: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Aggregates statistics by variable for each churn group.

    Parameters:
    - df: DataFrame containing the data.
    - variables: List of numerical column names to aggregate.

    Returns:
    A dictionary where keys are variable names and values are DataFrames with aggregated statistics.
    """
    results = {}

    for var in variables:
        churn_stats = (
            df.groupby("churn", observed=False)[var]
            .agg(count="count", sum="sum", mean="mean", median="median")
            .astype({"sum": int, "mean": int, "median": int})
        )
        results[var] = churn_stats

    return results


def display_numeric_results(results: Dict[str, pd.DataFrame]) -> None:
    """
    Displays the numeric results in an HTML table format.

    Parameters:
    - results: A dictionary where keys are variable names and values are DataFrames with aggregated statistics.
    """
    # Initialize HTML output
    html_output = """
    <style>
        .result-table { width: 100%; }
        .result-table td { vertical-align: top; padding: 10px; }
        .result-title { font-size: 12px; font-weight: bold; text-align: left; }
    </style>
    <table class="result-table">
    <tr>
    """

    # Generate HTML rows and columns
    rows = [
        f"<td><div class='result-title'>{var}_vs_churn:</div>{df.to_html()}</td>"
        for var, df in results.items()
    ]

    # Split rows into chunks of 3 columns each
    rows_chunks = [rows[i : i + 3] for i in range(0, len(rows), 3)]
    html_output += "</tr><tr>".join(["".join(chunk) for chunk in rows_chunks])
    html_output += "</tr></table>"

    # Display the HTML output
    display(HTML(html_output))


def aggregate_categorical_variables(data, target):
    # Convert the target column to numeric
    data[target] = pd.to_numeric(data[target], errors="coerce")

    # Select only categorical columns from the DataFrame
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns

    results = {}

    for col in categorical_columns:
        if col != target:
            agg_result = (
                data.groupby(col, observed=False)[target]
                .agg(["count", "mean"])
                .assign(mean=lambda x: (x["mean"]).round(2))
            )
            results[col] = agg_result

    data[target] = data[target].astype("category")
    return results


def display_categorical_results(results: Dict[str, pd.DataFrame]) -> None:
    """
    Displays the categorical results in an HTML table format.

    Parameters:
    - results: A dictionary where keys are variable names and values are DataFrames with aggregated statistics.
    """
    # Initialize HTML output
    html_output = """
    <style>
        .result-table { width: 100%; }
        .result-table td { vertical-align: top; padding: 10px; }
        .result-title { font-size: 12px; font-weight: bold; text-align: left; }
    </style>
    <table class="result-table">
    <tr>
    """

    # Generate HTML rows and columns
    rows = [
        f"<td><div class='result-title'>{var}_vs_churn:</div>{df.to_html()}</td>"
        for var, df in results.items()
    ]

    # Split rows into chunks of 2 columns each
    rows_chunks = [rows[i : i + 2] for i in range(0, len(rows), 2)]
    html_output += "</tr><tr>".join(["".join(chunk) for chunk in rows_chunks])
    html_output += "</tr></table>"

    # Display the HTML output
    display(HTML(html_output))

    from scipy.stats import ttest_ind


def compute_ttest(
    group1: pd.DataFrame, group2: pd.DataFrame, column_name: str
) -> Tuple[float, float]:
    """
    Computes a T-test between two groups for a specified column.

    Parameters:
    - group1: First group DataFrame.
    - group2: Second group DataFrame.
    - column_name: The column name on which to perform the T-test.

    Returns:
    A tuple containing the t-statistic and p-value.
    """
    t_stat, p_value = ttest_ind(group1[column_name], group2[column_name])
    return t_stat, p_value


def print_ttest_results(column_name: str, t_stat: float, p_value: float) -> None:
    """
    Prints the results of a T-test.

    Parameters:
    - column_name: The column name on which the T-test was performed.
    - t_stat: The t-statistic from the T-test.
    - p_value: The p-value from the T-test.
    """
    print(f"T-test for {column_name}: t-statistic = {t_stat}, p-value = {p_value}")


def perform_ttest(group1: pd.DataFrame, group2: pd.DataFrame, column_name: str) -> None:
    """
    Performs a T-test between two groups for a specified column and prints the results.

    Parameters:
    - group1: First group DataFrame.
    - group2: Second group DataFrame.
    - column_name: The column name on which to perform the T-test.
    """
    t_stat, p_value = compute_ttest(group1, group2, column_name)
    print_ttest_results(column_name, t_stat, p_value)


def calculate_cate_estimates(
    kf: KFold, X: pd.DataFrame, y: pd.Series, model: SVC
) -> List[float]:
    """
    Calculates CATE estimates using cross-validation.

    Parameters:
    - kf: KFold cross-validator.
    - X: Feature set.
    - y: Target set.
    - model: CATE model (e.g., ThresholdedSVC).

    Returns:
    A list of CATE estimates for each fold.
    """
    cate_estimates: List[float] = []

    for train_index, test_index in kf.split(X):
        # Train and test splits for each fold
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Train the CATE model
        model.fit(X_train_fold, y_train_fold)

        # Get the predicted treatment effects on the test fold
        treatment_effects_fold = model.predict_proba(X_test_fold)[
            :, 1
        ]  # Assuming you're predicting probabilities

        # Calculate CATE for the test fold
        cate_fold = np.mean(treatment_effects_fold)
        cate_estimates.append(cate_fold)

    return cate_estimates


def print_cate_statistics(cate_estimates: List[float]) -> None:
    """
    Prints the mean and variance of CATE estimates.

    Parameters:
    - cate_estimates: A list of CATE estimates.
    """
    mean_cate = np.mean(cate_estimates)
    variance_cate = np.var(cate_estimates)
    print(f"Mean CATE across folds: {mean_cate}")
    print(f"Variance of CATE across folds: {variance_cate}")


def separate_treatment_variable(
    X_train: pd.DataFrame, X_test: pd.DataFrame, treatment_col: str
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Separates the treatment variable from the feature sets.

    Parameters:
    - X_train: Training feature set.
    - X_test: Test feature set.
    - treatment_col: The name of the treatment column.

    Returns:
    A tuple containing the treatment variables and the feature sets without the treatment column.
    """
    treatment_train = X_train[treatment_col]
    treatment_test = X_test[treatment_col]
    X_train = X_train.drop(columns=treatment_col)
    X_test = X_test.drop(columns=treatment_col)
    return treatment_train, treatment_test, X_train, X_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, scaler: StandardScaler
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scales the feature sets using the provided scaler.

    Parameters:
    - X_train: Training feature set.
    - X_test: Test feature set.
    - scaler: Scaler to fit on the training data and transform both training and test data.

    Returns:
    A tuple containing the scaled training and test feature sets.
    """
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_train_scaled, X_test_scaled


def add_treatment_variable(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    treatment_train: pd.Series,
    treatment_test: pd.Series,
    treatment_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds the treatment variable back to the scaled feature sets.

    Parameters:
    - X_train_scaled: Scaled training feature set.
    - X_test_scaled: Scaled test feature set.
    - treatment_train: Treatment variable for the training set.
    - treatment_test: Treatment variable for the test set.
    - treatment_col: The name of the treatment column.

    Returns:
    A tuple containing the scaled training and test feature sets with the treatment variable added back.
    """
    X_train_scaled[treatment_col] = treatment_train.values
    X_test_scaled[treatment_col] = treatment_test.values
    return X_train_scaled, X_test_scaled


def define_cate_variables(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Defines covariates, treatment, and outcome for the CATE model.

    Parameters:
    - X_train_scaled: Scaled training feature set.
    - X_test_scaled: Scaled test feature set.
    - y_train: Training target set.
    - y_test: Test target set.

    Returns:
    A tuple containing the covariates, treatment, and outcome for both training and test sets.
    """
    covariates = ["customer_service_rating", "customer_happiness"]

    X_train_cate = X_train_scaled[covariates].values
    X_test_cate = X_test_scaled[covariates].values
    treatment_train = X_train_scaled["treatment"].values
    treatment_test = X_test_scaled["treatment"].values
    y_train_cate = y_train.values
    y_test_cate = y_test.values

    return (
        X_train_cate,
        X_test_cate,
        treatment_train,
        treatment_test,
        y_train_cate,
        y_test_cate,
    )


def get_top_customers_indices(
    uplift_scores: np.ndarray, top_percent: float = 0.1
) -> np.ndarray:
    """
    Identifies the indices of the top customers based on uplift scores.

    Parameters:
    - uplift_scores: Array of uplift scores.
    - top_percent: Percentage of top customers to select.

    Returns:
    An array of indices of the top customers.
    """
    n_customers = len(uplift_scores)
    top_customers_count = int(top_percent * n_customers)
    top_customers_indices = np.argsort(np.abs(uplift_scores))[-top_customers_count:]
    top_customers_sorted_indices = top_customers_indices[
        np.argsort(uplift_scores[top_customers_indices])
    ]
    return top_customers_sorted_indices


def unscale_features(
    scaled_features: pd.DataFrame, scaler: StandardScaler
) -> pd.DataFrame:
    """
    Unscales the features using the provided scaler.

    Parameters:
    - scaled_features: Scaled feature set.
    - scaler: Scaler used to scale the features.

    Returns:
    A DataFrame containing the unscaled features.
    """
    unscaled_features = pd.DataFrame(
        scaler.inverse_transform(scaled_features.values),
        columns=scaled_features.columns,
        index=scaled_features.index,
    )
    return unscaled_features


def get_top_customers_for_treatment(
    uplift_scores: np.ndarray,
    X_test_scaled: pd.DataFrame,
    scaler: StandardScaler,
    top_percent: float = 0.1,
) -> pd.DataFrame:
    """
    Identifies the top customers for treatment based on uplift scores and returns their unscaled features.

    Parameters:
    - uplift_scores: Array of uplift scores.
    - X_test_scaled: Scaled test feature set.
    - scaler: Scaler used to scale the features.
    - top_percent: Percentage of top customers to select for treatment.

    Returns:
    A DataFrame containing the unscaled features of the top customers for treatment along with their uplift scores and treatment status.
    """
    # Flatten uplift scores if necessary
    uplift_scores = uplift_scores.flatten()

    # Get the indices of the top customers
    top_customers_sorted_indices = get_top_customers_indices(uplift_scores, top_percent)

    # Apply the treatment to the top customers
    top_customers_for_treatment = X_test_scaled.iloc[top_customers_sorted_indices]

    # Optionally, get the uplift for this top group
    top_customers_uplift = uplift_scores[top_customers_sorted_indices]

    # Create a DataFrame to include both customer features and their uplift scores
    top_customers_df = top_customers_for_treatment.copy()
    top_customers_df["Uplift Score"] = top_customers_uplift

    # Drop the treatment column before unscaling
    top_customers_no_treatment = top_customers_for_treatment.drop(columns=["treatment"])

    # Unscale the features using the scaler
    top_customers_unscaled = unscale_features(top_customers_no_treatment, scaler)

    # Add the uplift scores and to the unscaled DataFrame
    top_customers_unscaled["Uplift Score"] = top_customers_uplift

    return top_customers_unscaled
