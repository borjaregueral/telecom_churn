import pandas as pd
import numpy as np
import scipy.stats as stats
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser, DecisionTreeDiscretiser, GeometricWidthDiscretiser
import seaborn as sns
import churn.config as cfg
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Type, Any
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def correlation_matrix(df: pd.DataFrame, method: str = 'pearson', figsize: tuple = (10, 8), cmap: str = 'coolwarm', annot: bool = True, fmt: str = '.3f', annot_size: int = 8, return_matrix: bool = False):
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
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, 
                vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=annot, fmt=fmt, annot_kws={"size": annot_size})

    # Adjust the size of the variable names
    ax.tick_params(axis='both', which='major', labelsize=annot_size)

    title = f'{method.capitalize()} Correlation Matrix'
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
    categorical_columns = data.select_dtypes(include=['category', 'object']).columns
    
    logging.info(f"Categorical columns: {categorical_columns}")
    
    # Set to keep track of processed pairs
    processed_pairs = set()
    
    # Dictionary comprehension to calculate Cramér's V for each unique pair of columns
    cramer_dict = {
        f"{col_1}_vs_{col_2}": calculate_cramers_v(data, col_1, col_2)
        for col_1 in categorical_columns
        for col_2 in categorical_columns
        if col_1 != col_2 and (col_2, col_1) not in processed_pairs and not processed_pairs.add((col_1, col_2))
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
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) if n > 1 else 0
    rcorr = r - ((r-1)**2)/(n-1) if n > 1 else 0
    kcorr = k - ((k-1)**2)/(n-1) if n > 1 else 0
    result = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else 0
    return result

def relationships_cat_vs_num(data: pd.DataFrame, categorical_columns: List[str], numerical_columns: List[str]) -> pd.DataFrame:
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
            'Categorical Variable': cat_col,
            'Numerical Variable': num_col,
            'Test Used': test_used,
            'P-Value': p_value
        }
        for cat_col in categorical_columns
        for num_col in numerical_columns
        for p_value, test_used in [best_test_for_categorical(data, num_col, cat_col)]
    ]

    results = pd.DataFrame(tests)
    return results

def best_test_for_categorical(data: pd.DataFrame, numerical_col: str, categorical_col: str) -> Tuple[float, str]:
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
        < stats.anderson(data[data[categorical_col] == group][numerical_col]).critical_values[2]
        for group in data[categorical_col].unique()
    ]

    # Convert boolean results to p-values-like structure
    normality_pvalues = [1.0 if result else 0.0 for result in normality_pvalues]
    
    # Check Homogeneity of Variances with Levene's test
    grouped_data = [data[data[categorical_col] == group][numerical_col] for group in data[categorical_col].unique()]
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

def discretize_and_calculate_cramers_v(data: pd.DataFrame, feature: str, target: str, bins: int, strategy: str) -> Tuple[float, Any]:
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
    if strategy == 'equal_frequency':
        discretizer = EqualFrequencyDiscretiser(q=bins, variables=[feature])
        discretized_feature = discretizer.fit_transform(data[[feature]])
    elif strategy == 'equal_width':
        discretizer = EqualWidthDiscretiser(bins=bins, variables=[feature])
        discretized_feature = discretizer.fit_transform(data[[feature]])
    elif strategy == 'decision_tree':
        discretizer = DecisionTreeDiscretiser(cv=5, scoring='balanced_accuracy', 
                                              variables=[feature], 
                                              regression=False, 
                                              param_grid={'max_depth': [2, 3, 4, 5]},
                                              random_state=cfg.SEED,
                                              bin_output = 'boundaries'
                                              )
        discretized_feature = discretizer.fit_transform(data[[feature]], data[target])
    elif strategy == 'geometric_width':
        discretizer = GeometricWidthDiscretiser(bins=bins, variables=[feature])
        discretized_feature = discretizer.fit_transform(data[[feature]])
    else:
        return 0.0, None
    
    cramers_v = calculate_cramers_v(pd.concat([discretized_feature, data[target]], axis=1), feature, target)
    return cramers_v, discretizer

def find_best_discretization_strategy(data: pd.DataFrame, feature: str, target: str, bin_range: range, threshold: float) -> Dict[str, Any]:
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
    best_strategy = ''
    best_discretizer = None
    
    strategies = ['decision_tree','equal_frequency', 'equal_width', 'geometric_width']
    
    for bins in bin_range:
        for strategy in strategies:
            try:
                current_cramers_v, discretizer = discretize_and_calculate_cramers_v(data, feature, target, bins, strategy)
                if current_cramers_v > best_cramers_v * (1 + threshold):
                    best_cramers_v = current_cramers_v
                    best_bins = bins
                    best_strategy = strategy
                    best_discretizer = discretizer
            except ValueError:
                continue
    
    return {
        'best_cramers_v': best_cramers_v,
        'best_bins': best_bins,
        'best_strategy': best_strategy,
        'best_discretizer': best_discretizer
    }

def analyze_features(data: pd.DataFrame, target: str, bin_range: range, threshold: float) -> Dict[str, Dict[str, Any]]:
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
        best_results[feature] = find_best_discretization_strategy(data, feature, target, bin_range, threshold)
    
    return best_results

def print_best_results(best_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print the best results for the most relevant features based on Cramér's V.
    
    Parameters:
    best_results (Dict[str, Dict[str, Any]]): The best results for each feature.
    """
    sorted_features = sorted(best_results.items(), key=lambda x: x[1]['best_cramers_v'], reverse=True)
    for feature, result in sorted_features:
        print(f'Feature: {feature}, Best Cramér\'s V: {result["best_cramers_v"]:.4f} with {result["best_bins"]} bins using {result["best_strategy"]} strategy')

def fit_best_discretizers(data: pd.DataFrame, best_results: Dict[str, Dict[str, Any]], target: str) -> Dict[str, Any]:
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
        strategy = result['best_strategy']
        bins = result['best_bins']
        if strategy == 'equal_frequency':
            discretizer = EqualFrequencyDiscretiser(q=bins, variables=[feature])
            discretizer.fit(data[[feature]])
        elif strategy == 'equal_width':
            discretizer = EqualWidthDiscretiser(bins=bins, variables=[feature])
            discretizer.fit(data[[feature]])
        elif strategy == 'decision_tree':
            discretizer = DecisionTreeDiscretiser(cv=5, scoring='balanced_accuracy',variables=[feature], regression=False,random_state=cfg.SEED)
            discretizer.fit(data[[feature]], data[target])
        elif strategy == 'geometric_width':
            discretizer = GeometricWidthDiscretiser(bins=bins, variables=[feature])
            discretizer.fit(data[[feature]])
        fitted_discretizers[feature] = discretizer
    
    return fitted_discretizers

def create_binned_dataset(data: pd.DataFrame, fitted_discretizers: Dict[str, Any], target: str) -> pd.DataFrame:
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
        #binned_feature.columns = [f'binned_{feature}']
        binned_data = pd.concat([binned_data, binned_feature], axis=1)
    
    return binned_data

def compare_variances(best_results: dict, train_features: pd.DataFrame, column_to_discretize: str) -> pd.DataFrame:
    """
    Compare the variances of the original column and within each bin after discretization.

    Parameters:
    - best_results: dict, contains the best discretizer for the column
    - train_features: pd.DataFrame, the dataset containing the column to be discretized
    - column_to_discretize: str, the name of the column to be discretized

    Returns:
    - pd.DataFrame, comparison of variances including bin edges
    """
    best_discretizer = best_results[column_to_discretize]['best_discretizer']

    # Retrieve bin edges from best_discretizer
    bin_edges = best_discretizer.binner_dict_[column_to_discretize]

    # Discretize the chosen column into bins using the bin edges and calculate variances
    comparison = (
        train_features
        .assign(
            binned_column=pd.cut(train_features[column_to_discretize], bins=bin_edges, labels=False, include_lowest=True)
        )
        .groupby('binned_column')[column_to_discretize]
        .var()
        .reset_index(name='Within Bin Variance')
        .assign(
            Bin=lambda df: df['binned_column'],
            Bin_Edges=lambda df: df['binned_column'].apply(lambda x: f"{bin_edges[x]} - {bin_edges[x+1]}"),
            Original_Variance=train_features[column_to_discretize].var()
        )
        .drop(columns=['binned_column'])
        .reindex(columns=['Bin', 'Bin_Edges', 'Within Bin Variance', 'Original_Variance'])
    )

    return comparison