import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis_functions.constants import ALPHA, DOCUMENTATION_METRIC_COLUMNS, MAINTENANCE_SCORE_COLUMN
from utils.dataset_validation import numeric_analysis_columns


# Correlation strength labels based on absolute value of Spearman's rho
def get_correlation_strength(correlation_value):
    if not np.isfinite(correlation_value):
        return "Not Available"
    magnitude = abs(correlation_value)
    for cutoff, label in [(0.2, "Very Weak"), (0.4, "Weak"), (0.6, "Moderate"), (0.8, "Strong")]:
        if magnitude < cutoff:
            return label
    return "Very Strong"


# Calculate Spearman correlation and p-value for a given metric column against the maintenance score column
def calculate_correlation_for_metric(dataframe, metric_column):
    numeric, _ = numeric_analysis_columns(dataframe, [metric_column, MAINTENANCE_SCORE_COLUMN])
    pair = numeric[[metric_column, MAINTENANCE_SCORE_COLUMN]].dropna()
    result = {
        "metric_name": metric_column,
        "spearman_correlation": np.nan,
        "p_value": np.nan,
        "sample_size": len(pair),
        "direction": "Not Available",
        "strength": "Not Available",
        "significance_result": "Not Available",
        "status": "",
    }
    if len(pair) < 3:
        result["status"] = "At least 3 valid paired observations are required."
        return result
    if pair.nunique().min() <= 1:
        result["status"] = "The metric or maintenance target has no variation in valid pairs."
        return result
    rho, p_value = spearmanr(pair[metric_column], pair[MAINTENANCE_SCORE_COLUMN])
    if not np.isfinite(rho) or not np.isfinite(p_value):
        result["status"] = "Spearman correlation or its p-value is undefined."
        return result
    result.update({
        "spearman_correlation": float(rho),
        "p_value": float(p_value),
        "direction": "Positive" if rho > 0 else "Negative" if rho < 0 else "No Direction",
        "strength": get_correlation_strength(rho),
        "significance_result": "Statistically Significant" if p_value < ALPHA else "Not Statistically Significant",
        "status": "Calculated",
    })
    return result


# Calculate correlations for all specified metric columns and return a sorted DataFrame of results
def calculate_correlations_for_all_metrics(dataframe, metric_columns=None):
    columns = DOCUMENTATION_METRIC_COLUMNS if metric_columns is None else metric_columns
    results = [calculate_correlation_for_metric(dataframe, column) for column in columns]
    return pd.DataFrame(results).sort_values(
        "spearman_correlation", ascending=False, na_position="last", kind="stable", ignore_index=True
    )


# Generate a full pairwise Spearman correlation matrix for the specified metric columns and the maintenance score column
def full_correlation_matrix(dataframe, metric_columns=None):
    metrics = DOCUMENTATION_METRIC_COLUMNS if metric_columns is None else metric_columns
    columns = [*metrics, MAINTENANCE_SCORE_COLUMN]
    numeric, _ = numeric_analysis_columns(dataframe, columns)
    matrix = numeric[columns].corr(method="spearman", min_periods=3)
    available = numeric[columns].notna().astype(int)
    sample_sizes = available.T.dot(available)
    return matrix, sample_sizes
