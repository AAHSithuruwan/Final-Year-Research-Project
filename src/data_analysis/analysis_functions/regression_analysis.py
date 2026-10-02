import numpy as np
import statsmodels.api as sm
from analysis_functions.constants import ALPHA, DOCUMENTATION_QUALITY_SCORE_COLUMN, MAINTENANCE_SCORE_COLUMN
from utils.dataset_validation import numeric_analysis_columns


# Prepare the regression dataframe by selecting the relevant columns and dropping rows with missing values
def prepare_regression_dataframe(dataframe):
    columns = [DOCUMENTATION_QUALITY_SCORE_COLUMN, MAINTENANCE_SCORE_COLUMN]
    numeric, _ = numeric_analysis_columns(dataframe, columns)
    regression_df = numeric[["repo_id", *columns]].dropna(subset=columns).copy()
    if len(regression_df) < 3:
        raise ValueError(f"Regression needs at least 3 valid X/Y observations; {len(regression_df)} are available.")
    for column in columns:
        if regression_df[column].nunique() <= 1:
            raise ValueError(f"Regression cannot be calculated: {column} has no variation in valid observations.")
    return regression_df


# Fit a simple linear regression model using OLS and return the fitted model
def run_simple_linear_regression(regression_df):
    design = sm.add_constant(regression_df[DOCUMENTATION_QUALITY_SCORE_COLUMN], has_constant="add")
    try:
        model = sm.OLS(regression_df[MAINTENANCE_SCORE_COLUMN], design, missing="raise").fit()
    except Exception as error:
        raise ValueError("Regression could not be fitted. Check the numeric values and variation in X and Y.") from error
    if model.model.rank < 2:
        raise ValueError("The regression design is numerically singular; the score needs sufficient variation.")
    return model


# Extract regression results, including p-values and significance decisions
def regression_results(model):
    score = DOCUMENTATION_QUALITY_SCORE_COLUMN
    p_value = float(model.pvalues[score])
    if not np.isfinite(p_value):
        significance = "Not Available"
        decision = "Hypothesis Cannot Be Evaluated"
    elif p_value < ALPHA:
        significance = "Statistically Significant"
        decision = "Alternative Hypothesis Supported"
    else:
        significance = "Not Statistically Significant"
        decision = "Alternative Hypothesis Not Supported"
    return {
        "sample_size": int(model.nobs),
        "intercept": float(model.params["const"]),
        "regression_coefficient": float(model.params[score]),
        "coefficient_p_value": p_value,
        "coefficient_standard_error": float(model.bse[score]),
        "coefficient_t_statistic": float(model.tvalues[score]),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "significance_result": significance,
        "hypothesis_decision": decision,
    }


# Analyze regression by preparing the dataframe, fitting the model, and returning results
def analyze_regression(dataframe):
    regression_df = prepare_regression_dataframe(dataframe)
    model = run_simple_linear_regression(regression_df)
    return regression_df, model, regression_results(model)
