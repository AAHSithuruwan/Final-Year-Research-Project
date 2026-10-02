import numpy as np
from analysis_functions.constants import ALPHA, readable_name


# Fotmat metric values for display, including p-values in scientific notation if very small.
def format_number(value, p_value=False):
    if value is None or not np.isfinite(value):
        return "Not Available"
    if p_value and 0 < abs(value) < 0.0001:
        return f"{value:.3e}"
    return f"{value:.6f}" if p_value else f"{value:.4f}"


# Explain the Spearman correlation result for a single metric, including direction, strength, and significance.
def metric_interpretation(result):
    name = readable_name(result["metric_name"])
    if not np.isfinite(result["spearman_correlation"]):
        return f"{name} cannot be evaluated: {result['status']} Valid sample size: {result['sample_size']}."
    direction = result["direction"].lower()
    relationship = (
        f"a {direction}, {result['strength'].lower()} relationship"
        if direction != "no direction" else "no directional monotonic relationship"
    )
    return (
        f"{name} shows {relationship} with software maintenance efficiency "
        f"(rho = {format_number(result['spearman_correlation'])}, "
        f"p = {format_number(result['p_value'], p_value=True)}, n = {result['sample_size']}). "
        f"The relationship is {result['significance_result'].lower()} at the {ALPHA:.2f} significance level. "
        "This association does not establish causation."
    )


# Summarize the overall correlation pattern, including sign distribution, strength distribution, and the strongest absolute association.
def correlation_pattern(results):
    valid = results.dropna(subset=["spearman_correlation", "p_value"])
    if valid.empty:
        return "No individual metric has sufficient valid observations and variation for a correlation result."
    positive = int((valid["spearman_correlation"] > 0).sum())
    negative = int((valid["spearman_correlation"] < 0).sum())
    zero = int((valid["spearman_correlation"] == 0).sum())
    significant = int((valid["p_value"] < ALPHA).sum())
    distribution = ", ".join(
        f"{count} {strength.lower()}" for strength, count in valid["strength"].value_counts().items()
    )
    strongest = valid.loc[valid["spearman_correlation"].abs().idxmax()]
    return (
        f"Among {len(valid)} evaluable metrics, {positive} correlations are positive, "
        f"{negative} are negative, and {zero} have no direction. Strengths: {distribution}. "
        f"{significant} are statistically significant at alpha = {ALPHA:.2f}. "
        f"The strongest association by absolute rho is {readable_name(strongest['metric_name'])} "
        f"(rho = {format_number(strongest['spearman_correlation'])}, "
        f"p = {format_number(strongest['p_value'], p_value=True)}, n = {strongest['sample_size']}). "
        f"{len(results) - len(valid)} metrics could not be evaluated."
    )


# Explain the ranked ordering of metrics based on their Spearman correlations.
def ranked_explanation(results):
    valid = results.dropna(subset=["spearman_correlation"]).sort_values("spearman_correlation", ascending=False)
    if valid.empty:
        return "No valid correlations are available. Unavailable metrics are marked separately."
    highest, lowest = valid.iloc[0], valid.iloc[-1]
    positives = valid[valid["spearman_correlation"] > 0]
    positive_text = "No evaluable metric has a positive correlation."
    if not positives.empty:
        top = positives.iloc[0]
        positive_text = (
            f"The highest positive correlation is {readable_name(top['metric_name'])} "
            f"(rho = {format_number(top['spearman_correlation'])})."
        )
    return (
        f"Metrics are ordered by signed rho from highest to lowest. {positive_text} "
        f"The highest signed value is {readable_name(highest['metric_name'])} "
        f"({format_number(highest['spearman_correlation'])}); the lowest is "
        f"{readable_name(lowest['metric_name'])} ({format_number(lowest['spearman_correlation'])}). "
        "Ranking alone does not establish statistical significance or causation."
    )


# Explain the overall pattern of pairwise correlations among documentation metrics, including the number of strong associations and the median absolute correlation.
def full_heatmap_explanation(matrix):
    documentation = matrix.iloc[:-1, :-1]
    upper = documentation.to_numpy()[np.triu_indices(len(documentation), k=1)]
    valid = upper[np.isfinite(upper)]
    if len(valid) == 0:
        return "No documentation metric pairs have a defined correlation. Blank cells indicate unavailable results."
    strong_count = int((np.abs(valid) >= 0.60).sum())
    return (
        f"Of {len(valid)} evaluable pairs of documentation metrics, {strong_count} have strong or "
        f"very strong associations (absolute rho >= 0.60). The median absolute pairwise rho is "
        f"{format_number(np.median(np.abs(valid)))}. "
        "Related documentation metrics may capture overlapping characteristics. "
        "Cells use pairwise complete observations; blank cells are undefined. "
        "Color and magnitude alone do not indicate statistical significance."
    )


# Explain the regression results, including coefficient direction, significance, R-squared, and the overall hypothesis decision.
def regression_interpretation(result):
    if result is None:
        return "Regression is unavailable, so the overall hypothesis cannot be evaluated."
    coefficient = result["regression_coefficient"]
    direction = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "non-directional"
    text = (
        f"The regression coefficient ({format_number(coefficient)}) indicates a {direction} "
        "relationship between documentation quality and software maintenance efficiency. "
    )
    p_value = result["coefficient_p_value"]
    if not np.isfinite(p_value):
        return text + "The coefficient p-value is undefined; statistical significance and the hypothesis cannot be evaluated."
    text += (
        f"The relationship is {result['significance_result'].lower()} at alpha = {ALPHA:.2f} "
        f"(p = {format_number(p_value, p_value=True)}, n = {result['sample_size']}). "
        f"R-squared = {format_number(result['r_squared'])}: the fitted model explains approximately "
        f"{result['r_squared'] * 100:.2f}% of the observed variation in maintenance efficiency. "
    )
    if p_value < ALPHA:
        text += "The alternative hypothesis is supported within this dataset and model. "
    else:
        text += "The available evidence is insufficient to support the alternative hypothesis. "
    return text + "These observational results do not establish a causal effect."


# Combine individual associations and the prespecified overall OLS decision.
def final_interpretation(results, regression_result):
    text = correlation_pattern(results) + "\n\n" + regression_interpretation(regression_result)
    if regression_result is None or not np.isfinite(regression_result["coefficient_p_value"]):
        return text + "\n\nNo overall hypothesis decision can be made while regression is unavailable."
    if regression_result["coefficient_p_value"] >= ALPHA:
        text += (
            "\n\nThe analysis does not provide sufficient statistical evidence to support the "
            "alternative hypothesis within the selected dataset. This result does not mean that "
            "technical documentation quality is unimportant. It indicates that, within the selected "
            "dataset and measurement approach, documentation quality did not demonstrate a "
            "statistically significant relationship with software maintenance efficiency. "
            "The null hypothesis has not been proven."
        )
    return text
