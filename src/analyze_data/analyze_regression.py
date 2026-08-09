import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path


MAINTENANCE_SCORE_COLUMN = "software_maintenance_efficiency_score"

DOCUMENTATION_QUALITY_SCORE_COLUMN = "documentation_quality_score"

DOCUMENTATION_METRIC_COLUMNS = [
    "documentation_completeness_score",
    "installation_guidance_score",
    "usage_guidance_score",
    "configuration_documentation_score",
    "dependency_documentation_score",
    "documentation_readability_score",
    "documentation_clarity_score",
    "documentation_coverage_score",
    "deployment_guidance_score",
    "troubleshooting_support_score",
    "architecture_documentation_score",
    "documentation_consistency_score",
    "example_availability_score",
    "command_availability_score",
    "version_information_score"
]


# Read final integrated Parquet file
def read_parquet_file(parquet_file_path):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    print(f"\nReading Parquet file: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(list(df.columns))

    return df, parquet_file_path


# Validate required columns
# Must have "repo_id" and "software_maintenance_efficiency_score" columns
# Must have at least one documentation quality metric column
def validate_required_columns(df):
    required_columns = [
        "repo_id",
        MAINTENANCE_SCORE_COLUMN
    ]

    missing_required_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_required_columns:
        raise ValueError(f"Missing required columns: {missing_required_columns}")

    available_documentation_metric_columns = [
        column for column in DOCUMENTATION_METRIC_COLUMNS
        if column in df.columns
    ]

    if not available_documentation_metric_columns:
        raise ValueError("No documentation quality metric columns found in the dataset.")

    print("\nDocumentation quality metric columns found:")
    print(available_documentation_metric_columns)

    return available_documentation_metric_columns


# Normalize repo_id column
def normalize_repo_id_column(df):
    df = df.copy()
    df["repo_id"] = df["repo_id"].astype(str).str.strip().str.lower()
    return df


# Convert analysis columns to numeric
def convert_analysis_columns_to_numeric(df, documentation_metric_columns):
    df = df.copy()

    analysis_columns = documentation_metric_columns + [MAINTENANCE_SCORE_COLUMN]

    for column in analysis_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# Create documentation quality score
# This score is calculated using the average of all selected documentation quality metrics
def create_documentation_quality_score(df, documentation_metric_columns):
    df = df.copy()

    df[DOCUMENTATION_QUALITY_SCORE_COLUMN] = df[
        documentation_metric_columns
    ].mean(axis=1)

    print("\nDocumentation Quality Score Created Successfully.")
    print(f"Created Column: {DOCUMENTATION_QUALITY_SCORE_COLUMN}")

    return df


# Prepare regression dataframe
# Drop rows with missing documentation quality score or maintenance efficiency score
def prepare_regression_dataframe(df):
    regression_df = df[
        [
            "repo_id",
            DOCUMENTATION_QUALITY_SCORE_COLUMN,
            MAINTENANCE_SCORE_COLUMN
        ]
    ].copy()

    rows_before_drop = len(regression_df)

    regression_df = regression_df.dropna(
        subset=[
            DOCUMENTATION_QUALITY_SCORE_COLUMN,
            MAINTENANCE_SCORE_COLUMN
        ]
    )

    rows_after_drop = len(regression_df)

    print("\nRegression Dataframe Prepared Successfully.")
    print(f"Rows with missing Documentation Quality Score or Maintenance Efficiency Score are Dropped.")
    print(f"Rows before dropping missing values: {rows_before_drop}")
    print(f"Rows after dropping missing values: {rows_after_drop}")

    if rows_after_drop < 3:
        raise ValueError("Not enough valid rows for regression analysis.")

    if regression_df[DOCUMENTATION_QUALITY_SCORE_COLUMN].nunique() <= 1:
        raise ValueError("Documentation quality score has no variation. Regression cannot be performed.")

    if regression_df[MAINTENANCE_SCORE_COLUMN].nunique() <= 1:
        raise ValueError("Software maintenance efficiency score has no variation. Regression cannot be performed.")

    return regression_df


# Run simple linear regression
# Dependent variable: software maintenance efficiency score
# Independent variable: documentation quality score
def run_simple_linear_regression(regression_df):
    print("\nRunning Simple Linear Regression...")

    x = regression_df[DOCUMENTATION_QUALITY_SCORE_COLUMN]
    y = regression_df[MAINTENANCE_SCORE_COLUMN]

    x = sm.add_constant(x)

    regression_model = sm.OLS(y, x).fit()

    print("Simple Linear Regression Completed Successfully.")

    return regression_model


# Create regression results dataframe
def create_regression_results_dataframe(regression_model, regression_df):
    intercept = regression_model.params["const"]
    documentation_quality_coefficient = regression_model.params[DOCUMENTATION_QUALITY_SCORE_COLUMN]

    intercept_p_value = regression_model.pvalues["const"]
    documentation_quality_p_value = regression_model.pvalues[DOCUMENTATION_QUALITY_SCORE_COLUMN]

    intercept_standard_error = regression_model.bse["const"]
    documentation_quality_standard_error = regression_model.bse[DOCUMENTATION_QUALITY_SCORE_COLUMN]

    intercept_t_value = regression_model.tvalues["const"]
    documentation_quality_t_value = regression_model.tvalues[DOCUMENTATION_QUALITY_SCORE_COLUMN]

    r_squared = regression_model.rsquared
    adjusted_r_squared = regression_model.rsquared_adj
    f_statistic = regression_model.fvalue
    f_p_value = regression_model.f_pvalue

    sample_size = len(regression_df)

    if documentation_quality_p_value < 0.05:
        significance_result = "Statistically Significant"
        hypothesis_decision = "Alternative Hypothesis Supported"
    else:
        significance_result = "Not Statistically Significant"
        hypothesis_decision = "Alternative Hypothesis Not Supported"

    regression_results_df = pd.DataFrame([
        {
            "model": "Simple Linear Regression",
            "dependent_variable": MAINTENANCE_SCORE_COLUMN,
            "independent_variable": DOCUMENTATION_QUALITY_SCORE_COLUMN,
            "sample_size": sample_size,
            "intercept": intercept,
            "intercept_p_value": intercept_p_value,
            "intercept_standard_error": intercept_standard_error,
            "intercept_t_value": intercept_t_value,
            "documentation_quality_coefficient": documentation_quality_coefficient,
            "documentation_quality_p_value": documentation_quality_p_value,
            "documentation_quality_standard_error": documentation_quality_standard_error,
            "documentation_quality_t_value": documentation_quality_t_value,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "f_statistic": f_statistic,
            "f_p_value": f_p_value,
            "significance_result": significance_result,
            "hypothesis_decision": hypothesis_decision
        }
    ])

    return regression_results_df


# Save dataset with documentation quality score
def save_dataset_with_documentation_quality_score(df, output_folder_path):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    dataset_output_file_path = output_folder_path / "final_analyzed_dataset.parquet"

    df.to_parquet(
        dataset_output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nDataset with Documentation Quality Score Saved Successfully.")
    print(f"Parquet output file: {dataset_output_file_path}")

    return dataset_output_file_path


# Save regression results as a Parquet file
def save_regression_results(regression_results_df, output_folder_path):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    parquet_output_file_path = output_folder_path / "regression_results.parquet"

    regression_results_df.to_parquet(
        parquet_output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nRegression Results Saved Successfully.")
    print(f"Parquet output file: {parquet_output_file_path}")

    return parquet_output_file_path


# Format figure label for better readability
def format_figure_label(column_name):
    return (
        column_name
        .replace("_score", "")
        .replace("_", " ")
        .title()
    )


# Create regression scatter plot
def create_regression_scatter_plot(regression_df, output_folder_path):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    regression_plot_output_file_path = output_folder_path / "regression_scatter_plot.png"

    plot_df = regression_df.rename(
        columns=lambda column_name: format_figure_label(column_name)
    )

    x_column = format_figure_label(DOCUMENTATION_QUALITY_SCORE_COLUMN)
    y_column = format_figure_label(MAINTENANCE_SCORE_COLUMN)

    plt.figure(figsize=(10, 7))

    sns.regplot(
        data=plot_df,
        x=x_column,
        y=y_column,
        scatter_kws={
            "alpha": 0.7,
            "s": 45
        },
        line_kws={
            "linewidth": 2
        }
    )

    plt.xlabel("Documentation Quality")
    plt.ylabel("Software Maintenance Efficiency")
    plt.tight_layout()

    plt.savefig(
        regression_plot_output_file_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Regression Scatter Plot saved to: {regression_plot_output_file_path}")

    return regression_plot_output_file_path


# Save Markdown summary
def save_regression_analysis_summary_markdown(
    regression_df,
    regression_results_df,
    input_file_path,
    output_folder_path,
    output_files
):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "regression_analysis_summary.md"

    result_row = regression_results_df.iloc[0]

    coefficient = result_row["documentation_quality_coefficient"]
    p_value = result_row["documentation_quality_p_value"]
    r_squared = result_row["r_squared"]
    adjusted_r_squared = result_row["adjusted_r_squared"]
    sample_size = result_row["sample_size"]
    significance_result = result_row["significance_result"]
    hypothesis_decision = result_row["hypothesis_decision"]

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# Regression Analysis Summary\n\n")

        file.write("## Input File\n\n")
        file.write(f"- `{input_file_path}`\n\n")

        file.write("## Regression Method\n\n")
        file.write("Simple Linear Regression was used to examine whether Documentation Quality predicts Software Maintenance Efficiency.\n\n")

        file.write("## Regression Model\n\n")
        file.write("```text\n")
        file.write("software_maintenance_efficiency_score = β0 + β1(documentation_quality_score)\n")
        file.write("```\n\n")

        file.write("## Dependent Variable\n\n")
        file.write(f"- `{MAINTENANCE_SCORE_COLUMN}`\n\n")

        file.write("## Independent Variable\n\n")
        file.write(f"- `{DOCUMENTATION_QUALITY_SCORE_COLUMN}`\n\n")

        file.write("## Dataset Summary\n\n")
        file.write("| Item | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Repositories used for regression | {len(regression_df)} |\n")
        file.write(f"| Valid sample size | {sample_size} |\n\n")

        file.write("## Results\n\n")
        file.write("| Metric | Value |\n")
        file.write("|---|---:|\n")
        file.write(f"| Regression coefficient | {coefficient:.6f} |\n")
        file.write(f"| p-value | {p_value:.6f} |\n")
        file.write(f"| R-squared | {r_squared:.6f} |\n")
        file.write(f"| Adjusted R-squared | {adjusted_r_squared:.6f} |\n")
        file.write(f"| Significance result | {significance_result} |\n\n")

        file.write("## Interpretation\n\n")

        file.write(
            f"The regression coefficient for Documentation Quality Score is {coefficient:.6f}. "
        )

        if coefficient > 0:
            file.write(
                "This indicates a positive relationship between documentation quality and software maintenance efficiency. "
            )
        elif coefficient < 0:
            file.write(
                "This indicates a negative relationship between documentation quality and software maintenance efficiency. "
            )
        else:
            file.write(
                "This indicates no directional relationship between documentation quality and software maintenance efficiency. "
            )

        file.write(
            f"The p-value is {p_value:.6f}. Therefore, the relationship is {significance_result.lower()} at the 0.05 significance level. "
        )

        file.write(
            f"The R-squared value is {r_squared:.6f}, meaning the regression model explains approximately {r_squared * 100:.2f}% of the variation in software maintenance efficiency.\n\n"
        )

        file.write("## Hypothesis Decision\n\n")
        file.write(f"- {hypothesis_decision}\n\n")

        file.write("## Output Files\n\n")
        for output_name, output_path in output_files.items():
            file.write(f"- **{output_name}:** `{output_path}`\n")

    print(f"Regression Analysis Summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to run regression analysis
def regression_analysis(
    input_file_path,
    output_folder_path
):
    original_df, input_file_path = read_parquet_file(input_file_path)

    original_df = normalize_repo_id_column(original_df)

    documentation_metric_columns = validate_required_columns(original_df)

    original_df = convert_analysis_columns_to_numeric(
        df=original_df,
        documentation_metric_columns=documentation_metric_columns
    )

    dataset_with_documentation_quality_score_df = create_documentation_quality_score(
        df=original_df,
        documentation_metric_columns=documentation_metric_columns
    )

    dataset_output_file_path = save_dataset_with_documentation_quality_score(
        df=dataset_with_documentation_quality_score_df,
        output_folder_path=output_folder_path
    )

    regression_df = prepare_regression_dataframe(
        df=dataset_with_documentation_quality_score_df
    )

    regression_model = run_simple_linear_regression(
        regression_df=regression_df
    )

    regression_results_df = create_regression_results_dataframe(
        regression_model=regression_model,
        regression_df=regression_df
    )

    regression_results_output_file_path = save_regression_results(
        regression_results_df=regression_results_df,
        output_folder_path=output_folder_path
    )

    regression_plot_output_file_path = create_regression_scatter_plot(
        regression_df=regression_df,
        output_folder_path=output_folder_path
    )

    output_files = {
        "Dataset with Documentation Quality Score": dataset_output_file_path,
        "Regression Results Parquet": regression_results_output_file_path,
        "Regression Scatter Plot": regression_plot_output_file_path
    }

    summary_file_path = save_regression_analysis_summary_markdown(
        regression_df=regression_df,
        regression_results_df=regression_results_df,
        input_file_path=input_file_path,
        output_folder_path=output_folder_path,
        output_files=output_files
    )

    return output_files, summary_file_path


if __name__ == "__main__":
    input_file_path = input("Enter the Final Integrated Dataset Parquet file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    regression_analysis(
        input_file_path=input_file_path,
        output_folder_path=output_folder_path
    )