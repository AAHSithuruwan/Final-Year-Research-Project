import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, ConstantInputWarning


MAINTENANCE_SCORE_COLUMN = "software_maintenance_efficiency_score"

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
# Drop rows with missing maintenance efficiency score
def prepare_analysis_dataframe(df, documentation_metric_columns):
    df = df.copy()

    analysis_columns = documentation_metric_columns + [MAINTENANCE_SCORE_COLUMN]

    for column in analysis_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    analysis_df = df[["repo_id"] + analysis_columns].copy()

    rows_before_drop = len(analysis_df)

    analysis_df = analysis_df.dropna(
        subset=[MAINTENANCE_SCORE_COLUMN]
    )

    rows_after_drop = len(analysis_df)

    print("\nAnalysis Dataframe Prepared Successfully.")
    print(f"Rows before dropping maintenance score missing rows: {rows_before_drop}")
    print(f"Rows after dropping maintenance score missing rows: {rows_after_drop}")

    return analysis_df


# Get correlation strength
def get_correlation_strength(correlation_value):
    if pd.isna(correlation_value):
        return "Not available"

    # Use absolute value of correlation for strength
    absolute_value = abs(correlation_value)

    if absolute_value < 0.20:
        return "Very weak"
    elif absolute_value < 0.40:
        return "Weak"
    elif absolute_value < 0.60:
        return "Moderate"
    elif absolute_value < 0.80:
        return "Strong"
    else:
        return "Very strong"


# Calculate Spearman correlation for one documentation metric
def calculate_correlation_for_metric(
    analysis_df,
    metric_column
):
    # Keep only the Metric Column and the Maintenance Score Column
    metric_target_df = analysis_df[
        [metric_column, MAINTENANCE_SCORE_COLUMN]
    ].dropna()

    sample_size = len(metric_target_df)

    # If there are fewer than 3 valid samples, return NaN for correlation and p-value
    if sample_size < 3:
        return {
            "metric_name": metric_column,
            "sample_size": sample_size,
            "spearman_correlation": np.nan,
            "spearman_p_value": np.nan,
            "absolute_spearman_correlation": np.nan,
            "correlation_strength": "Not available",
        }

    metric_values = metric_target_df[metric_column]
    target_values = metric_target_df[MAINTENANCE_SCORE_COLUMN]

    # If the metric values or target values have no variation (all values are the same), 
    # return NaN for correlation and p-value
    if metric_values.nunique() <= 1 or target_values.nunique() <= 1:
        return {
            "metric_name": metric_column,
            "sample_size": sample_size,
            "spearman_correlation": np.nan,
            "spearman_p_value": np.nan,
            "absolute_spearman_correlation": np.nan,
            "correlation_strength": "Not available"
        }

    # Calculate Spearman correlation and p-value
    spearman_correlation, spearman_p_value = spearmanr(
        metric_values,
        target_values
    )

    # Calculate absolute Spearman correlation
    absolute_spearman_correlation = abs(spearman_correlation)

    # Get correlation strength based on absolute Spearman correlation
    correlation_strength = get_correlation_strength(
        absolute_spearman_correlation
    )

    return {
        "metric_name": metric_column,
        "sample_size": sample_size,
        "spearman_correlation": spearman_correlation,
        "spearman_p_value": spearman_p_value,
        "absolute_spearman_correlation": absolute_spearman_correlation,
        "correlation_strength": correlation_strength
    }


# Calculate Spearman correlation for all documentation metrics
def calculate_correlations_for_all_metrics(
    analysis_df,
    documentation_metric_columns
):
    correlation_rows = []

    print("\nCalculating Spearman Correlation between Documentation Quality Metrics and Software Maintenance Score...")

    for metric_column in documentation_metric_columns:
        print(f"Analyzing Metric: {metric_column}")

        correlation_row = calculate_correlation_for_metric(
            analysis_df=analysis_df,
            metric_column=metric_column
        )

        correlation_rows.append(correlation_row)

    correlation_results_df = pd.DataFrame(correlation_rows)

    print("\nSpearman Correlation Analysis Completed Successfully.")

    return correlation_results_df


# Save correlation results as a Parquet File
def save_correlation_results(correlation_results_df, output_folder_path):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    parquet_output_file_path = output_folder_path / "spearman_correlation_results.parquet"

    correlation_results_df.to_parquet(
        parquet_output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nSpearman Correlation Results Saved Successfully.")
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


# Create full Spearman correlation heatmap
# This measures the correlation between all documentation quality metrics and the software maintenance efficiency score
def create_full_correlation_heatmap(
    analysis_df,
    documentation_metric_columns,
    output_folder_path
):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    heatmap_columns = documentation_metric_columns + [MAINTENANCE_SCORE_COLUMN]

    heatmap_df = analysis_df[heatmap_columns].copy()

    heatmap_df = heatmap_df.rename(
        columns=lambda column_name: format_figure_label(column_name)
    )

    correlation_matrix = heatmap_df.corr(method="spearman")

    heatmap_output_file_path = output_folder_path / "full_spearman_correlation_heatmap.png"

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=False,
        cbar=True
    )

    # plt.title("Full Spearman Correlation Heatmap: Documentation Quality Metrics and Maintenance Efficiency Score")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        heatmap_output_file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Full Spearman Correlation Heatmap saved to: {heatmap_output_file_path}")

    return heatmap_output_file_path


# Create focused speraman correlation heatmap
# This mesasures the correlation between each documentation quality metric and the software maintenance efficiency score
def create_focused_correlation_heatmap(
    correlation_results_df,
    output_folder_path
):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    focused_correlation_df = correlation_results_df[
        ["metric_name", "spearman_correlation"]
    ].copy()

    focused_correlation_df = focused_correlation_df.set_index("metric_name")

    focused_correlation_df = focused_correlation_df.rename(
        index=lambda metric_name: format_figure_label(metric_name)
    )

    focused_correlation_df.index.name = "Documentation Quality Metric"

    focused_correlation_df = focused_correlation_df.rename(
        columns={
            "spearman_correlation": "Spearman Correlation"
        }
    )

    focused_correlation_output_file_path = output_folder_path / "focused_spearman_correlation_heatmap.png"

    plt.figure(figsize=(8, max(6, len(focused_correlation_df) * 0.4)))

    sns.heatmap(
        focused_correlation_df,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        cbar_kws={
            "pad": 0.08
        }
    )

    # plt.title("Focused Spearman Correlation Heatmap: Documentation Quality Metrics vs Software Maintenance Efficiency Score")
    # plt.xlabel("Software Maintenance Efficiency Score")
    # plt.ylabel("Documentation Quality Metrics")
    plt.tight_layout()

    plt.savefig(
        focused_correlation_output_file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Focused Spearman Correlation Heatmap saved to: {focused_correlation_output_file_path}")

    return focused_correlation_output_file_path


# Create ranked Spearman correlation bar chart
def create_ranked_correlation_bar_chart(
    correlation_results_df,
    output_folder_path
):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    bar_chart_df = correlation_results_df.dropna(
        subset=["spearman_correlation"]
    ).copy()

    bar_chart_df = bar_chart_df.sort_values(
        by="spearman_correlation",
        ascending=False
    )

    # Create readable metric labels only for display
    bar_chart_df["display_metric_name"] = bar_chart_df["metric_name"].apply(
        format_figure_label
    )

    bar_chart_output_file_path = output_folder_path / "spearman_correlation_bar_chart.png"

    plt.figure(figsize=(12, max(7, len(bar_chart_df) * 0.45)))

    ax = sns.barplot(
        data=bar_chart_df,
        x="spearman_correlation",
        y="display_metric_name"
    )

    plt.axvline(0, linestyle="--", linewidth=1)

    # Add right padding
    x_min = bar_chart_df["spearman_correlation"].min()
    x_max = bar_chart_df["spearman_correlation"].max()
    x_range = x_max - x_min

    if x_range == 0:
        x_range = 0.1

    plt.xlim(
        min(0, x_min) - (x_range * 0.10),
        x_max + (x_range * 0.20)
    )

    plt.xlabel("Spearman Correlation with Software Maintenance Efficiency")
    plt.ylabel("Documentation Quality Metric")
    plt.tight_layout()

    plt.savefig(
        bar_chart_output_file_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Spearman Correlation Bar Chart saved to: {bar_chart_output_file_path}")

    return bar_chart_output_file_path


# Print summary in terminal
def print_correlation_analysis_summary(
    correlation_results_df
):

    print("\nSpearman Correlation Analysis Summary")
    print(f"Metrics Analyzed: {len(correlation_results_df)}")

    print("\nTop Spearman Correlation Results:")
    print(correlation_results_df.head(10).to_string(index=False))


# Save Markdown summary
def save_correlation_analysis_summary_markdown(
    analysis_df,
    correlation_results_df,
    input_file_path,
    output_folder_path,
    output_files
):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "spearman_correlation_analysis_summary.md"

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# Spearman Correlation Analysis Summary\n\n")

        file.write("## Input File\n\n")
        file.write(f"- `{input_file_path}`\n\n")

        file.write("## Correlation Method\n\n")
        file.write("Spearman Correlation was used to measure the relationship between each Documentation Quality Metric and Software Maintenance Efficiency.\n\n")

        file.write("## Target Variable\n\n")
        file.write(f"- `{MAINTENANCE_SCORE_COLUMN}`\n\n")

        file.write("## Dataset Summary\n\n")
        file.write("| Item | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Repositories Analyzed | {len(analysis_df)} |\n")
        file.write(f"| Documentation Quality Metrics Analyzed | {len(correlation_results_df)} |\n")

        file.write("## Analyzed Documentation Quality Metrics\n\n")

        for metric_name in correlation_results_df["metric_name"]:
            file.write(f"- `{metric_name}`\n")

        file.write("\n## Output Files\n\n")
        for output_name, output_path in output_files.items():
            file.write(f"- **{output_name}:** `{output_path}`\n")

    print(f"Spearman Correlation Analysis Summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to run Spearman correlation analysis
def spearman_correlation_analysis(
    input_file_path,
    output_folder_path
):
    original_df, input_file_path = read_parquet_file(input_file_path)

    original_df = normalize_repo_id_column(original_df)

    documentation_metric_columns = validate_required_columns(original_df)

    analysis_df = prepare_analysis_dataframe(
        df=original_df,
        documentation_metric_columns=documentation_metric_columns
    )

    correlation_results_df = calculate_correlations_for_all_metrics(
        analysis_df=analysis_df,
        documentation_metric_columns=documentation_metric_columns
    )

    parquet_output_file_path = save_correlation_results(
        correlation_results_df=correlation_results_df,
        output_folder_path=output_folder_path
    )

    full_heatmap_output_file_path = create_full_correlation_heatmap(
        analysis_df=analysis_df,
        documentation_metric_columns=documentation_metric_columns,
        output_folder_path=output_folder_path
    )

    focused_heatmap_output_file_path = create_focused_correlation_heatmap(
        correlation_results_df=correlation_results_df,
        output_folder_path=output_folder_path
    )

    bar_chart_output_file_path = create_ranked_correlation_bar_chart(
        correlation_results_df=correlation_results_df,
        output_folder_path=output_folder_path
    )

    print_correlation_analysis_summary(
        correlation_results_df=correlation_results_df
    )

    output_files = {
        "Spearman Correlation Results Parquet": parquet_output_file_path,
        "Full Spearman Correlation Heatmap": full_heatmap_output_file_path,
        "Focused Spearman Correlation Heatmap": focused_heatmap_output_file_path,
        "Ranked Spearman Correlation Bar Chart": bar_chart_output_file_path
    }

    summary_file_path = save_correlation_analysis_summary_markdown(
        analysis_df=analysis_df,
        correlation_results_df=correlation_results_df,
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

    spearman_correlation_analysis(
        input_file_path=input_file_path,
        output_folder_path=output_folder_path
    )