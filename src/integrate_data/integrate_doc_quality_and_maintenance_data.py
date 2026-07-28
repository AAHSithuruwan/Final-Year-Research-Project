import pandas as pd
from pathlib import Path


# Read a Parquet file
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


# Validate repo_id column
def validate_repo_id_column(df, file_path):
    if "repo_id" not in df.columns:
        raise ValueError(f"Missing required column 'repo_id' in file: {file_path}")

    return True


# Normalize repo_id column
def normalize_repo_id_column(df):
    df = df.copy()
    df["repo_id"] = df["repo_id"].astype(str).str.strip().str.lower()
    return df


# Read and merge all maintenance metric Parquet files
def merge_maintenance_metric_files(maintenance_file_paths):
    merged_maintenance_df = None

    for maintenance_file_path in maintenance_file_paths:
        df, maintenance_file_path = read_parquet_file(maintenance_file_path)

        validate_repo_id_column(df, maintenance_file_path)

        df = normalize_repo_id_column(df)

        if merged_maintenance_df is None:
            merged_maintenance_df = df
        else:
            merged_maintenance_df = pd.merge(
                merged_maintenance_df,
                df,
                on="repo_id",
                how="outer"
            )

    if merged_maintenance_df is None:
        raise ValueError("No maintenance metric files were provided.")

    print("\nMaintenance Metric Files Merged Successfully.")
    print(f"Merged Maintenance Metric Rows: {len(merged_maintenance_df)}")
    print(f"Merged Maintenance Metric Columns: {len(merged_maintenance_df.columns)}")

    return merged_maintenance_df


# Create maintenance score columns for Average Issue Resolution Time and Average Issue Comment Count
# The lesser the Average Issue Resolution Time and Average Issue Comment Count, the better the Maintenance Score.
# Therefore, we calculate the Maintenance Score as 1 - normalized value of these metrics.
def calculate_maintenance_score_columns(maintenance_df):
    maintenance_df = maintenance_df.copy()

    if "github_normalized_avg_resolution_time_days" in maintenance_df.columns:
        maintenance_df["github_avg_issue_resolution_time_score"] = (
            1 - maintenance_df["github_normalized_avg_resolution_time_days"]
        )

    if "github_normalized_avg_issue_comment_count" in maintenance_df.columns:
        maintenance_df["github_avg_issue_comment_count_score"] = (
            1 - maintenance_df["github_normalized_avg_issue_comment_count"]
        )

    return maintenance_df


# Calculate final average software maintenance efficiency score
def calculate_software_maintenance_efficiency_score(maintenance_df):
    maintenance_df = maintenance_df.copy()

    maintenance_score_columns = []

    if "github_issue_closure_rate" in maintenance_df.columns:
        maintenance_score_columns.append("github_issue_closure_rate")

    if "github_avg_issue_resolution_time_score" in maintenance_df.columns:
        maintenance_score_columns.append("github_avg_issue_resolution_time_score")

    if "github_pr_merge_rate" in maintenance_df.columns:
        maintenance_score_columns.append("github_pr_merge_rate")

    if "github_avg_issue_comment_count_score" in maintenance_df.columns:
        maintenance_score_columns.append("github_avg_issue_comment_count_score")

    if not maintenance_score_columns:
        raise ValueError("No valid maintenance score columns found.")

    maintenance_df["software_maintenance_efficiency_score"] = maintenance_df[maintenance_score_columns].mean(axis=1)

    print("\nSoftware Maintenance Efficiency Score Calculated Successfully.")
    print("Maintenance Efficiency Score Columns used:")
    print(maintenance_score_columns)

    return maintenance_df, maintenance_score_columns


# Select final maintenance columns
def select_final_maintenance_columns(maintenance_df):
    preferred_columns = [
        "repo_id",

        "github_issue_closure_rate",

        "github_avg_issue_resolution_time_score",

        "github_pr_merge_rate",

        "github_avg_issue_comment_count_score",

        "software_maintenance_efficiency_score"
    ]

    available_columns = [
        column for column in preferred_columns
        if column in maintenance_df.columns
    ]

    final_maintenance_df = maintenance_df[available_columns].copy()

    return final_maintenance_df


# Read documentation quality metrics dataset
def read_documentation_quality_metrics_file(documentation_quality_file_path):
    documentation_quality_df, documentation_quality_file_path = read_parquet_file(
        documentation_quality_file_path
    )

    validate_repo_id_column(documentation_quality_df, documentation_quality_file_path)

    documentation_quality_df = normalize_repo_id_column(documentation_quality_df)

    return documentation_quality_df, documentation_quality_file_path


# Integrate maintenance metrics with documentation quality metrics
def integrate_maintenance_and_documentation_quality_metrics(
    maintenance_df,
    documentation_quality_df
):
    integrated_df = pd.merge(
        documentation_quality_df,
        maintenance_df,
        on="repo_id",
        how="inner"
    )

    print("\nDocumentation Quality Metrics and Maintenance Efficiency Metrics Integrated Successfully.")
    print(f"Final Integrated Rows: {len(integrated_df)}")
    print(f"Final Integrated Columns: {len(integrated_df.columns)}")

    return integrated_df


# Save final integrated dataset
def save_final_integrated_dataset(integrated_df, output_folder_path):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "final_integrated_dataset.parquet"

    integrated_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nFinal Integrated Dataset Saved Successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(integrated_df)}")
    print(f"Columns: {len(integrated_df.columns)}")

    return output_file_path


# Print final integrated dataset summary
def print_final_integration_summary(
    integrated_df,
    documentation_quality_columns,
    maintenance_score_columns
):
    print("\nFinal Integrated Dataset Summary")
    print(f"Rows: {len(integrated_df)}")
    print(f"Columns: {len(integrated_df.columns)}")

    print("\nDocumentation Quality Metric Columns Used:")
    for column in documentation_quality_columns:
            print(f"- {column}")

    print("\nMaintenance Score Columns Used:")
    for column in maintenance_score_columns:
        print(f"- {column}")

    print("\nFinal Integrated Dataset Columns:")
    print(list(integrated_df.columns))


# Save final integration summary as Markdown
def save_final_integration_summary_markdown(
    integrated_df,
    documentation_quality_columns,
    maintenance_score_columns,
    documentation_quality_file_path,
    maintenance_file_paths,
    output_folder_path
):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "final_integration_summary.md"

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# Final Integration Summary\n\n")

        file.write("## Documentation Quality Metrics File\n\n")
        file.write(f"- `{documentation_quality_file_path}`\n\n")

        file.write("## Maintenance Metric Files\n\n")
        for maintenance_file_path in maintenance_file_paths:
            file.write(f"- `{maintenance_file_path}`\n")

        file.write("\n## Output File\n\n")
        file.write("- `final_integrated_dataset.parquet`\n\n")

        file.write("## Dataset Summary\n\n")
        file.write("| Item | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Final Rows | {len(integrated_df)} |\n")
        file.write(f"| Final Columns | {len(integrated_df.columns)} |\n\n")

        file.write("## Documentation Quality Metric Columns Used\n\n")
        for column in documentation_quality_columns:
            file.write(f"- `{column}`\n")

        file.write("## Software Maintenance Score Calculation\n\n")
        file.write("The final software maintenance score was calculated as the average of the selected maintenance score columns.\n\n")

        file.write("### Maintenance Score Columns Used\n\n")
        for column in maintenance_score_columns:
            file.write(f"- `{column}`\n")

        file.write("## Final Dataset Columns\n\n")
        for column in integrated_df.columns:
            file.write(f"- `{column}`\n")

    print(f"Final Integration Summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to create final integrated dataset
def create_final_integrated_dataset(
    documentation_quality_file_path,
    maintenance_file_paths,
    output_folder_path
):
    documentation_quality_df, documentation_quality_file_path = read_documentation_quality_metrics_file(
        documentation_quality_file_path
    )

    maintenance_df = merge_maintenance_metric_files(
        maintenance_file_paths
    )

    maintenance_df = calculate_maintenance_score_columns(
        maintenance_df
    )

    maintenance_df, maintenance_score_columns = calculate_software_maintenance_efficiency_score(
        maintenance_df
    )

    final_maintenance_df = select_final_maintenance_columns(
        maintenance_df
    )

    final_integrated_df = integrate_maintenance_and_documentation_quality_metrics(
        maintenance_df=final_maintenance_df,
        documentation_quality_df=documentation_quality_df
    )

    output_file_path = save_final_integrated_dataset(
        final_integrated_df,
        output_folder_path
    )

    print_final_integration_summary(
        final_integrated_df,
        documentation_quality_df.columns.tolist(),
        maintenance_score_columns
    )

    summary_file_path = save_final_integration_summary_markdown(
        final_integrated_df,
        documentation_quality_df.columns.tolist(),
        maintenance_score_columns,
        documentation_quality_file_path,
        maintenance_file_paths,
        output_folder_path
    )

    return output_file_path, summary_file_path


if __name__ == "__main__":
    documentation_quality_file_path = input("Enter the documentation quality metrics Parquet file path: ").strip()
    documentation_quality_file_path = documentation_quality_file_path.strip('"').strip("'")

    print("\nEnter the maintenance metric Parquet file paths.")
    print("\nPress Enter without typing a file path when you are finished.\n")

    maintenance_file_paths = []

    while True:
        maintenance_file_path = input(
            f"Enter maintenance metric Parquet file path {len(maintenance_file_paths) + 1}: "
        ).strip()

        maintenance_file_path = maintenance_file_path.strip('"').strip("'")

        if maintenance_file_path == "":
            break

        maintenance_file_paths.append(maintenance_file_path)

    if not maintenance_file_paths:
        raise ValueError("At least one maintenance metric Parquet file path is required.")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    create_final_integrated_dataset(
        documentation_quality_file_path=documentation_quality_file_path,
        maintenance_file_paths=maintenance_file_paths,
        output_folder_path=output_folder_path
    )