import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "owner",
    "repo",
    "repo_id",
    "repo_url",
    "doc_count",
    "total_content_length",
    "doc_paths",
    "documentation"
]

# Read a Parquet file
def read_parquet_file(parquet_file_path, dataset_name):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"{dataset_name} Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError(f"{dataset_name} input file must be a .parquet file")

    print(f"Reading {dataset_name}: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print(f"\n{dataset_name} loaded")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(list(df.columns))

    return df, parquet_file_path


# Validate that both datasets contain repo_id
def validate_repo_id_column(gooddocs_v0_keyword_based_selected_df, gooddocs_v0_enterprise_oss_based_selected_df):
    if "repo_id" not in gooddocs_v0_keyword_based_selected_df.columns:
        raise ValueError("Gooddocs_v0_keyword_based_selected dataset does not contain repo_id column.")

    if "repo_id" not in gooddocs_v0_enterprise_oss_based_selected_df.columns:
        raise ValueError("Gooddocs_v0_enterprise_oss_based_selected dataset does not contain repo_id column.")

    return True


# Normalize repo_id column
def normalize_repo_id_column(df):
    df = df.copy()

    df["repo_id"] = (
        df["repo_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


#Remove unnecessary columns and keep only required columns
def remove_unnecessary_columns(df, required_columns):
    df = df.copy()

    columns_to_drop = [col for col in df.columns if col not in required_columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


# Combine the two datasets
def combine_selected_datasets(gooddocs_v0_keyword_based_selected_df, gooddocs_v0_enterprise_oss_based_selected_df):
    gooddocs_v0_keyword_based_selected_df = gooddocs_v0_keyword_based_selected_df.copy()
    gooddocs_v0_keyword_based_selected_df = remove_unnecessary_columns(gooddocs_v0_keyword_based_selected_df, REQUIRED_COLUMNS)
    gooddocs_v0_enterprise_oss_based_selected_df = gooddocs_v0_enterprise_oss_based_selected_df.copy()
    gooddocs_v0_enterprise_oss_based_selected_df = remove_unnecessary_columns(gooddocs_v0_enterprise_oss_based_selected_df, REQUIRED_COLUMNS)

    gooddocs_v0_combined_df = pd.concat(
        [gooddocs_v0_keyword_based_selected_df, gooddocs_v0_enterprise_oss_based_selected_df],
        ignore_index=True
    )

    return gooddocs_v0_combined_df


# Remove duplicate projects using repo_id
def remove_duplicate_projects(gooddocs_v0_combined_df):
    gooddocs_v0_combined_df = gooddocs_v0_combined_df.copy()

    before_duplicate_removal_count = len(gooddocs_v0_combined_df)

    gooddocs_v0_combined_df = gooddocs_v0_combined_df.drop_duplicates(
        subset=["repo_id"],
        keep="first"
    )

    duplicate_rows_removed = before_duplicate_removal_count - len(gooddocs_v0_combined_df)

    return gooddocs_v0_combined_df, duplicate_rows_removed


# Save the combined gooddocs_v0 dataset as Parquet file
def save_combined_gooddocs_v0_dataset(gooddocs_v0_combined_df, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "gooddocs_v0_combined_selected.parquet"

    gooddocs_v0_combined_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nCombined Gooddocs_v0 dataset saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(gooddocs_v0_combined_df)}")
    print(f"Columns: {len(gooddocs_v0_combined_df.columns)}")

    return output_file_path


# Print combination summary
def print_combination_summary(
    gooddocs_v0_keyword_based_selected_df,
    gooddocs_v0_enterprise_oss_based_selected_df,
    combined_gooddocs_v0_before_duplicate_removal_df,
    combined_gooddocs_v0_after_duplicate_removal_df,
    duplicate_rows_removed
):
    print("\nGooddocs_v0 Selected Datasets Combination Summary")
    print(f"Gooddocs_v0_keyword_based_selected dataset rows: {len(gooddocs_v0_keyword_based_selected_df)}")
    print(f"Gooddocs_v0_enterprise_oss_based_selected dataset rows: {len(gooddocs_v0_enterprise_oss_based_selected_df)}")
    print(f"Combined rows before duplicate removal: {len(combined_gooddocs_v0_before_duplicate_removal_df)}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Final combined rows: {len(combined_gooddocs_v0_after_duplicate_removal_df)}")
    print(f"Final columns: {len(combined_gooddocs_v0_after_duplicate_removal_df.columns)}")

    print("\nFinal columns:")
    print(list(combined_gooddocs_v0_after_duplicate_removal_df.columns))

    print("\nTop 10 selected repositories:")

    if len(combined_gooddocs_v0_after_duplicate_removal_df) > 0:
        for repo_id in combined_gooddocs_v0_after_duplicate_removal_df["repo_id"].head(10):
            print(repo_id)
    else:
        print("No rows available after combining.")


# Save combination summary as Markdown
def save_combination_summary_markdown(
    gooddocs_v0_keyword_based_selected_df,
    gooddocs_v0_enterprise_oss_based_selected_df,
    combined_gooddocs_v0_before_duplicate_removal_df,
    combined_gooddocs_v0_after_duplicate_removal_df,
    duplicate_rows_removed,
    gooddocs_v0_keyword_based_parquet_file_path,
    gooddocs_v0_enterprise_oss_based_parquet_file_path,
    output_folder_path
):
    gooddocs_v0_keyword_based_selected_file_path = Path(gooddocs_v0_keyword_based_parquet_file_path)
    gooddocs_v0_enterprise_oss_based_selected_file_path = Path(gooddocs_v0_enterprise_oss_based_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "gooddocs_v0_combination_summary.md"

    gooddocs_v0_keyword_based_selected_rows = len(gooddocs_v0_keyword_based_selected_df)
    gooddocs_v0_enterprise_oss_based_selected_rows = len(gooddocs_v0_enterprise_oss_based_selected_df)
    combined_gooddocs_v0_before_duplicate_removal_rows = len(combined_gooddocs_v0_before_duplicate_removal_df)
    combined_gooddocs_v0_after_duplicate_removal_rows = len(combined_gooddocs_v0_after_duplicate_removal_df)

    duplicate_percentage = 0

    if combined_gooddocs_v0_before_duplicate_removal_rows > 0:
        duplicate_percentage = round((duplicate_rows_removed / combined_gooddocs_v0_before_duplicate_removal_rows) * 100, 2)

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# Gooddocs_v0 Dataset Combination Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Gooddocs_v0_keyword_based_selected file:** `{gooddocs_v0_keyword_based_selected_file_path}`\n")
        file.write(f"- **Gooddocs_v0_enterprise_oss_based_selected file:** `{gooddocs_v0_enterprise_oss_based_selected_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write(f"- **Combined gooddocs_v0 file:** `{output_folder_path / 'gooddocs_v0_combined_selected.parquet'}`\n\n")

        file.write("## Combination Method\n\n")
        file.write("- Selected Gooddocs_v0 datasets were combined row-wise.\n")
        file.write("- Duplicate projects were removed using the `repo_id` column.\n")

        file.write("## Row Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Gooddocs_v0_keyword_based_selected dataset rows | {gooddocs_v0_keyword_based_selected_rows} |\n")
        file.write(f"| Gooddocs_v0_enterprise_oss_based_selected dataset rows | {gooddocs_v0_enterprise_oss_based_selected_rows} |\n")
        file.write(f"| Combined rows before duplicate removal | {combined_gooddocs_v0_before_duplicate_removal_rows} |\n")
        file.write(f"| Duplicate rows removed | {duplicate_rows_removed} |\n")
        file.write(f"| Duplicate percentage | {duplicate_percentage}% |\n")
        file.write(f"| Combined rows after duplicate removal | {combined_gooddocs_v0_after_duplicate_removal_rows} |\n\n")

        file.write("## Final Columns\n\n")

        for column in combined_gooddocs_v0_after_duplicate_removal_df.columns:
            file.write(f"- `{column}`\n")

        file.write("\n## Top 10 Selected Repositories\n\n")

        if len(combined_gooddocs_v0_after_duplicate_removal_df) > 0:
            file.write("| Rank | Repository ID |\n")
            file.write("|---:|---|\n")

            for rank, repo_id in enumerate(combined_gooddocs_v0_after_duplicate_removal_df["repo_id"].head(10), start=1):
                file.write(f"| {rank} | `{repo_id}` |\n")
        else:
            file.write("No rows available after combining.\n")

        file.write("\n## Rules Applied\n\n")
        file.write("The following operations were applied:\n\n")
        file.write("1. Loaded the Gooddocs_v0_keyword_based_selected dataset.\n")
        file.write("2. Loaded the Gooddocs_v0_enterprise_oss_based_selected dataset.\n")
        file.write("3. Validated that both datasets contain `repo_id`.\n")
        file.write("4. Normalized `repo_id` values by trimming spaces and converting to lowercase.\n")
        file.write("5. Combined both datasets row-wise.\n")
        file.write("6. Removed duplicate repositories using `repo_id`.\n")
        file.write("7. Saved the final combined dataset as a Parquet file.\n")

    print(f"Combination summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to combine two selected Gooddocs_v0 datasets
def combine_gooddocs_v0_selected_datasets(
    gooddocs_v0_keyword_based_selected_parquet_file_path,
    gooddocs_v0_enterprise_oss_based_selected_parquet_file_path,
    output_folder_path
):
    gooddocs_v0_keyword_based_selected_df, gooddocs_v0_keyword_based_selected_parquet_file_path = read_parquet_file(
        gooddocs_v0_keyword_based_selected_parquet_file_path,
        "Gooddocs_v0_keyword_based_selected dataset"
    )

    gooddocs_v0_enterprise_oss_based_selected_df, gooddocs_v0_enterprise_oss_based_selected_parquet_file_path = read_parquet_file(
        gooddocs_v0_enterprise_oss_based_selected_parquet_file_path,
        "Gooddocs_v0_enterprise_oss_based_selected dataset"
    )

    validate_repo_id_column(gooddocs_v0_keyword_based_selected_df, gooddocs_v0_enterprise_oss_based_selected_df)

    gooddocs_v0_keyword_based_selected_df = normalize_repo_id_column(gooddocs_v0_keyword_based_selected_df)
    gooddocs_v0_enterprise_oss_based_selected_df = normalize_repo_id_column(gooddocs_v0_enterprise_oss_based_selected_df)

    combined_gooddocs_v0_before_duplicate_removal_df = combine_selected_datasets(
        gooddocs_v0_keyword_based_selected_df,
        gooddocs_v0_enterprise_oss_based_selected_df
    )

    combined_gooddocs_v0_after_duplicate_removal_df, duplicate_rows_removed = remove_duplicate_projects(
        combined_gooddocs_v0_before_duplicate_removal_df
    )

    print_combination_summary(
        gooddocs_v0_keyword_based_selected_df,
        gooddocs_v0_enterprise_oss_based_selected_df,
        combined_gooddocs_v0_before_duplicate_removal_df,
        combined_gooddocs_v0_after_duplicate_removal_df,
        duplicate_rows_removed
    )

    output_file_path = save_combined_gooddocs_v0_dataset(
        combined_gooddocs_v0_after_duplicate_removal_df,
        output_folder_path
    )

    summary_file_path = save_combination_summary_markdown(
        gooddocs_v0_keyword_based_selected_df,
        gooddocs_v0_enterprise_oss_based_selected_df,
        combined_gooddocs_v0_before_duplicate_removal_df,
        combined_gooddocs_v0_after_duplicate_removal_df,
        duplicate_rows_removed,
        gooddocs_v0_keyword_based_selected_parquet_file_path,
        gooddocs_v0_enterprise_oss_based_selected_parquet_file_path,
        output_folder_path
    )

    return output_file_path, summary_file_path


if __name__ == "__main__":
    first_file_path = input("Enter the Gooddocs_v0_keyword_based_selected Parquet file path: ").strip()
    first_file_path = first_file_path.strip('"').strip("'")

    second_file_path = input("Enter the Gooddocs_v0_enterprise_oss_based_selected Parquet file path: ").strip()
    second_file_path = second_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    combine_gooddocs_v0_selected_datasets(
        gooddocs_v0_keyword_based_selected_parquet_file_path=first_file_path,
        gooddocs_v0_enterprise_oss_based_selected_parquet_file_path=second_file_path,
        output_folder_path=output_folder_path
    )