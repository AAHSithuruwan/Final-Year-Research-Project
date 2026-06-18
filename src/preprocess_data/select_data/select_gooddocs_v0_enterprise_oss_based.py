import pandas as pd
from pathlib import Path


# Read Parquet file
def read_parquet_file(parquet_file_path, dataset_name):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"{dataset_name} Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError(f"{dataset_name} input file must be a .parquet file")

    print(f"\nReading {dataset_name} dataset: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print(f"{dataset_name} dataset loaded")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(list(df.columns))

    return df, parquet_file_path


# Validate that both datasets contain repo_id
def validate_repo_id_column(gooddocs_v0_df, enterprise_oss_df):
    if "repo_id" not in gooddocs_v0_df.columns:
        raise ValueError("Gooddocs_v0 dataset does not contain repo_id column.")

    if "repo_id" not in enterprise_oss_df.columns:
        raise ValueError("Enterprise_oss dataset does not contain repo_id column.")

    return True


# Normalize repo_id values before matching
def normalize_repo_id_column(df):
    df = df.copy()

    df["repo_id"] = (
        df["repo_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# Select Gooddocs_v0 rows where repo_id exists in Enterprise_oss dataset
def select_matching_rows(gooddocs_v0_df, enterprise_oss_df):
    gooddocs_v0_df = gooddocs_v0_df.copy()
    enterprise_oss_df = enterprise_oss_df.copy()

    enterprise_repo_ids = set(enterprise_oss_df["repo_id"])

    selected_gooddocs_v0_df = gooddocs_v0_df[
        gooddocs_v0_df["repo_id"].isin(enterprise_repo_ids)
    ].copy()

    return selected_gooddocs_v0_df


# Save selected Gooddocs_v0 dataset as a Parquet file
def save_selected_gooddocs_v0_dataset(selected_gooddocs_v0_df, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "gooddocs_v0_enterprise_oss_based_selected.parquet"

    selected_gooddocs_v0_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nSelected Gooddocs_v0 dataset saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(selected_gooddocs_v0_df)}")
    print(f"Columns: {len(selected_gooddocs_v0_df.columns)}")

    return output_file_path


# Print selection summary in the terminal
def print_selection_summary(gooddocs_v0_df, enterprise_oss_df, selected_gooddocs_v0_df):
    original_gooddocs_v0_rows = len(gooddocs_v0_df)
    enterprise_oss_rows = len(enterprise_oss_df)
    selected_rows = len(selected_gooddocs_v0_df)
    removed_rows = original_gooddocs_v0_rows - selected_rows

    selected_percentage = 0

    if original_gooddocs_v0_rows > 0:
        selected_percentage = round((selected_rows / original_gooddocs_v0_rows) * 100, 2)

    print("\nGooddocs_v0 Enterprise_oss Based Selection Summary")
    print(f"Original Gooddocs_v0 rows: {original_gooddocs_v0_rows}")
    print(f"Enterprise_oss rows: {enterprise_oss_rows}")
    print(f"Selected Gooddocs_v0 rows: {selected_rows}")
    print(f"Removed Gooddocs_v0 rows: {removed_rows}")
    print(f"Selected percentage: {selected_percentage}%")

    print("\nFinal selected Gooddocs_v0 columns:")
    print(list(selected_gooddocs_v0_df.columns))

    if "repo_id" in selected_gooddocs_v0_df.columns and len(selected_gooddocs_v0_df) > 0:
        print("\nTop 10 selected repositories:")
        print(selected_gooddocs_v0_df["repo_id"].head(10).reset_index(drop=True))
    else:
        print("\nNo matching Gooddocs_v0 repositories found.")


# Save selection summary as a Markdown file
def save_selection_summary_markdown(
    gooddocs_v0_df,
    enterprise_oss_df,
    selected_gooddocs_v0_df,
    gooddocs_v0_parquet_file_path,
    enterprise_oss_parquet_file_path,
    output_folder_path
):
    gooddocs_v0_parquet_file_path = Path(gooddocs_v0_parquet_file_path)
    enterprise_oss_parquet_file_path = Path(enterprise_oss_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "gooddocs_v0_enterprise_oss_based_selection_summary.md"

    original_gooddocs_v0_rows = len(gooddocs_v0_df)
    enterprise_oss_rows = len(enterprise_oss_df)
    selected_rows = len(selected_gooddocs_v0_df)
    removed_rows = original_gooddocs_v0_rows - selected_rows

    selected_percentage = 0

    if original_gooddocs_v0_rows > 0:
        selected_percentage = round((selected_rows / original_gooddocs_v0_rows) * 100, 2)

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# Gooddocs_v0 Enterprise_oss Based Selection Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Gooddocs_v0 input file:** `{gooddocs_v0_parquet_file_path}`\n")
        file.write(f"- **Enterprise_oss input file:** `{enterprise_oss_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write("- **Selected Gooddocs_v0 file:** `gooddocs_v0_enterprise_selected.parquet`\n\n")

        file.write("## Selection Method\n\n")
        file.write("- The Enterprise_oss dataset was used as a selector dataset.\n")
        file.write("- Matching was performed using the `repo_id` column.\n")
        file.write("- Gooddocs_v0 rows were kept only if their `repo_id` exists in the Enterprise_oss dataset.\n")

        file.write("## Row Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Original Gooddocs_v0 rows | {original_gooddocs_v0_rows} |\n")
        file.write(f"| Enterprise_oss rows | {enterprise_oss_rows} |\n")
        file.write(f"| Selected Gooddocs_v0 rows | {selected_rows} |\n")
        file.write(f"| Removed Gooddocs_v0 rows | {removed_rows} |\n")
        file.write(f"| Selected percentage | {selected_percentage}% |\n\n")

        file.write("## Final Selected Gooddocs_v0 Columns\n\n")

        for column in selected_gooddocs_v0_df.columns:
            file.write(f"- `{column}`\n")

        file.write("\n## Selection Rules Applied\n\n")
        file.write("The following operations were applied:\n\n")
        file.write("1. Loaded the Gooddocs_v0 summarized Parquet dataset.\n")
        file.write("2. Loaded the cleaned Enterprise_oss Parquet dataset.\n")
        file.write("3. Validated that both datasets contain `repo_id`.\n")
        file.write("4. Normalized `repo_id` values by trimming spaces and converting to lowercase.\n")
        file.write("5. Selected only Gooddocs_v0 rows where `repo_id` exists in Enterprise_oss.\n")
        file.write("6. Saved the selected Gooddocs_v0 dataset as Parquet.\n")

    print(f"Selection summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to select Gooddocs_v0 rows using Enterprise_oss repo_ids
def select_gooddocs_v0_using_enterprise_oss(
    gooddocs_v0_parquet_file_path,
    enterprise_oss_parquet_file_path,
    output_folder_path
):
    gooddocs_v0_df, gooddocs_v0_parquet_file_path = read_parquet_file(
        gooddocs_v0_parquet_file_path,
        "Gooddocs_v0"
    )

    enterprise_oss_df, enterprise_oss_parquet_file_path = read_parquet_file(
        enterprise_oss_parquet_file_path,
        "Enterprise_oss"
    )

    validate_repo_id_column(gooddocs_v0_df, enterprise_oss_df)

    gooddocs_v0_df = normalize_repo_id_column(gooddocs_v0_df)
    enterprise_oss_df = normalize_repo_id_column(enterprise_oss_df)

    selected_gooddocs_v0_df = select_matching_rows(
        gooddocs_v0_df,
        enterprise_oss_df
    )

    print_selection_summary(
        gooddocs_v0_df,
        enterprise_oss_df,
        selected_gooddocs_v0_df
    )

    output_file_path = save_selected_gooddocs_v0_dataset(
        selected_gooddocs_v0_df,
        output_folder_path
    )

    summary_file_path = save_selection_summary_markdown(
        gooddocs_v0_df,
        enterprise_oss_df,
        selected_gooddocs_v0_df,
        gooddocs_v0_parquet_file_path,
        enterprise_oss_parquet_file_path,
        output_folder_path
    )

    return output_file_path, summary_file_path


if __name__ == "__main__":
    gooddocs_v0_file_path = input("Enter the Gooddocs_v0 summarized Parquet file path: ").strip()
    gooddocs_v0_file_path = gooddocs_v0_file_path.strip('"').strip("'")

    enterprise_oss_file_path = input("Enter the Enterprise_oss cleaned Parquet file path: ").strip()
    enterprise_oss_file_path = enterprise_oss_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    select_gooddocs_v0_using_enterprise_oss(
        gooddocs_v0_parquet_file_path=gooddocs_v0_file_path,
        enterprise_oss_parquet_file_path=enterprise_oss_file_path,
        output_folder_path=output_folder_path
    )