import pandas as pd
import numpy as np
from pathlib import Path


DOCUMENTATION_EXTENSIONS = (".md", ".mdx", ".rst", ".txt", ".adoc")


# Read the Parquet file
def read_parquet_file(parquet_file_path):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    print(f"Reading Parquet file: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print("\nOriginal dataset")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(list(df.columns))

    return df, parquet_file_path


# Standardize column names
def standardize_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return df


# Rename GoodDocs v0 columns and create repo_id/path
def rename_columns_if_needed(df):
    df = df.copy()

    # GoodDocs v0 uses file_rel_repo as the file path inside the GitHub repository
    if "file_rel_repo" in df.columns:
        df = df.rename(columns={"file_rel_repo": "path"})

    # Create repo_id using owner and repo
    if "repo_id" not in df.columns:
        if "owner" in df.columns and "repo" in df.columns:
            df["repo_id"] = (
                df["owner"].astype(str).str.strip().str.lower()
                + "__"
                + df["repo"].astype(str).str.strip().str.lower()
            )
        else:
            raise ValueError("Cannot create repo_id because owner or repo column is missing.")

    required_columns = {"owner", "repo", "repo_id", "path", "content"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns after renaming: {missing_columns}")

    return df


# Keep only useful columns
def keep_required_columns(df):
    df = df.copy()

    useful_columns = [
        "owner",
        "repo",
        "repo_id",
        "path",
        "size",
        "mtime",
        "lang",
        "content"
    ]

    important_columns = ["owner", "repo", "repo_id", "path", "content"]

    missing_important_columns = [
        column for column in important_columns if column not in df.columns
    ]

    if missing_important_columns:
        raise ValueError(f"Missing required columns: {missing_important_columns}")

    existing_columns = [column for column in useful_columns if column in df.columns]

    df = df[existing_columns]

    return df


# Replace empty strings and common null values with NaN
def replace_empty_values_with_nan(df):
    df = df.copy()

    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.replace(["nan", "None", "NULL", "null", "NaN"], np.nan)

    return df


# Remove rows with missing important values
def remove_missing_required_values(df):
    df = df.copy()

    required_columns = [
        "owner",
        "repo",
        "repo_id",
        "path",
        "content"
    ]

    df = df.dropna(subset=required_columns)

    return df


# Strip spaces from text columns
def strip_text_columns(df):
    df = df.copy()

    text_columns = [
        "owner",
        "repo",
        "repo_id",
        "path",
        "lang",
        "content"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    return df


# Normalize project-related fields
def normalize_project_columns(df):
    df = df.copy()

    df["owner"] = df["owner"].str.lower()
    df["repo"] = df["repo"].str.lower()
    df["repo_id"] = df["repo_id"].str.lower()

    if "lang" in df.columns:
        df["lang"] = df["lang"].str.lower()

    return df


# Keep only documentation-like files
def keep_documentation_files_only(df):
    df = df.copy()

    df = df[
        df["path"]
        .str.lower()
        .str.endswith(DOCUMENTATION_EXTENSIONS)
    ]

    return df


# Add content length
def add_content_length(df):
    df = df.copy()

    df["content_length"] = df["content"].str.len()

    return df


# Remove short documentation content
def remove_short_content(df, minimum_content_length=200):
    df = df.copy()

    df = df[df["content_length"] >= minimum_content_length]

    return df


# Remove exact duplicate documentation rows
def remove_exact_duplicates(df):
    df = df.copy()

    df = df.drop_duplicates(
        subset=["repo_id", "path", "content"],
        keep="first"
    )

    return df


# Add possible GitHub URLs
def add_github_url_columns(df):
    df = df.copy()

    df["github_url_main"] = (
        "https://github.com/"
        + df["owner"]
        + "/"
        + df["repo"]
        + "/blob/main/"
        + df["path"]
    )

    df["github_url_master"] = (
        "https://github.com/"
        + df["owner"]
        + "/"
        + df["repo"]
        + "/blob/master/"
        + df["path"]
    )

    return df


# Reorder final columns
def reorder_columns(df):
    df = df.copy()

    column_order = [
        "owner",
        "repo",
        "repo_id",
        "path",
        "size",
        "mtime",
        "lang",
        "content",
        "content_length",
        "github_url_main",
        "github_url_master"
    ]

    existing_columns = [column for column in column_order if column in df.columns]

    df = df[existing_columns]

    return df


# Save cleaned dataset as Parquet
def save_cleaned_dataset(df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / f"gooddocs_v0_cleaned.parquet"

    df.to_parquet(output_file_path, index=False, engine="pyarrow")

    print("\nCleaned dataset saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    return output_file_path


# Print cleaning summary
def print_cleaning_summary(original_df, cleaned_df):
    print("\nCleaning summary")
    print(f"Original rows: {len(original_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Removed rows: {len(original_df) - len(cleaned_df)}")
    print(f"Original columns: {len(original_df.columns)}")
    print(f"Cleaned columns: {len(cleaned_df.columns)}")

    print("\nCleaned columns:")
    print(list(cleaned_df.columns))

    print("\nTop 10 projects by documentation row count:")

    if len(cleaned_df) > 0:
        print(cleaned_df["repo_id"].value_counts().head(10))
    else:
        print("No rows available after cleaning.")


# Save cleaning summary as a Markdown file
def save_cleaning_summary_markdown(original_df, cleaned_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / f"gooddocs_v0_cleaning_summary.md"

    original_rows = len(original_df)
    cleaned_rows = len(cleaned_df)
    removed_rows = original_rows - cleaned_rows

    original_columns = len(original_df.columns)
    cleaned_columns = len(cleaned_df.columns)

    removed_percentage = 0

    if original_rows > 0:
        removed_percentage = round((removed_rows / original_rows) * 100, 2)

    top_projects = cleaned_df["repo_id"].value_counts().head(10)

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# GoodDocs Dataset Cleaning Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write(f"- **Cleaned file:** `gooddocs_v0_cleaned.parquet`\n\n")

        file.write("## Row Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Original rows | {original_rows} |\n")
        file.write(f"| Cleaned rows | {cleaned_rows} |\n")
        file.write(f"| Removed rows | {removed_rows} |\n")
        file.write(f"| Removed percentage | {removed_percentage}% |\n\n")

        file.write("## Column Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Original columns | {original_columns} |\n")
        file.write(f"| Cleaned columns | {cleaned_columns} |\n\n")

        file.write("## Original Columns\n\n")
        for column in original_df.columns:
            file.write(f"- `{column}`\n")

        file.write("\n## Cleaned Columns\n\n")
        for column in cleaned_df.columns:
            file.write(f"- `{column}`\n")

        file.write("\n## Top 10 Projects by Documentation Row Count\n\n")

        if len(top_projects) > 0:
            file.write("| Rank | Repository ID | Documentation Rows |\n")
            file.write("|---:|---|---:|\n")

            for rank, (repo_id, count) in enumerate(top_projects.items(), start=1):
                file.write(f"| {rank} | `{repo_id}` | {count} |\n")
        else:
            file.write("No rows available after cleaning.\n")

        file.write("\n## Cleaning Rules Applied\n\n")
        file.write("The following cleaning operations were applied:\n\n")
        file.write("1. Standardized column names.\n")
        file.write("2. Renamed `file_rel_repo` to `path`.\n")
        file.write("3. Created `repo_id` using `owner + '__' + repo`.\n")
        file.write("4. Kept only useful documentation-related columns.\n")
        file.write("5. Replaced empty values with `NaN`.\n")
        file.write("6. Removed rows with missing `owner`, `repo`, `repo_id`, `path`, or `content`.\n")
        file.write("7. Trimmed extra spaces from text columns.\n")
        file.write("8. Normalized project identifiers to lowercase.\n")
        file.write("9. Kept only documentation-like files: `.md`, `.mdx`, `.rst`, `.txt`, `.adoc`.\n")
        file.write("10. Removed documentation rows with short content.\n")
        file.write("11. Removed exact duplicate rows based on `repo_id`, `path`, and `content`.\n")
        file.write("12. Added possible GitHub URL columns for `main` and `master` branches.\n")

    print(f"Cleaning summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common cleaning method
def clean_gooddocs_dataset(input_parquet_file_path, output_folder_path, minimum_content_length=200):
    original_df, input_parquet_file_path = read_parquet_file(input_parquet_file_path)

    cleaned_df = standardize_column_names(original_df)
    cleaned_df = rename_columns_if_needed(cleaned_df)
    cleaned_df = keep_required_columns(cleaned_df)
    cleaned_df = replace_empty_values_with_nan(cleaned_df)
    cleaned_df = remove_missing_required_values(cleaned_df)
    cleaned_df = strip_text_columns(cleaned_df)
    cleaned_df = normalize_project_columns(cleaned_df)
    cleaned_df = keep_documentation_files_only(cleaned_df)
    cleaned_df = add_content_length(cleaned_df)
    cleaned_df = remove_short_content(
        cleaned_df,
        minimum_content_length=minimum_content_length
    )
    cleaned_df = remove_exact_duplicates(cleaned_df)
    cleaned_df = add_github_url_columns(cleaned_df)
    cleaned_df = reorder_columns(cleaned_df)

    print_cleaning_summary(original_df, cleaned_df)

    output_file_path = save_cleaned_dataset(
        cleaned_df,
        input_parquet_file_path,
        output_folder_path
    )

    summary_file_path = save_cleaning_summary_markdown(
        original_df,
        cleaned_df,
        input_parquet_file_path,
        output_folder_path
    )

    return output_file_path, summary_file_path


if __name__ == "__main__":
    input_file_path = input("Enter the input GoodDocs Parquet file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    min_length_input = input(
        "Enter minimum content length, or press Enter to use 200: "
    ).strip()

    if min_length_input == "":
        minimum_content_length = 200
    elif min_length_input.isdigit():
        minimum_content_length = int(min_length_input)
    else:
        raise ValueError("Minimum content length must be a positive number.")

    clean_gooddocs_dataset(
        input_parquet_file_path=input_file_path,
        output_folder_path=output_folder_path,
        minimum_content_length=minimum_content_length
    )