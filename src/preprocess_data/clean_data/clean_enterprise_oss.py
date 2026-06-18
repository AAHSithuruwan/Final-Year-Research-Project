import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse


MANDATORY_COLUMNS = [
    "repo_url",
    "project_id",
    "project_name",
]


OPTIONAL_COLUMNS = [
    "owner_login",
    "company_name",
    "owner_company",
    "dominant_domain",
    "star_number",
    "license"
]


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


# Standardize column names and rename url to repo_url
def standardize_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if "url" in df.columns and "repo_url" not in df.columns:
        df = df.rename(columns={"url": "repo_url"})

    return df


# Replace empty strings and common null values with NaN
def replace_empty_values_with_nan(df):
    df = df.copy()

    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.replace(["nan", "None", "NULL", "null", "NaN"], np.nan)

    return df


# Check whether all mandatory columns exist in the dataset
def validate_mandatory_columns_exist(df):
    missing_mandatory_columns = [
        column for column in MANDATORY_COLUMNS if column not in df.columns
    ]

    if missing_mandatory_columns:
        raise ValueError(f"Missing mandatory columns in dataset: {missing_mandatory_columns}")

    return True


# Remove rows with missing mandatory column values
def remove_rows_with_missing_mandatory_values(df):
    df = df.copy()

    validate_mandatory_columns_exist(df)

    df = df.dropna(subset=MANDATORY_COLUMNS)

    return df


# Keep only the required columns (Mandatory and Optional columns)
def keep_required_columns(df):
    df = df.copy()

    validate_mandatory_columns_exist(df)

    existing_optional_columns = [
        column for column in OPTIONAL_COLUMNS if column in df.columns
    ]

    selected_columns = MANDATORY_COLUMNS + existing_optional_columns

    df = df[selected_columns]

    return df


# Strip spaces from text columns
def strip_text_columns(df):
    df = df.copy()

    text_columns = [
        "repo_url",
        "project_name",
        "owner_login",
        "company_name",
        "owner_company",
        "dominant_domain",
        "license"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df


# Check whether the repository URL is a valid GitHub repository URL
def is_valid_github_repo_url(repo_url):
    if pd.isna(repo_url):
        return False

    parsed_url = urlparse(str(repo_url).strip())

    if parsed_url.netloc.lower() != "github.com":
        return False

    path_parts = [
        part for part in parsed_url.path.strip("/").split("/")
        if part
    ]

    if len(path_parts) < 2:
        return False

    return True


# Remove rows with invalid GitHub repository URLs
def remove_rows_with_invalid_github_repo_urls(df):
    df = df.copy()

    df = df[df["repo_url"].apply(is_valid_github_repo_url)]

    return df


# Extract owner and repo from GitHub repository URL
def extract_owner_repo_from_url(repo_url):
    parsed_url = urlparse(str(repo_url).strip())

    path_parts = [
        part for part in parsed_url.path.strip("/").split("/")
        if part
    ]

    owner = path_parts[0].lower()
    repo = path_parts[1].lower()

    return owner, repo


# Add repository identity columns (owner, repo, repo_id)
def add_repository_identity_columns(df):
    df = df.copy()

    owners = []
    repos = []

    for repo_url in df["repo_url"]:
        owner, repo = extract_owner_repo_from_url(repo_url)
        owners.append(owner)
        repos.append(repo)

    df["owner"] = owners
    df["repo"] = repos
    df["repo_id"] = df["owner"] + "__" + df["repo"]

    return df


# Normalize project-related fields
def normalize_project_columns(df):
    df = df.copy()

    df["owner"] = df["owner"].str.lower()
    df["repo"] = df["repo"].str.lower()
    df["repo_id"] = df["repo_id"].str.lower()

    if "owner_login" in df.columns:
        df["owner_login"] = df["owner_login"].str.lower()

    if "project_name" in df.columns:
        df["project_name"] = df["project_name"].str.lower()

    if "dominant_domain" in df.columns:
        df["dominant_domain"] = df["dominant_domain"].str.lower()

    return df


# Convert numeric columns
def convert_numeric_columns(df):
    df = df.copy()

    numeric_columns = [
        "project_id",
        "star_number"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# Remove duplicate repositories based on repo_id
def remove_duplicate_repositories(df):
    df = df.copy()

    if "star_number" in df.columns:
        df = df.sort_values(
            by=["star_number"],
            ascending=[False]
        )

    df = df.drop_duplicates(
        subset=["repo_id"],
        keep="first"
    )

    return df


# Reorder final columns
def reorder_columns(df):
    df = df.copy()

    column_order = [
        "owner",
        "repo",
        "repo_id",
        "repo_url",
        "project_id",
        "project_name",
        "owner_login",
        "company_name",
        "owner_company",
        "dominant_domain",
        "star_number",
        "license"
    ]

    existing_columns = [column for column in column_order if column in df.columns]

    df = df[existing_columns]

    return df


# Save cleaned dataset as Parquet
def save_cleaned_dataset(df, input_parquet_file_path, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "enterprise_oss_cleaned.parquet"

    df.to_parquet(output_file_path, index=False, engine="pyarrow")

    print("\nCleaned dataset saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    return output_file_path


# Print cleaning summary in terminal
def print_cleaning_summary(original_df, cleaned_df):
    print("\nCleaning summary")
    print(f"Original rows: {len(original_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Removed rows: {len(original_df) - len(cleaned_df)}")
    print(f"Original columns: {len(original_df.columns)}")
    print(f"Cleaned columns: {len(cleaned_df.columns)}")

    print("\nMandatory columns:")
    print(MANDATORY_COLUMNS)

    print("\nOptional columns:")
    print(OPTIONAL_COLUMNS)

    print("\nCleaned columns:")
    print(list(cleaned_df.columns))

    print("\nTop 10 owners by enterprise repository count:")

    if len(cleaned_df) > 0:
        print(cleaned_df["owner"].value_counts().head(10))
    else:
        print("No rows available after cleaning.")


# Save cleaning summary as a Markdown file
def save_cleaning_summary_markdown(original_df, cleaned_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "enterprise_oss_cleaning_summary.md"

    original_rows = len(original_df)
    cleaned_rows = len(cleaned_df)
    removed_rows = original_rows - cleaned_rows

    original_columns = len(original_df.columns)
    cleaned_columns = len(cleaned_df.columns)

    removed_percentage = 0

    if original_rows > 0:
        removed_percentage = round((removed_rows / original_rows) * 100, 2)

    top_owners = cleaned_df["owner"].value_counts().head(10)

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# Enterprise OSS Dataset Cleaning Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write("- **Cleaned file:** `enterprise_oss_cleaned.parquet`\n\n")

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

        file.write("## Mandatory Columns\n\n")
        file.write("Rows were removed if one of these mandatory values was missing:\n\n")

        for column in MANDATORY_COLUMNS:
            file.write(f"- `{column}`\n")

        file.write("\n## Optional Columns\n\n")

        for column in OPTIONAL_COLUMNS:
            file.write(f"- `{column}`\n")

        file.write("\n## Original Columns\n\n")
        for column in original_df.columns:
            file.write(f"- `{column}`\n")

        file.write("\n## Cleaned Columns\n\n")
        for column in cleaned_df.columns:
            file.write(f"- `{column}`\n")

        file.write("\n## Top 10 Owners by Enterprise Repository Count\n\n")

        if len(top_owners) > 0:
            file.write("| Rank | Owner | Repository Count |\n")
            file.write("|---:|---|---:|\n")

            for rank, (owner, count) in enumerate(top_owners.items(), start=1):
                file.write(f"| {rank} | `{owner}` | {count} |\n")
        else:
            file.write("No rows available after cleaning.\n")

        file.write("\n## Cleaning Rules Applied\n\n")
        file.write("The following cleaning operations were applied:\n\n")
        file.write("1. Standardized column names and renamed `url` to `repo_url`.\n")
        file.write("2. Replaced empty values with `NaN`.\n")
        file.write("3. Removed rows with missing mandatory column values.\n")
        file.write("4. Kept only required columns.\n")
        file.write("5. Trimmed extra spaces from text columns.\n")
        file.write("6. Removed rows with invalid GitHub repository URLs.\n")
        file.write("7. Created `owner`, `repo`, and `repo_id` from the GitHub repository URL.\n")
        file.write("8. Normalized repository identifiers to lowercase.\n")
        file.write("9. Converted `project_id` and `star_number` into numeric values.\n")
        file.write("10. Removed duplicate repositories based on `repo_id`.\n")

    print(f"Cleaning summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common cleaning method
def clean_enterprise_oss_dataset(input_parquet_file_path, output_folder_path):
    original_df, input_parquet_file_path = read_parquet_file(input_parquet_file_path)

    cleaned_df = standardize_column_names(original_df)
    cleaned_df = replace_empty_values_with_nan(cleaned_df)
    cleaned_df = remove_rows_with_missing_mandatory_values(cleaned_df)
    cleaned_df = keep_required_columns(cleaned_df)
    cleaned_df = strip_text_columns(cleaned_df)
    cleaned_df = remove_rows_with_invalid_github_repo_urls(cleaned_df)
    cleaned_df = add_repository_identity_columns(cleaned_df)
    cleaned_df = normalize_project_columns(cleaned_df)
    cleaned_df = convert_numeric_columns(cleaned_df)
    cleaned_df = remove_duplicate_repositories(cleaned_df)
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
    input_file_path = input("Enter the input Enterprise OSS Parquet file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    clean_enterprise_oss_dataset(
        input_parquet_file_path=input_file_path,
        output_folder_path=output_folder_path
    )