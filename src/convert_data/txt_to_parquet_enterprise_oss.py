import pandas as pd
from pathlib import Path


# Define the column names based on the structure of the Enterprise OSS raw text file.
COLUMN_NAMES = [
    "url",
    "project_id",
    "sdtc",
    "mcpc",
    "mcve",
    "star_number",
    "commit_count",
    "files",
    "lines",
    "pull_requests",
    "github_repo_creation",
    "earliest_commit",
    "most_recent_commit",
    "committer_count",
    "author_count",
    "dominant_domain",
    "dominant_domain_committer_commits",
    "dominant_domain_author_commits",
    "dominant_domain_committers",
    "dominant_domain_authors",
    "cik",
    "fg500",
    "sec10k",
    "sec20f",
    "project_name",
    "owner_login",
    "company_name",
    "owner_company",
    "license"
]


# Function to validate the input text file path and output folder path.
def validate_file_paths(input_txt_file_path, output_folder_path):
    input_txt_file_path = Path(input_txt_file_path)
    output_folder_path = Path(output_folder_path)

    if not input_txt_file_path.exists():
        raise FileNotFoundError(f"Input text file not found: {input_txt_file_path}")

    if input_txt_file_path.suffix.lower() != ".txt":
        raise ValueError("Input file must be a .txt file")

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_parquet_file_path = output_folder_path / f"{input_txt_file_path.stem}.parquet"

    return input_txt_file_path, output_folder_path, output_parquet_file_path


# Function to read the Enterprise OSS tab-separated text file.
def read_enterprise_oss_txt(input_txt_file_path):
    print(f"Reading tab-separated file: {input_txt_file_path}")

    df = pd.read_csv(
        input_txt_file_path,
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
        encoding="utf-8"
    )

    print("\nFile loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# Function to save the converted dataset as a Parquet file.
def save_enterprise_oss_parquet(df, output_parquet_file_path):
    df.to_parquet(
        output_parquet_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nParquet file created successfully.")
    print(f"Output file: {output_parquet_file_path}")

    return output_parquet_file_path


# Function to print the conversion summary in the terminal.
def print_conversion_summary(df, input_txt_file_path, output_parquet_file_path):
    print("\nEnterprise OSS TXT to Parquet Conversion Summary")
    print(f"Input file: {input_txt_file_path}")
    print(f"Output file: {output_parquet_file_path}")
    print(f"Rows converted: {len(df)}")
    print(f"Columns converted: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")


# Common method to convert the Enterprise OSS text dataset into Parquet.
def convert_txt_to_parquet_enterprise_oss(input_txt_file_path, output_folder_path):
    input_txt_file_path, output_folder_path, output_parquet_file_path = validate_file_paths(
        input_txt_file_path,
        output_folder_path
    )

    df = read_enterprise_oss_txt(input_txt_file_path)

    save_enterprise_oss_parquet(df, output_parquet_file_path)

    print_conversion_summary(
        df,
        input_txt_file_path,
        output_parquet_file_path
    )

    return output_parquet_file_path


if __name__ == "__main__":
    input_file_path = input("Enter the Enterprise OSS input .txt file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    convert_txt_to_parquet_enterprise_oss(
        input_txt_file_path=input_file_path,
        output_folder_path=output_folder_path
    )