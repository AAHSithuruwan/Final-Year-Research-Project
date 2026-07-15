import json
import re
import pandas as pd
from pathlib import Path


# python -m src.extract_doc_quality_metrics.extract_doc_quality_metrics


# Read the selected Parquet file which contains the documentation data
def read_parquet_file(parquet_file_path):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    print(f"Reading Parquet file: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(list(df.columns))

    return df, parquet_file_path


# Read the documentation quality keywords JSON file
def read_keyword_json_file(keyword_json_file_path):
    keyword_json_file_path = Path(keyword_json_file_path)

    if not keyword_json_file_path.exists():
        raise FileNotFoundError(f"Keyword JSON file not found: {keyword_json_file_path}")

    if keyword_json_file_path.suffix.lower() != ".json":
        raise ValueError("Keyword file must be a .json file")

    print(f"\nReading keyword JSON file: {keyword_json_file_path}")

    with open(keyword_json_file_path, "r", encoding="utf-8") as file:
        keyword_data = json.load(file)

    print("Keyword JSON file loaded successfully.")
    print(f"Metric groups found: {len(keyword_data)}")
    print("Metric names:")
    print(list(keyword_data.keys()))

    return keyword_data, keyword_json_file_path


# Validate that the dataset contains the required columns
def validate_required_columns(df):
    required_columns = [
        "repo_id",
        "documentation"
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


# Normalize required columns
def normalize_required_columns(df):
    df = df.copy()

    df["repo_id"] = df["repo_id"].astype(str).str.strip().str.lower()
    df["documentation"] = df["documentation"].fillna("").astype(str)

    return df


# Normalize text before keyword matching
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


# Check whether a keyword exists in documentation text
def check_keyword_exists(documentation_text, keyword):
    keyword = str(keyword).lower().strip()

    if keyword == "":
        return False

    pattern = r"\b" + re.escape(keyword) + r"\b"

    if re.search(pattern, documentation_text):
        return True

    # Fallback for keywords with symbols such as ".env", "ci/cd", "docker-compose"
    if keyword in documentation_text:
        return True

    return False


# Calculate the score for a group of keywords
# The number of matched keywords / total number of keywords
def calculate_keyword_score(documentation_text, keywords):
    if not keywords:
        return 0.0

    matched_keywords = 0

    for keyword in keywords:
        if check_keyword_exists(documentation_text, keyword):
            matched_keywords += 1

    return matched_keywords / len(keywords)


# Extract documentation quality metrics for one repository
def extract_documentation_quality_metrics_for_repository(repo_id, documentation_text, keyword_data):
    documentation_text = normalize_text(documentation_text)

    metric_row = {
        "repo_id": repo_id
    }

    for metric_name, metric_keywords in keyword_data.items():
        metric_score = calculate_keyword_score(
            documentation_text=documentation_text,
            keywords=metric_keywords
        )

        metric_row[metric_name] = metric_score

    return metric_row


# Extract documentation quality metrics for the full dataset
def extract_documentation_quality_metrics_for_dataset(df, keyword_data):
    df = df.copy()

    documentation_quality_rows = []

    print("\nExtracting Documentation Quality Metrics...")

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        repo_id = row["repo_id"]
        documentation_text = row["documentation"]

        print(f"Processing {row_number}/{len(df)}: {repo_id}")

        metric_row = extract_documentation_quality_metrics_for_repository(
            repo_id=repo_id,
            documentation_text=documentation_text,
            keyword_data=keyword_data
        )

        documentation_quality_rows.append(metric_row)

    documentation_quality_df = pd.DataFrame(documentation_quality_rows)

    return documentation_quality_df


# Save documentation quality metrics dataset as a Parquet file
def save_documentation_quality_metrics_dataset(documentation_quality_df, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "doc_quality_metrics.parquet"

    documentation_quality_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nDocumentation Quality Metrics Dataset Saved Successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(documentation_quality_df)}")
    print(f"Columns: {len(documentation_quality_df.columns)}")

    return output_file_path


# Print documentation quality metric extraction summary in the terminal
def print_extraction_summary(original_df, documentation_quality_df):
    print("\nDocumentation Quality Metric Extraction Summary")
    print(f"Input dataset rows: {len(original_df)}")
    print(f"Documentation quality metrics dataset rows: {len(documentation_quality_df)}")

    print("\nOutput Columns:")
    print(list(documentation_quality_df.columns))


# Save the documentation quality metric extraction summary as a Markdown file
def save_extraction_summary_markdown(
    original_df,
    documentation_quality_df,
    input_parquet_file_path,
    keyword_json_file_path,
    output_folder_path
):
    input_parquet_file_path = Path(input_parquet_file_path)
    keyword_json_file_path = Path(keyword_json_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "doc_quality_metrics_extraction_summary.md"

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# Documentation Quality Metrics Extraction Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Keyword JSON file:** `{keyword_json_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write("- **Output file:** `doc_quality_metrics.parquet`\n\n")

        file.write("## Extraction Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Input dataset rows | {len(original_df)} |\n")
        file.write(f"| Documentation quality metrics dataset rows | {len(documentation_quality_df)} |\n")

        file.write("## Extraction Method\n\n")
        file.write("1. Loaded the selected Gooddocs_v0 Parquet dataset.\n")
        file.write("2. Used the `repo_id` and `documentation` columns for metric extraction.\n")
        file.write("3. Loaded predefined documentation quality keywords from the JSON file.\n")
        file.write("4. Normalized documentation text by converting it to lowercase and removing repeated spaces.\n")
        file.write("5. Calculated documentation quality metric scores using keyword matching.\n")
        file.write("6. Saved a separate Parquet file containing `repo_id` and documentation quality metrics.\n\n")

        file.write("## Final Columns\n\n")

        for column in documentation_quality_df.columns:
            file.write(f"- `{column}`\n")

    print(f"Extraction Summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to extract documentation quality metrics
def extract_documentation_quality_metrics(input_parquet_file_path, keyword_json_file_path, output_folder_path):
    original_df, input_parquet_file_path = read_parquet_file(input_parquet_file_path)

    validate_required_columns(original_df)

    normalized_df = normalize_required_columns(original_df)

    keyword_data, keyword_json_file_path = read_keyword_json_file(keyword_json_file_path)

    documentation_quality_df = extract_documentation_quality_metrics_for_dataset(
        normalized_df,
        keyword_data
    )

    output_file_path = save_documentation_quality_metrics_dataset(
        documentation_quality_df,
        output_folder_path
    )

    print_extraction_summary(
        original_df,
        documentation_quality_df
    )

    summary_file_path = save_extraction_summary_markdown(
        original_df,
        documentation_quality_df,
        input_parquet_file_path,
        keyword_json_file_path,
        output_folder_path
    )

    return output_file_path, summary_file_path


if __name__ == "__main__":
    input_file_path = input("Enter the selected Gooddocs_v0 Parquet file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    keyword_json_file_path = input("Enter the documentation quality keywords JSON file path: ").strip()
    keyword_json_file_path = keyword_json_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    extract_documentation_quality_metrics(
        input_parquet_file_path=input_file_path,
        keyword_json_file_path=keyword_json_file_path,
        output_folder_path=output_folder_path
    )