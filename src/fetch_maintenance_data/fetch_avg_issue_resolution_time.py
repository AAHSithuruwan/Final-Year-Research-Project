import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.github_api.github_api_utils import call_github_paginated_api, create_github_headers


# python -m src.fetch_maintenance_data.fetch_avg_issue_resolution_time


# Read the selected Gooddocs_v0 Parquet file
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


# Validate that the dataset contains the required repository columns
def validate_required_columns(df):
    required_columns = [
        "repo_id",
        "owner",
        "repo"
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


# Normalize the required columns
def normalize_required_columns(df):
    df = df.copy()

    df["repo_id"] = df["repo_id"].astype(str).str.strip().str.lower()
    df["owner"] = df["owner"].astype(str).str.strip().str.lower()
    df["repo"] = df["repo"].astype(str).str.strip().str.lower()

    return df


# Convert GitHub datetime string into Python datetime
def parse_github_datetime(datetime_value):
    if datetime_value is None:
        return None

    return datetime.strptime(
        datetime_value,
        "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)


# Calculate the Issue Resolution Time in days
def calculate_issue_resolution_time_days(created_at, closed_at):
    created_datetime = parse_github_datetime(created_at)
    closed_datetime = parse_github_datetime(closed_at)

    if created_datetime is None or closed_datetime is None:
        return None

    resolution_time = closed_datetime - created_datetime

    return resolution_time.total_seconds() / 86400


# Create the GitHub closed issues endpoint
def create_github_closed_issues_endpoint(owner, repo):
    return f"https://api.github.com/repos/{owner}/{repo}/issues"


# Fetch all closed issues of a repository
def get_closed_issues_for_repository(owner, repo, headers, request_delay_seconds=1):
    url = create_github_closed_issues_endpoint(owner, repo)

    closed_issues = []

    params = {
        "state": "closed",
        "per_page": 100
    }

    data = call_github_paginated_api(
        url=url,
        headers=headers,
        params=params,
        request_delay_seconds=request_delay_seconds
    )

    if data is None:
        return None

    for issue in data:
        if "pull_request" in issue:
            continue

        closed_issues.append(issue)

    print(f"Fetched closed issues for {owner}/{repo}. Valid Closed Issues: {len(closed_issues)}")

    return closed_issues


# Get Average Issue Resolution Time for one repository
def get_avg_issue_resolution_time_for_repository(owner, repo, headers, request_delay_seconds=1):
    closed_issues = get_closed_issues_for_repository(
        owner=owner,
        repo=repo,
        headers=headers,
        request_delay_seconds=request_delay_seconds
    )

    if closed_issues is None:
        return {
            "github_total_closed_issues": None,
            "github_avg_resolution_time_days": None
        }

    resolution_time_values = []

    for issue in closed_issues:
        created_at = issue.get("created_at")
        closed_at = issue.get("closed_at")

        resolution_time_days = calculate_issue_resolution_time_days(
            created_at=created_at,
            closed_at=closed_at
        )

        if resolution_time_days is not None:
            resolution_time_values.append(resolution_time_days)

    github_total_closed_issues = len(resolution_time_values)

    if github_total_closed_issues == 0:
        github_avg_resolution_time_days = None
    else:
        github_avg_resolution_time_days = sum(resolution_time_values) / github_total_closed_issues

    return {
        "github_total_closed_issues": github_total_closed_issues,
        "github_avg_resolution_time_days": github_avg_resolution_time_days
    }


# Get average issue resolution time for every repository in the dataset
def get_avg_issue_resolution_times_for_dataset(df, request_delay_seconds=1):
    df = df.copy()

    headers = create_github_headers()

    issue_resolution_time_rows = []

    print("\nFetching GitHub Average Issue Resolution Times...")

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        repo_id = row["repo_id"]
        owner = row["owner"]
        repo = row["repo"]

        print(f"\nProcessing {row_number}/{len(df)}: {owner}/{repo}")

        issue_resolution_time = get_avg_issue_resolution_time_for_repository(
            owner=owner,
            repo=repo,
            headers=headers,
            request_delay_seconds=request_delay_seconds
        )

        issue_resolution_time_rows.append({
            "repo_id": repo_id,
            "github_total_closed_issues": issue_resolution_time["github_total_closed_issues"],
            "github_avg_resolution_time_days": issue_resolution_time["github_avg_resolution_time_days"]
        })

        print(f"Total Closed Issues: {issue_resolution_time['github_total_closed_issues']}")
        print(f"Average Issue Resolution Time in Days: {issue_resolution_time['github_avg_resolution_time_days']}")

        time.sleep(request_delay_seconds)

    issue_resolution_time_df = pd.DataFrame(issue_resolution_time_rows)

    return issue_resolution_time_df


# Normalize average issue resolution times dataset (between 0 and 1 using Min-Max normalization)
def normalize_avg_issue_resolution_times_dataset(issue_resolution_time_df):

    print("\nNormalizing GitHub Average Issue Resolution Time Dataset...")

    issue_resolution_time_df = issue_resolution_time_df.copy()

    metric_column = "github_avg_resolution_time_days"
    normalized_column = "github_normalized_avg_resolution_time_days"

    valid_values = issue_resolution_time_df[metric_column].dropna()

    if valid_values.empty:
        issue_resolution_time_df[normalized_column] = None
        return issue_resolution_time_df

    min_value = valid_values.min()
    max_value = valid_values.max()

    if min_value == max_value:
        issue_resolution_time_df[normalized_column] = issue_resolution_time_df[metric_column].apply(
            lambda value: 0.0 if pd.notna(value) else None
        )
        return issue_resolution_time_df

    issue_resolution_time_df[normalized_column] = issue_resolution_time_df[metric_column].apply(
        lambda value: (value - min_value) / (max_value - min_value) if pd.notna(value) else None
    )

    print("Normalization completed successfully.")

    return issue_resolution_time_df


# Save Normalized average issue resolution times dataset as a Parquet file
def save_normalized_avg_issue_resolution_times_dataset(issue_resolution_time_df, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "github_avg_issue_resolution_time.parquet"

    issue_resolution_time_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nNormalized GitHub Average Issue Resolution Time Dataset Saved Successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(issue_resolution_time_df)}")
    print(f"Columns: {len(issue_resolution_time_df.columns)}")

    return output_file_path


# Print the GitHub average issue resolution time fetching summary in the terminal
def print_fetching_summary(original_df, issue_resolution_time_df):
    success_count = issue_resolution_time_df["github_avg_resolution_time_days"].notna().sum()
    failed_count = issue_resolution_time_df["github_avg_resolution_time_days"].isna().sum()

    print("\nGitHub Average Issue Resolution Time Fetching Summary")
    print(f"Input dataset rows: {len(original_df)}")
    print(f"Fetched average issue resolution time dataset rows: {len(issue_resolution_time_df)}")
    print(f"Repositories with successful average issue resolution time fetching: {success_count}")
    print(f"Repositories with failed average issue resolution time fetching: {failed_count}")

    print("\nOutput Columns:")
    print(list(issue_resolution_time_df.columns))


# Save the GitHub issue resolution time fetching summary as a Markdown file
def save_fetching_summary_markdown(original_df, issue_resolution_time_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "github_avg_issue_resolution_time_fetching_summary.md"

    success_count = issue_resolution_time_df["github_avg_resolution_time_days"].notna().sum()
    failed_count = issue_resolution_time_df["github_avg_resolution_time_days"].isna().sum()

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# GitHub Average Issue Resolution Time Fetching Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write("- **Output file:** `github_avg_issue_resolution_time.parquet`\n\n")

        file.write("## Metrics Collected\n\n")
        file.write("- `github_total_closed_issues`\n")
        file.write("- `github_avg_resolution_time_days`\n")
        file.write("- `github_normalized_avg_resolution_time_days`\n\n")

        file.write("## Collection Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Input dataset rows | {len(original_df)} |\n")
        file.write(f"| Fetched average issue resolution time dataset rows | {len(issue_resolution_time_df)} |\n")
        file.write(f"| Repositories with successful average issue resolution time fetching | {success_count} |\n")
        file.write(f"| Repositories with failed average issue resolution time fetching | {failed_count} |\n\n")

        file.write("## Fetching Method\n\n")
        file.write("1. Loaded the selected Gooddocs_v0 Parquet dataset.\n")
        file.write("2. Used `owner` and `repo` columns for GitHub API calls.\n")
        file.write("3. Called GitHub Issues API for closed issues using `state=closed`.\n")
        file.write("4. Skipped pull requests returned by the GitHub Issues API.\n")
        file.write("5. Calculated issue resolution time as `closed_at - created_at`.\n")
        file.write("6. Calculated the average issue resolution time in days for each repository.\n")
        file.write("7. Applied Min-Max normalization to the calculated average issue resolution time.\n")
        file.write("8. Saved a separate Parquet file containing `repo_id` and issue resolution time metrics.\n\n")

        file.write("## Final Columns\n\n")

        for column in issue_resolution_time_df.columns:
            file.write(f"- `{column}`\n")

    print(f"Fetching Summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to collect Normalized average issue resolution times
def get_normalized_avg_issue_resolution_times(input_parquet_file_path, output_folder_path, request_delay_seconds=1):
    original_df, input_parquet_file_path = read_parquet_file(input_parquet_file_path)

    validate_required_columns(original_df)

    normalized_df = normalize_required_columns(original_df)

    issue_resolution_time_df = get_avg_issue_resolution_times_for_dataset(
        normalized_df,
        request_delay_seconds=request_delay_seconds
    )

    normalized_issue_resolution_time_df = normalize_avg_issue_resolution_times_dataset(issue_resolution_time_df)

    output_file_path = save_normalized_avg_issue_resolution_times_dataset(
        normalized_issue_resolution_time_df,
        output_folder_path
    )

    print_fetching_summary(
        original_df,
        normalized_issue_resolution_time_df
    )

    summary_file_path = save_fetching_summary_markdown(
        original_df,
        normalized_issue_resolution_time_df,
        input_parquet_file_path,
        output_folder_path
    )

    return output_file_path, summary_file_path


if __name__ == "__main__":
    input_file_path = input("Enter the selected Gooddocs_v0 Parquet file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    delay_input = input("Enter request delay in seconds, or press Enter to use 1: ").strip()

    if delay_input == "":
        request_delay_seconds = 1
    else:
        request_delay_seconds = float(delay_input)

    get_normalized_avg_issue_resolution_times(
        input_parquet_file_path=input_file_path,
        output_folder_path=output_folder_path,
        request_delay_seconds=request_delay_seconds
    )