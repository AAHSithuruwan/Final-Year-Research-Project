import time
import pandas as pd
from pathlib import Path
from src.github_api.github_api_utils import create_github_headers, call_github_paginated_api


# python -m src.fetch_maintenance_data.fetch_avg_issue_comment_count


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


# Create the GitHub issues endpoint
def create_github_issues_endpoint(owner, repo):
    return f"https://api.github.com/repos/{owner}/{repo}/issues"


# Fetch all issues of a repository
def get_issues_for_repository(owner, repo, headers, request_delay_seconds=1):
    url = create_github_issues_endpoint(owner, repo)

    params = {
        "state": "all",
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

    issues = []

    for issue in data:
        # GitHub issues endpoint also returns pull requests
        # Pull requests contain the "pull_request" key, so they are skipped
        if "pull_request" in issue:
            continue

        issues.append(issue)

    print(f"Fetched issues for {owner}/{repo}. Valid Issues Fetched: {len(issues)}")

    return issues


# Get average issue comment count for one repository
def get_github_avg_issue_comment_count_for_repository(owner, repo, headers, request_delay_seconds=1):
    issues = get_issues_for_repository(
        owner=owner,
        repo=repo,
        headers=headers,
        request_delay_seconds=request_delay_seconds
    )

    if issues is None:
        return {
            "github_total_issues": None,
            "github_total_issue_comments": None,
            "github_avg_issue_comment_count": None
        }

    github_total_issues = len(issues)
    github_total_issue_comments = 0

    for issue in issues:
        comment_count = issue.get("comments")

        if comment_count is not None:
            github_total_issue_comments += comment_count

    if github_total_issues is None or github_total_issue_comments is None:
        github_avg_issue_comment_count = None

    elif github_total_issues == 0 or github_total_issue_comments == 0:
        github_avg_issue_comment_count = None

    else:
        github_avg_issue_comment_count = github_total_issue_comments / github_total_issues

    return {
        "github_total_issues": github_total_issues,
        "github_total_issue_comments": github_total_issue_comments,
        "github_avg_issue_comment_count": github_avg_issue_comment_count
    }


# Get average issue comment counts for every repository in the dataset
def get_github_avg_issue_comment_counts_for_dataset(df, request_delay_seconds=1):
    df = df.copy()

    headers = create_github_headers()

    avg_issue_comment_count_rows = []

    print("\nFetching GitHub Average Issue Comment Count...")

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        repo_id = row["repo_id"]
        owner = row["owner"]
        repo = row["repo"]

        print(f"\nProcessing {row_number}/{len(df)}: {owner}/{repo}")

        avg_issue_comment_count = get_github_avg_issue_comment_count_for_repository(
            owner=owner,
            repo=repo,
            headers=headers,
            request_delay_seconds=request_delay_seconds
        )

        avg_issue_comment_count_rows.append({
            "repo_id": repo_id,
            "github_total_issues": avg_issue_comment_count["github_total_issues"],
            "github_total_issue_comments": avg_issue_comment_count["github_total_issue_comments"],
            "github_avg_issue_comment_count": avg_issue_comment_count["github_avg_issue_comment_count"]
        })

        print(f"Total Issues: {avg_issue_comment_count['github_total_issues']}")
        print(f"Total Issue Comments: {avg_issue_comment_count['github_total_issue_comments']}")
        print(f"Average Issue Comment Count: {avg_issue_comment_count['github_avg_issue_comment_count']}")

        time.sleep(request_delay_seconds)

    avg_issue_comment_count_df = pd.DataFrame(avg_issue_comment_count_rows)

    return avg_issue_comment_count_df


# Normalize average issue comment count dataset (between 0 and 1 using Min-Max normalization)
def normalize_avg_issue_comment_count_dataset(avg_issue_comment_count_df):
    avg_issue_comment_count_df = avg_issue_comment_count_df.copy()

    metric_column = "github_avg_issue_comment_count"
    normalized_column = "github_normalized_avg_issue_comment_count"

    valid_values = avg_issue_comment_count_df[metric_column].dropna()

    if valid_values.empty:
        avg_issue_comment_count_df[normalized_column] = None
        return avg_issue_comment_count_df

    min_value = valid_values.min()
    max_value = valid_values.max()

    if min_value == max_value:
        avg_issue_comment_count_df[normalized_column] = avg_issue_comment_count_df[metric_column].apply(
            lambda value: 0.0 if pd.notna(value) else None
        )
        return avg_issue_comment_count_df

    avg_issue_comment_count_df[normalized_column] = avg_issue_comment_count_df[metric_column].apply(
        lambda value: (value - min_value) / (max_value - min_value) if pd.notna(value) else None
    )

    print("Normalization completed successfully.")

    return avg_issue_comment_count_df


# Save normalized average issue comment count dataset as a Parquet file
def save_normalized_avg_issue_comment_count_dataset(avg_issue_comment_count_df, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "github_avg_issue_comment_count.parquet"

    avg_issue_comment_count_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nNormalized GitHub Average Issue Comment Count Dataset Saved Successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(avg_issue_comment_count_df)}")
    print(f"Columns: {len(avg_issue_comment_count_df.columns)}")

    return output_file_path


# Print the average issue comment count fetching summary in the terminal
def print_fetching_summary(original_df, avg_issue_comment_count_df):
    success_count = avg_issue_comment_count_df["github_avg_issue_comment_count"].notna().sum()
    failed_count = avg_issue_comment_count_df["github_avg_issue_comment_count"].isna().sum()

    print("\nGitHub Average Issue Comment Count Fetching Summary")
    print(f"Input dataset rows: {len(original_df)}")
    print(f"Average issue comment count dataset rows: {len(avg_issue_comment_count_df)}")
    print(f"Repositories with successful average issue comment count fetching: {success_count}")
    print(f"Repositories with failed average issue comment count fetching: {failed_count}")

    print("\nOutput Columns:")
    print(list(avg_issue_comment_count_df.columns))


# Save the average issue comment count fetching summary as a Markdown file
def save_fetching_summary_markdown(original_df, avg_issue_comment_count_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "github_avg_issue_comment_count_fetching_summary.md"

    success_count = avg_issue_comment_count_df["github_avg_issue_comment_count"].notna().sum()
    failed_count = avg_issue_comment_count_df["github_avg_issue_comment_count"].isna().sum()

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# GitHub Average Issue Comment Count Fetching Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write("- **Output file:** `github_avg_issue_comment_count.parquet`\n\n")

        file.write("## Metrics Collected\n\n")
        file.write("- `github_total_issues`\n")
        file.write("- `github_total_issue_comments`\n")
        file.write("- `github_avg_issue_comment_count`\n")
        file.write("- `github_normalized_avg_issue_comment_count`\n\n")

        file.write("## Collection Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Input dataset rows | {len(original_df)} |\n")
        file.write(f"| Average issue comment count dataset rows | {len(avg_issue_comment_count_df)} |\n")
        file.write(f"| Repositories with successful average issue comment count fetching | {success_count} |\n")
        file.write(f"| Repositories with failed average issue comment count fetching | {failed_count} |\n\n")

        file.write("## Fetching Method\n\n")
        file.write("1. Loaded the selected Gooddocs_v0 Parquet dataset.\n")
        file.write("2. Used `owner` and `repo` columns for GitHub API calls.\n")
        file.write("3. Called GitHub Issues API using `state=all`.\n")
        file.write("4. Used paginated API fetching to collect issues from all available pages.\n")
        file.write("5. Skipped pull requests returned by the GitHub Issues API.\n")
        file.write("6. Read the `comments` field from each valid issue.\n")
        file.write("7. Calculated average issue comment count as `total issue comments / total issues`.\n")
        file.write("8. Applied Min-Max normalization to `github_avg_issue_comment_count`.\n")
        file.write("9. Saved a separate Parquet file containing `repo_id` and issue comment count metrics.\n\n")

        file.write("## Final Columns\n\n")

        for column in avg_issue_comment_count_df.columns:
            file.write(f"- `{column}`\n")

    print(f"Fetching Summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to fetch GitHub average issue comment counts
def get_github_avg_issue_comment_counts(input_parquet_file_path, output_folder_path, request_delay_seconds=1):
    original_df, input_parquet_file_path = read_parquet_file(input_parquet_file_path)

    validate_required_columns(original_df)

    normalized_df = normalize_required_columns(original_df)

    avg_issue_comment_count_df = get_github_avg_issue_comment_counts_for_dataset(
        normalized_df,
        request_delay_seconds=request_delay_seconds
    )

    avg_issue_comment_count_df = normalize_avg_issue_comment_count_dataset(
        avg_issue_comment_count_df
    )
    
    output_file_path = save_normalized_avg_issue_comment_count_dataset(
        avg_issue_comment_count_df,
        output_folder_path
    )

    print_fetching_summary(
        original_df,
        avg_issue_comment_count_df
    )

    summary_file_path = save_fetching_summary_markdown(
        original_df,
        avg_issue_comment_count_df,
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

    get_github_avg_issue_comment_counts(
        input_parquet_file_path=input_file_path,
        output_folder_path=output_folder_path,
        request_delay_seconds=request_delay_seconds
    )