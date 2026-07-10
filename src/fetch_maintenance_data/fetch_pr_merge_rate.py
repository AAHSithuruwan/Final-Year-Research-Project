import time
import pandas as pd
from pathlib import Path
from src.github_api.github_api_utils import create_github_headers, call_github_api


GITHUB_SEARCH_ISSUES_ENDPOINT = "https://api.github.com/search/issues"


# python -m src.fetch_maintenance_data.fetch_pr_merge_rate


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


# Create the GitHub pull request search query
def create_pr_search_query(owner, repo, merged_only=False):
    query_parts = [
        f"repo:{owner}/{repo}",
        "is:pr"
    ]

    if merged_only:
        query_parts.append("is:merged")

    return " ".join(query_parts)


# Get pull request count from the GitHub Search Issues API
def get_github_pr_count(owner, repo, headers, merged_only=False):
    query = create_pr_search_query(
        owner=owner,
        repo=repo,
        merged_only=merged_only
    )

    params = {
        "q": query,
        "per_page": 1
    }

    data = call_github_api(
        url=GITHUB_SEARCH_ISSUES_ENDPOINT,
        headers=headers,
        params=params
    )

    if data is None:
        return None

    return data.get("total_count")


# Calculate pull request merge rate
def calculate_pr_merge_rate(total_pull_requests, merged_pull_requests):
    if total_pull_requests is None or merged_pull_requests is None:
        return None

    if total_pull_requests == 0:
        return None

    return merged_pull_requests / total_pull_requests


# Get GitHub pull request merge rate for one repository
def get_github_pr_merge_rate_for_repository(owner, repo, headers):
    total_pull_requests = get_github_pr_count(
        owner=owner,
        repo=repo,
        headers=headers,
        merged_only=False
    )

    merged_pull_requests = get_github_pr_count(
        owner=owner,
        repo=repo,
        headers=headers,
        merged_only=True
    )

    pr_merge_rate = calculate_pr_merge_rate(
        total_pull_requests=total_pull_requests,
        merged_pull_requests=merged_pull_requests
    )

    return {
        "github_total_pull_requests": total_pull_requests,
        "github_merged_pull_requests": merged_pull_requests,
        "github_pr_merge_rate": pr_merge_rate
    }


# Get GitHub pull request merge rates for every repository in the dataset
def get_github_pr_merge_rates_for_dataset(df, request_delay_seconds=1):
    df = df.copy()

    headers = create_github_headers()

    pr_merge_rate_rows = []

    print("\nFetching GitHub Pull Request Merge Rates...")

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        repo_id = row["repo_id"]
        owner = row["owner"]
        repo = row["repo"]

        print(f"\nProcessing {row_number}/{len(df)}: {owner}/{repo}")

        pr_merge_rate = get_github_pr_merge_rate_for_repository(
            owner=owner,
            repo=repo,
            headers=headers
        )

        pr_merge_rate_rows.append({
            "repo_id": repo_id,
            "github_total_pull_requests": pr_merge_rate["github_total_pull_requests"],
            "github_merged_pull_requests": pr_merge_rate["github_merged_pull_requests"],
            "github_pr_merge_rate": pr_merge_rate["github_pr_merge_rate"]
        })

        print(f"Total pull requests: {pr_merge_rate['github_total_pull_requests']}")
        print(f"Merged pull requests: {pr_merge_rate['github_merged_pull_requests']}")
        print(f"Pull request merge rate: {pr_merge_rate['github_pr_merge_rate']}")

        time.sleep(request_delay_seconds)

    pr_merge_rate_df = pd.DataFrame(pr_merge_rate_rows)

    return pr_merge_rate_df


# Save pull request merge rate dataset as a Parquet file
def save_github_pr_merge_rate_dataset(pr_merge_rate_df, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "github_pr_merge_rate.parquet"

    pr_merge_rate_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nGitHub pull request merge rate dataset saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(pr_merge_rate_df)}")
    print(f"Columns: {len(pr_merge_rate_df.columns)}")

    return output_file_path


# Print the GitHub pull request merge rate fetching summary in the terminal
def print_fetching_summary(original_df, pr_merge_rate_df):
    success_count = pr_merge_rate_df["github_pr_merge_rate"].notna().sum()
    failed_count = pr_merge_rate_df["github_pr_merge_rate"].isna().sum()

    print("\nGitHub Pull Request Merge Rate Fetching Summary")
    print(f"Input dataset rows: {len(original_df)}")
    print(f"PR Merge Rate dataset rows: {len(pr_merge_rate_df)}")
    print(f"Repositories with successful PR merge rate fetching: {success_count}")
    print(f"Repositories with failed PR merge rate fetching: {failed_count}")

    print("\nOutput columns:")
    print(list(pr_merge_rate_df.columns))


# Save the GitHub pull request merge rate fetching summary as a Markdown file
def save_fetching_summary_markdown(original_df, pr_merge_rate_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "github_pr_merge_rate_fetching_summary.md"

    success_count = pr_merge_rate_df["github_pr_merge_rate"].notna().sum()
    failed_count = pr_merge_rate_df["github_pr_merge_rate"].isna().sum()

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# GitHub Pull Request Merge Rate Fetching Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write("- **Output file:** `github_pr_merge_rate.parquet`\n\n")

        file.write("## Metrics Collected\n\n")
        file.write("- `github_total_pull_requests`\n")
        file.write("- `github_merged_pull_requests`\n")
        file.write("- `github_pr_merge_rate`\n\n")

        file.write("## Collection Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Input dataset rows | {len(original_df)} |\n")
        file.write(f"| PR Merge Rate dataset rows | {len(pr_merge_rate_df)} |\n")
        file.write(f"| Repositories with successful PR merge rate fetching | {success_count} |\n")
        file.write(f"| Repositories with failed PR merge rate fetching | {failed_count} |\n\n")

        file.write("## Fetching Method\n\n")
        file.write("1. Loaded the selected Gooddocs_v0 Parquet dataset.\n")
        file.write("2. Used `owner` and `repo` columns for GitHub API calls.\n")
        file.write("3. Called GitHub Search Issues API to count total pull requests using `repo:owner/repo is:pr`.\n")
        file.write("4. Called GitHub Search Issues API to count merged pull requests using `repo:owner/repo is:pr is:merged`.\n")
        file.write("5. Calculated pull request merge rate as `merged pull requests / total pull requests`.\n")
        file.write("6. Saved a separate Parquet file containing `repo_id` and pull request merge rate metrics.\n\n")

        file.write("## Final Columns\n\n")

        for column in pr_merge_rate_df.columns:
            file.write(f"- `{column}`\n")

    print(f"Fetching Summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to fetch GitHub pull request merge rates
def get_github_pr_merge_rates(input_parquet_file_path, output_folder_path, request_delay_seconds=1):
    original_df, input_parquet_file_path = read_parquet_file(input_parquet_file_path)

    validate_required_columns(original_df)

    normalized_df = normalize_required_columns(original_df)

    pr_merge_rate_df = get_github_pr_merge_rates_for_dataset(
        normalized_df,
        request_delay_seconds=request_delay_seconds
    )

    output_file_path = save_github_pr_merge_rate_dataset(
        pr_merge_rate_df,
        output_folder_path
    )

    print_fetching_summary(
        original_df,
        pr_merge_rate_df
    )

    summary_file_path = save_fetching_summary_markdown(
        original_df,
        pr_merge_rate_df,
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

    get_github_pr_merge_rates(
        input_parquet_file_path=input_file_path,
        output_folder_path=output_folder_path,
        request_delay_seconds=request_delay_seconds
    )