import time
import pandas as pd
from pathlib import Path
from src.github_api.github_api_utils import create_github_headers, call_github_api


GITHUB_SEARCH_ISSUES_ENDPOINT = "https://api.github.com/search/issues"

#python -m src.fetch_maintenance_data.fetch_issue_closure_rate


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


# Build the GitHub issue search query
def build_issue_search_query(owner, repo, issue_state=None):
    query_parts = [
        f"repo:{owner}/{repo}",
        "is:issue"
    ]

    if issue_state is not None:
        query_parts.append(f"is:{issue_state}")

    return " ".join(query_parts)


# Get issue count from the GitHub Search Issues API
def get_github_issue_count(owner, repo, issue_state, headers):
    query = build_issue_search_query(
        owner=owner,
        repo=repo,
        issue_state=issue_state
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


# Calculate the issue closure rate
def calculate_issue_closure_rate(total_issues, closed_issues):
    if total_issues is None or closed_issues is None:
        return None

    if total_issues == 0:
        return None

    return closed_issues / total_issues


# Get GitHub issue closure rate for one repository
def get_github_issue_closure_rate_for_repository(owner, repo, headers):
    total_issues = get_github_issue_count(
        owner=owner,
        repo=repo,
        issue_state=None,
        headers=headers
    )

    closed_issues = get_github_issue_count(
        owner=owner,
        repo=repo,
        issue_state="closed",
        headers=headers
    )

    issue_closure_rate = calculate_issue_closure_rate(
        total_issues=total_issues,
        closed_issues=closed_issues
    )

    return {
        "github_total_issues": total_issues,
        "github_closed_issues": closed_issues,
        "github_issue_closure_rate": issue_closure_rate
    }


# Get GitHub issue closure rates for every repository in the dataset
def get_github_issue_closure_rates_for_dataset(df, request_delay_seconds=1):
    df = df.copy()

    headers = create_github_headers()

    issue_closure_rate_rows = []

    print("\nFetching GitHub Issue Closure Rates...")

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        repo_id = row["repo_id"]
        owner = row["owner"]
        repo = row["repo"]

        print(f"\nProcessing {row_number}/{len(df)}: {owner}/{repo}")

        issue_closure_rate = get_github_issue_closure_rate_for_repository(
            owner=owner,
            repo=repo,
            headers=headers
        )

        issue_closure_rate_rows.append({
            "repo_id": repo_id,
            "github_total_issues": issue_closure_rate["github_total_issues"],
            "github_closed_issues": issue_closure_rate["github_closed_issues"],
            "github_issue_closure_rate": issue_closure_rate["github_issue_closure_rate"]
        })

        print(f"Total issues: {issue_closure_rate['github_total_issues']}")
        print(f"Closed issues: {issue_closure_rate['github_closed_issues']}")
        print(f"Issue closure rate: {issue_closure_rate['github_issue_closure_rate']}")

        time.sleep(request_delay_seconds)

    issue_closure_rate_df = pd.DataFrame(issue_closure_rate_rows)

    return issue_closure_rate_df


# Save Github issue closure rate dataset as a Parquet file
def save_github_issue_closure_rate_dataset(issue_closure_rate_df, output_folder_path):
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / "github_issue_closure_rate.parquet"

    issue_closure_rate_df.to_parquet(
        output_file_path,
        index=False,
        engine="pyarrow"
    )

    print("\nGitHub issue closure rate dataset saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(issue_closure_rate_df)}")
    print(f"Columns: {len(issue_closure_rate_df.columns)}")

    return output_file_path


# Print the GitHub issue closure rate fetching summary in the terminal
def print_fetching_summary(original_df, issue_closure_rate_df):
    success_count = issue_closure_rate_df["github_total_issues"].notna().sum()
    failed_count = issue_closure_rate_df["github_total_issues"].isna().sum()

    print("\nGitHub Issue Closure Rate Fetching Summary")
    print(f"Input dataset rows: {len(original_df)}")
    print(f"Issue closure rate dataset rows: {len(issue_closure_rate_df)}")
    print(f"Repositories with successful issue closure rate fetching: {success_count}")
    print(f"Repositories with failed issue closure rate fetching: {failed_count}")

    print("\nOutput columns:")
    print(list(issue_closure_rate_df.columns))


# Save the GitHub issue closure rate fetching summary as a Markdown file.
def save_fetching_summary_markdown(original_df, issue_closure_rate_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / "github_issue_closure_rate_fetching_summary.md"

    success_count = issue_closure_rate_df["github_total_issues"].notna().sum()
    failed_count = issue_closure_rate_df["github_total_issues"].isna().sum()

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# GitHub Issue Closure Rate Fetching Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write("- **Output file:** `github_issue_closure_rate.parquet`\n\n")

        file.write("## Metrics Collected\n\n")
        file.write("- `github_total_issues`\n")
        file.write("- `github_closed_issues`\n")
        file.write("- `github_issue_closure_rate`\n\n")

        file.write("## Collection Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Input dataset rows | {len(original_df)} |\n")
        file.write(f"| Issue closure rate dataset rows | {len(issue_closure_rate_df)} |\n")
        file.write(f"| Repositories with successful issue closure rate fetching | {success_count} |\n")
        file.write(f"| Repositories with failed issue closure rate fetching | {failed_count} |\n\n")

        file.write("## Fetching Method\n\n")
        file.write("1. Loaded the selected Gooddocs_v0 Parquet dataset.\n")
        file.write("2. Used `owner`, and `repo` columns for GitHub API calls.\n")
        file.write("3. Called GitHub Search Issues API for total issues using `repo:owner/repo is:issue`.\n")
        file.write("4. Called GitHub Search Issues API for closed issues using `repo:owner/repo is:issue is:closed`.\n")
        file.write("5. Calculated issue closure rate as `closed issues / total issues`.\n")
        file.write("6. Saved a separate Parquet file containing `repo_id` and issue closure rate metrics.\n")

        file.write("## Final Columns\n\n")

        for column in issue_closure_rate_df.columns:
            file.write(f"- `{column}`\n")

    print(f"Fetching summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to collect GitHub issue closure rates
def get_github_issue_closure_rates(input_parquet_file_path, output_folder_path, request_delay_seconds=1):
    original_df, input_parquet_file_path = read_parquet_file(input_parquet_file_path)

    validate_required_columns(original_df)

    normalized_df = normalize_required_columns(original_df)

    issue_closure_rate_df = get_github_issue_closure_rates_for_dataset(
        normalized_df,
        request_delay_seconds=request_delay_seconds
    )

    output_file_path = save_github_issue_closure_rate_dataset(
        issue_closure_rate_df,
        output_folder_path
    )

    print_fetching_summary(
        original_df,
        issue_closure_rate_df
    )

    summary_file_path = save_fetching_summary_markdown(
        original_df,
        issue_closure_rate_df,
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

    get_github_issue_closure_rates(
        input_parquet_file_path=input_file_path,
        output_folder_path=output_folder_path,
        request_delay_seconds=request_delay_seconds
    )