import pandas as pd
from pathlib import Path


# Read the cleaned GoodDocs Parquet file
def read_cleaned_parquet_file(parquet_file_path):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Cleaned Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    print(f"Reading cleaned Parquet file: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print("\nCleaned dataset")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(list(df.columns))

    return df, parquet_file_path


# Validate required columns for summarization
def validate_required_columns(df):
    required_columns = [
        "owner",
        "repo",
        "repo_id",
        "path",
        "content",
        "content_length"
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


# Create the GitHub repository URL
def add_repo_url(df):
    df = df.copy()

    df["repo_url"] = (
        "https://github.com/"
        + df["owner"].astype(str)
        + "/"
        + df["repo"].astype(str)
    )

    return df


# Combine all documentation file paths of a project
def combine_doc_paths(paths):
    unique_paths = (
        pd.Series(paths)
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return " | ".join(unique_paths)


# Combine all documentation content of a project
def combine_documentation(group):
    documentation_parts = []

    for _, row in group.iterrows():
        path = str(row["path"])
        content = str(row["content"])

        documentation_part = (
            f"# File: {path}\n\n"
            f"{content}"
        )

        documentation_parts.append(documentation_part)

    return "\n\n---\n\n".join(documentation_parts)


# Create one summarized row per project
def create_summarized_dataset(df):
    df = df.copy()

    print("\nSummarizing data by project...")

    summarized_rows = []

    grouped_data = df.groupby(["owner", "repo", "repo_id"], sort=False)

    total_projects = grouped_data.ngroups
    print(f"Total projects to summarize: {total_projects}")

    for index, ((owner, repo, repo_id), group) in enumerate(grouped_data, start=1):
        repo_url = group["repo_url"].iloc[0]
        doc_count = len(group)
        total_content_length = group["content_length"].sum()
        doc_paths = combine_doc_paths(group["path"])
        documentation = combine_documentation(group)

        summarized_rows.append({
            "owner": owner,
            "repo": repo,
            "repo_id": repo_id,
            "repo_url": repo_url,
            "doc_count": doc_count,
            "total_content_length": total_content_length,
            "doc_paths": doc_paths,
            "documentation": documentation
        })

        if index % 500 == 0:
            print(f"Summarized {index} / {total_projects} projects")

    summarized_df = pd.DataFrame(summarized_rows)

    summarized_df = summarized_df.sort_values(
        by=["doc_count", "total_content_length"],
        ascending=[False, False]
    )

    print("\nData summarization completed.")
    print(f"Summarized projects: {len(summarized_df)}")
    print(f"Columns: {len(summarized_df.columns)}")

    return summarized_df


# Save summarized dataset as Parquet
def save_summarized_dataset(summarized_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / f"gooddocs_v0_summarized.parquet"

    summarized_df.to_parquet(output_file_path, index=False, engine="pyarrow")

    print("\nSummarized dataset saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Final rows: {len(summarized_df)}")
    print(f"Final columns: {len(summarized_df.columns)}")

    return output_file_path


# Print summarization summary in terminal
def print_summarization_summary(original_df, summarized_df):
    print("\nSummarization summary")
    print(f"Original documentation rows: {len(original_df)}")
    print(f"Summarized project rows: {len(summarized_df)}")
    print(f"Original columns: {len(original_df.columns)}")
    print(f"Summarized columns: {len(summarized_df.columns)}")

    print("\nSummarized columns:")
    print(list(summarized_df.columns))

    print("\nTop 10 projects by documentation count:")

    if len(summarized_df) > 0:
        print(
            summarized_df[
                ["repo_id", "doc_count", "total_content_length"]
            ].head(10)
        )
    else:
        print("No rows available after summarization.")


# Save summarization summary as a Markdown file
def save_summarization_summary_markdown(original_df, summarized_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / f"gooddocs_v0_summarization_summary.md"

    original_rows = len(original_df)
    summarized_rows = len(summarized_df)

    original_columns = len(original_df.columns)
    summarized_columns = len(summarized_df.columns)

    total_docs = summarized_df["doc_count"].sum() if len(summarized_df) > 0 else 0
    average_docs_per_project = round(summarized_df["doc_count"].mean(), 2) if len(summarized_df) > 0 else 0
    average_content_length_per_project = round(summarized_df["total_content_length"].mean(), 2) if len(summarized_df) > 0 else 0

    top_projects = summarized_df.head(10)

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# GoodDocs Data Summarization Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write(f"- **Summarized file:** `gooddocs_v0_summarized.parquet`\n\n")

        file.write("## Row Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Original documentation rows | {original_rows} |\n")
        file.write(f"| Summarized project rows | {summarized_rows} |\n")
        file.write(f"| Total documentation files represented | {total_docs} |\n")
        file.write(f"| Average docs per project | {average_docs_per_project} |\n")
        file.write(f"| Average content length per project | {average_content_length_per_project} |\n\n")

        file.write("## Column Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Original columns | {original_columns} |\n")
        file.write(f"| Summarized columns | {summarized_columns} |\n\n")

        file.write("## Summarized Columns\n\n")
        for column in summarized_df.columns:
            file.write(f"- `{column}`\n")

        file.write("\n## Top 10 Projects by Documentation Count\n\n")

        if len(top_projects) > 0:
            file.write("| Rank | Repository ID | Docs | Total Content Length |\n")
            file.write("|---:|---|---:|---:|\n")

            for rank, (_, row) in enumerate(top_projects.iterrows(), start=1):
                file.write(
                    f"| {rank} | `{row['repo_id']}` | {row['doc_count']} | {row['total_content_length']} |\n"
                )
        else:
            file.write("No rows available after summarization.\n")

        file.write("\n## Summarization Rules Applied\n\n")
        file.write("The following summarization operations were applied:\n\n")
        file.write("1. Read the cleaned GoodDocs Parquet dataset.\n")
        file.write("2. Validated required columns: `owner`, `repo`, `repo_id`, `path`, `content`, and `content_length`.\n")
        file.write("3. Created the main GitHub repository URL using `https://github.com/{owner}/{repo}`.\n")
        file.write("4. Grouped documentation rows by `owner`, `repo`, and `repo_id`.\n")
        file.write("5. Counted documentation files for each project as `doc_count`.\n")
        file.write("6. Summed documentation content length as `total_content_length`.\n")
        file.write("7. Combined all documentation file paths into `doc_paths`.\n")
        file.write("8. Combined all documentation content into one `documentation` field per project.\n")
        file.write("9. Saved the summarized project-level dataset as a Parquet file.\n")

    print(f"Summarization summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to summarize GoodDocs data
def summarize_gooddocs_dataset(input_parquet_file_path, output_folder_path):
    original_df, input_parquet_file_path = read_cleaned_parquet_file(input_parquet_file_path)

    validate_required_columns(original_df)

    summarized_df = add_repo_url(original_df)
    summarized_df = create_summarized_dataset(summarized_df)

    print_summarization_summary(original_df, summarized_df)

    output_file_path = save_summarized_dataset(
        summarized_df,
        input_parquet_file_path,
        output_folder_path
    )

    summary_file_path = save_summarization_summary_markdown(
        original_df,
        summarized_df,
        input_parquet_file_path,
        output_folder_path
    )

    return output_file_path, summary_file_path


if __name__ == "__main__":
    input_file_path = input("Enter the cleaned GoodDocs Parquet file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    summarize_gooddocs_dataset(
        input_parquet_file_path=input_file_path,
        output_folder_path=output_folder_path
    )