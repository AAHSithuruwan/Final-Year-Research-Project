import json
import pandas as pd
from pathlib import Path


# Read the summarized project-level Parquet dataset
def read_summarized_dataset(input_parquet_file_path):
    input_parquet_file_path = Path(input_parquet_file_path)

    if not input_parquet_file_path.exists():
        raise FileNotFoundError(f"Summarized Parquet file not found: {input_parquet_file_path}")

    if input_parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    print(f"Reading summarized dataset: {input_parquet_file_path}")

    df = pd.read_parquet(input_parquet_file_path)

    print("\nSummarized dataset loaded")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(list(df.columns))

    return df, input_parquet_file_path


# Read enterprise keyword configuration from JSON file
def read_enterprise_config(config_file_path):
    config_file_path = Path(config_file_path)

    if not config_file_path.exists():
        raise FileNotFoundError(f"Enterprise keyword config file not found: {config_file_path}")

    if config_file_path.suffix.lower() != ".json":
        raise ValueError("Config file must be a .json file")

    print(f"Reading enterprise keyword config: {config_file_path}")

    with open(config_file_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    return config, config_file_path


# Validate that the summarized dataset has the required columns
def validate_required_dataset_columns(df):
    required_columns = [
        "owner",
        "repo",
        "repo_id",
        "repo_url",
        "doc_count",
        "total_content_length",
        "doc_paths",
        "documentation"
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")

    return True


# Validate that the keyword configuration has the required sections
def validate_required_config_sections(config):
    required_sections = [
        "enterprise_organizations",
        "repo_keywords",
        "documentation_keywords",
        "negative_keywords",
        "scoring_weights",
        "minimum_threshold"
    ]

    missing_sections = [
        section for section in required_sections if section not in config
    ]

    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")

    return True


# Normalize text values before keyword matching
def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).lower().strip()


# Find keywords that appear inside a text value
def find_matched_keywords(text, keywords):
    text = normalize_text(text)

    matched_keywords = []

    for keyword in keywords:
        keyword_normalized = normalize_text(keyword)

        if keyword_normalized != "" and keyword_normalized in text:
            matched_keywords.append(keyword)

    return sorted(set(matched_keywords))


# Find categorized documentation keyword matches
def find_documentation_keyword_matches_by_category(text, documentation_keywords):
    category_matches = {}

    for category_name, keywords in documentation_keywords.items():
        matched_keywords = find_matched_keywords(text, keywords)

        if matched_keywords:
            category_matches[category_name] = matched_keywords

    return category_matches


# Convert categorized keyword matches into readable text
def format_category_matches(category_matches):
    if not category_matches:
        return ""

    formatted_parts = []

    for category_name, keywords in category_matches.items():
        formatted_parts.append(
            f"{category_name}: {', '.join(keywords)}"
        )

    return " | ".join(formatted_parts)


# Calculate organization score based on known enterprise organizations
def calculate_organization_score(owner, enterprise_organizations, organization_weight):
    owner = normalize_text(owner)

    enterprise_organizations = [
        normalize_text(organization) for organization in enterprise_organizations
    ]

    if owner in enterprise_organizations:
        return organization_weight, owner

    return 0, ""


# Calculate repository keyword score using repo and repo_id
def calculate_repo_keyword_score(repo, repo_id, repo_keywords, repo_keyword_weight):
    searchable_text = (
        f"{normalize_text(repo)} "
        f"{normalize_text(repo_id)}"
    )

    matched_keywords = find_matched_keywords(
        searchable_text,
        repo_keywords
    )

    score = len(matched_keywords) * repo_keyword_weight

    return score, matched_keywords


# Calculate documentation keyword score using doc_paths and documentation
def calculate_documentation_keyword_score(doc_paths, documentation, documentation_keywords, documentation_keyword_weight):
    searchable_text = (
        f"{normalize_text(doc_paths)} "
        f"{normalize_text(documentation)}"
    )

    category_matches = find_documentation_keyword_matches_by_category(
        searchable_text,
        documentation_keywords
    )

    all_matched_keywords = []

    for keywords in category_matches.values():
        all_matched_keywords.extend(keywords)

    all_matched_keywords = sorted(set(all_matched_keywords))

    score = len(all_matched_keywords) * documentation_keyword_weight

    return score, all_matched_keywords, category_matches


# Calculate negative keyword penalty using repo, doc paths, and documentation
def calculate_negative_keyword_penalty(repo, repo_id, doc_paths, documentation, negative_keywords, negative_keyword_penalty):
    searchable_text = (
        f"{normalize_text(repo)} "
        f"{normalize_text(repo_id)} "
        f"{normalize_text(doc_paths)} "
        f"{normalize_text(documentation)}"
    )

    matched_negative_keywords = find_matched_keywords(
        searchable_text,
        negative_keywords
    )

    penalty = len(matched_negative_keywords) * negative_keyword_penalty

    return penalty, matched_negative_keywords


# Add enterprise selection scores to every project
def score_enterprise_projects(df, config):
    df = df.copy()

    enterprise_organizations = config["enterprise_organizations"]
    repo_keywords = config["repo_keywords"]
    documentation_keywords = config["documentation_keywords"]
    negative_keywords = config["negative_keywords"]

    scoring_weights = config["scoring_weights"]

    organization_weight = scoring_weights["enterprise_organization_match"]
    repo_keyword_weight = scoring_weights["repo_keyword_match"]
    documentation_keyword_weight = scoring_weights["documentation_keyword_match"]
    negative_keyword_penalty = scoring_weights["negative_keyword_penalty"]

    organization_scores = []
    matched_organizations = []

    repo_keyword_scores = []
    matched_repo_keywords_list = []

    documentation_keyword_scores = []
    matched_documentation_keywords_list = []
    matched_documentation_categories_list = []

    negative_keyword_scores = []
    matched_negative_keywords_list = []

    print("\nScoring enterprise relevance...")

    for index, row in df.iterrows():
        organization_score, matched_organization = calculate_organization_score(
            row["owner"],
            enterprise_organizations,
            organization_weight
        )

        repo_keyword_score, matched_repo_keywords = calculate_repo_keyword_score(
            row["repo"],
            row["repo_id"],
            repo_keywords,
            repo_keyword_weight
        )

        documentation_keyword_score, matched_documentation_keywords, category_matches = calculate_documentation_keyword_score(
            row["doc_paths"],
            row["documentation"],
            documentation_keywords,
            documentation_keyword_weight
        )

        negative_keyword_score, matched_negative_keywords = calculate_negative_keyword_penalty(
            row["repo"],
            row["repo_id"],
            row["doc_paths"],
            row["documentation"],
            negative_keywords,
            negative_keyword_penalty
        )

        organization_scores.append(organization_score)
        matched_organizations.append(matched_organization)

        repo_keyword_scores.append(repo_keyword_score)
        matched_repo_keywords_list.append(", ".join(matched_repo_keywords))

        documentation_keyword_scores.append(documentation_keyword_score)
        matched_documentation_keywords_list.append(", ".join(matched_documentation_keywords))
        matched_documentation_categories_list.append(format_category_matches(category_matches))

        negative_keyword_scores.append(negative_keyword_score)
        matched_negative_keywords_list.append(", ".join(matched_negative_keywords))

        if index % 500 == 0 and index != 0:
            print(f"Scored {index} / {len(df)} projects")

    df["organization_score"] = organization_scores
    df["repo_keyword_score"] = repo_keyword_scores
    df["documentation_keyword_score"] = documentation_keyword_scores
    df["negative_keyword_score"] = negative_keyword_scores

    df["total_enterprise_score"] = (
        df["organization_score"]
        + df["repo_keyword_score"]
        + df["documentation_keyword_score"]
        + df["negative_keyword_score"]
    )

    df["matched_organization"] = matched_organizations
    df["matched_repo_keywords"] = matched_repo_keywords_list
    df["matched_documentation_keywords"] = matched_documentation_keywords_list
    df["matched_documentation_categories"] = matched_documentation_categories_list
    df["matched_negative_keywords"] = matched_negative_keywords_list

    return df


# Create a short reason explaining why each project was selected or rejected
def create_selection_reason(row):
    reasons = []

    if row["organization_score"] > 0:
        reasons.append("known enterprise/open-source organization")

    if row["repo_keyword_score"] > 0:
        reasons.append("enterprise keyword found in repository name")

    if row["documentation_keyword_score"] > 0:
        reasons.append("enterprise keyword found in documentation")

    if row["negative_keyword_score"] < 0:
        reasons.append("negative keyword penalty applied")

    if row["doc_count"] < row["minimum_doc_count"]:
        reasons.append("documentation count below threshold")

    if row["total_content_length"] < row["minimum_content_length"]:
        reasons.append("content length below threshold")

    if row["total_enterprise_score"] < row["minimum_enterprise_score"]:
        reasons.append("enterprise score below threshold")

    return "; ".join(reasons)


# Apply minimum thresholds to select enterprise-level projects
def apply_enterprise_selection_thresholds(df, config):
    df = df.copy()

    minimum_threshold = config["minimum_threshold"]

    minimum_enterprise_score = minimum_threshold["minimum_enterprise_score"]
    minimum_doc_count = minimum_threshold["minimum_doc_count"]
    minimum_content_length = minimum_threshold["minimum_content_length"]

    df["minimum_enterprise_score"] = minimum_enterprise_score
    df["minimum_doc_count"] = minimum_doc_count
    df["minimum_content_length"] = minimum_content_length

    df["is_enterprise_project"] = (
        (df["total_enterprise_score"] >= minimum_enterprise_score)
        & (df["doc_count"] >= minimum_doc_count)
        & (df["total_content_length"] >= minimum_content_length)
    )

    df["selection_reason"] = df.apply(create_selection_reason, axis=1)

    selected_df = df[df["is_enterprise_project"] == True].copy()

    selected_df = selected_df.sort_values(
        by=["total_enterprise_score", "doc_count", "total_content_length"],
        ascending=[False, False, False]
    )

    df = df.sort_values(
        by=["total_enterprise_score", "doc_count", "total_content_length"],
        ascending=[False, False, False]
    )

    print("\nEnterprise project selection completed.")
    print(f"Original projects: {len(df)}")
    print(f"Selected enterprise projects: {len(selected_df)}")
    print(f"Removed projects: {len(df) - len(selected_df)}")

    return selected_df, df


# Reorder selected/scored output columns
def reorder_output_columns(df):
    df = df.copy()

    column_order = [
        "owner",
        "repo",
        "repo_id",
        "repo_url",
        "doc_count",
        "total_content_length",
        "doc_paths",
        "documentation",
        "organization_score",
        "repo_keyword_score",
        "documentation_keyword_score",
        "negative_keyword_score",
        "total_enterprise_score",
        "matched_organization",
        "matched_repo_keywords",
        "matched_documentation_keywords",
        "matched_documentation_categories",
        "matched_negative_keywords",
        "selection_reason",
        "is_enterprise_project",
        "minimum_enterprise_score",
        "minimum_doc_count",
        "minimum_content_length"
    ]

    existing_columns = [
        column for column in column_order if column in df.columns
    ]

    return df[existing_columns]


# Save selected enterprise projects as a Parquet file
def save_selected_enterprise_projects(selected_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / f"gooddocs_v0_selected.parquet"

    selected_df.to_parquet(output_file_path, index=False, engine="pyarrow")

    print("\nSelected enterprise projects saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(selected_df)}")
    print(f"Columns: {len(selected_df.columns)}")

    return output_file_path


# Save all scored projects as a Parquet file for review
def save_all_scored_projects(scored_df, input_parquet_file_path, output_folder_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_folder_path / f"gooddocs_v0_scored.parquet"

    scored_df.to_parquet(output_file_path, index=False, engine="pyarrow")

    print("\nAll scored projects saved successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(scored_df)}")
    print(f"Columns: {len(scored_df.columns)}")

    return output_file_path


# Print a summary of enterprise selection in the terminal
def print_selection_summary(original_df, selected_df, config):
    minimum_threshold = config["minimum_threshold"]

    print("\nEnterprise Selection Summary")
    print(f"Original projects: {len(original_df)}")
    print(f"Selected enterprise projects: {len(selected_df)}")
    print(f"Removed projects: {len(original_df) - len(selected_df)}")
    print(f"Minimum enterprise score: {minimum_threshold['minimum_enterprise_score']}")
    print(f"Minimum doc count: {minimum_threshold['minimum_doc_count']}")
    print(f"Minimum content length: {minimum_threshold['minimum_content_length']}")

    print("\nTop 10 selected projects:")

    if len(selected_df) > 0:
        print(
            selected_df[
                [
                    "repo_id",
                    "doc_count",
                    "total_content_length",
                    "total_enterprise_score"
                ]
            ].head(10)
        )
    else:
        print("No projects selected.")


# Save enterprise selection summary as a Markdown file
def save_selection_summary_markdown(original_df, selected_df, scored_df, input_parquet_file_path, output_folder_path, config, config_file_path):
    input_parquet_file_path = Path(input_parquet_file_path)
    output_folder_path = Path(output_folder_path)
    config_file_path = Path(config_file_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)

    summary_file_path = output_folder_path / f"gooddocs_v0_selection_summary.md"

    minimum_threshold = config["minimum_threshold"]
    scoring_weights = config["scoring_weights"]

    original_count = len(original_df)
    selected_count = len(selected_df)
    removed_count = original_count - selected_count

    selected_percentage = 0

    if original_count > 0:
        selected_percentage = round((selected_count / original_count) * 100, 2)

    top_projects = selected_df.head(10)

    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write("# GoodDocs Enterprise Project Selection Summary\n\n")

        file.write("## Input and Output Information\n\n")
        file.write(f"- **Input file:** `{input_parquet_file_path}`\n")
        file.write(f"- **Keyword config file:** `{config_file_path}`\n")
        file.write(f"- **Output folder:** `{output_folder_path}`\n")
        file.write(f"- **Selected projects file:** `{input_parquet_file_path.stem}_enterprise_projects.parquet`\n")
        file.write(f"- **All scored projects file:** `{input_parquet_file_path.stem}_enterprise_scored_all.parquet`\n\n")

        file.write("## Selection Thresholds\n\n")
        file.write("| Threshold | Value |\n")
        file.write("|---|---:|\n")
        file.write(f"| Minimum enterprise score | {minimum_threshold['minimum_enterprise_score']} |\n")
        file.write(f"| Minimum documentation count | {minimum_threshold['minimum_doc_count']} |\n")
        file.write(f"| Minimum content length | {minimum_threshold['minimum_content_length']} |\n\n")

        file.write("## Scoring Weights\n\n")
        file.write("| Score Component | Weight |\n")
        file.write("|---|---:|\n")
        file.write(f"| Enterprise organization match | {scoring_weights['enterprise_organization_match']} |\n")
        file.write(f"| Repository keyword match | {scoring_weights['repo_keyword_match']} |\n")
        file.write(f"| Documentation keyword match | {scoring_weights['documentation_keyword_match']} |\n")
        file.write(f"| Negative keyword penalty | {scoring_weights['negative_keyword_penalty']} |\n\n")

        file.write("## Project Selection Summary\n\n")
        file.write("| Metric | Count |\n")
        file.write("|---|---:|\n")
        file.write(f"| Original projects | {original_count} |\n")
        file.write(f"| Selected enterprise projects | {selected_count} |\n")
        file.write(f"| Removed projects | {removed_count} |\n")
        file.write(f"| Selected percentage | {selected_percentage}% |\n\n")

        file.write("## Top 10 Selected Projects\n\n")

        if len(top_projects) > 0:
            file.write("| Rank | Repository ID | Docs | Content Length | Enterprise Score |\n")
            file.write("|---:|---|---:|---:|---:|\n")

            for rank, (_, row) in enumerate(top_projects.iterrows(), start=1):
                file.write(
                    f"| {rank} | `{row['repo_id']}` | {row['doc_count']} | {row['total_content_length']} | {row['total_enterprise_score']} |\n"
                )
        else:
            file.write("No projects were selected.\n")

        file.write("\n## Scoring Method\n\n")
        file.write("The following enterprise selection method was applied:\n\n")
        file.write("1. Loaded enterprise organizations, repository keywords, documentation keywords, negative keywords, scoring weights, and thresholds from JSON configuration.\n")
        file.write("2. Checked the repository owner against known enterprise/open-source organizations.\n")
        file.write("3. Checked repository name and repository ID against repository keywords.\n")
        file.write("4. Checked documentation paths and documentation text against categorized documentation keywords.\n")
        file.write("5. Applied negative keyword penalties for demo, tutorial, personal, and non-enterprise indicators.\n")
        file.write("6. Calculated total enterprise score.\n")
        file.write("7. Selected projects that passed the minimum enterprise score, documentation count, and content length thresholds.\n\n")

        file.write("## Output Columns\n\n")
        for column in selected_df.columns:
            file.write(f"- `{column}`\n")

    print(f"Enterprise selection summary Markdown saved to: {summary_file_path}")

    return summary_file_path


# Common method to select enterprise-level projects
def select_enterprise_projects(input_parquet_file_path, config_file_path, output_folder_path):
    original_df, input_parquet_file_path = read_summarized_dataset(input_parquet_file_path)

    config, config_file_path = read_enterprise_config(config_file_path)

    validate_required_dataset_columns(original_df)
    validate_required_config_sections(config)

    scored_df = score_enterprise_projects(original_df, config)

    selected_df, scored_df = apply_enterprise_selection_thresholds(
        scored_df,
        config
    )

    selected_df = reorder_output_columns(selected_df)
    scored_df = reorder_output_columns(scored_df)

    print_selection_summary(
        original_df,
        selected_df,
        config
    )

    selected_file_path = save_selected_enterprise_projects(
        selected_df,
        input_parquet_file_path,
        output_folder_path
    )

    scored_file_path = save_all_scored_projects(
        scored_df,
        input_parquet_file_path,
        output_folder_path
    )

    summary_file_path = save_selection_summary_markdown(
        original_df,
        selected_df,
        scored_df,
        input_parquet_file_path,
        output_folder_path,
        config,
        config_file_path
    )

    return selected_file_path, scored_file_path, summary_file_path


if __name__ == "__main__":
    input_file_path = input("Enter the summarized GoodDocs Parquet file path: ").strip()
    input_file_path = input_file_path.strip('"').strip("'")

    config_file_path = input("Enter the enterprise keyword JSON config file path: ").strip()
    config_file_path = config_file_path.strip('"').strip("'")

    output_folder_path = input("Enter the output folder path: ").strip()
    output_folder_path = output_folder_path.strip('"').strip("'")

    select_enterprise_projects(
        input_parquet_file_path=input_file_path,
        config_file_path=config_file_path,
        output_folder_path=output_folder_path
    )