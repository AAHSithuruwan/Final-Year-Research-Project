import numpy as np
import pandas as pd
from analysis_functions.constants import (
    DOCUMENTATION_METRIC_COLUMNS,
    DOCUMENTATION_QUALITY_SCORE_COLUMN,
    MAINTENANCE_SCORE_COLUMN,
)

# Validate that the dataset has the required columns and is not empty.
def validate_required_columns(dataframe):
    if not dataframe.columns.is_unique:
        raise ValueError("The dataset has duplicate column names. Use unique column names.")
    required = ["repo_id", *DOCUMENTATION_METRIC_COLUMNS, MAINTENANCE_SCORE_COLUMN]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if dataframe.empty:
        raise ValueError("The dataset has no repository rows. Upload a non-empty dataset.")


# Perform numeric analysis on specified columns.
def numeric_analysis_columns(dataframe, columns):
    prepared = dataframe.copy()
    conversion_rows = []
    for column in columns:
        original = dataframe[column]
        values = pd.to_numeric(original, errors="coerce").astype(float)
        values = values.replace([np.inf, -np.inf], np.nan)
        invalid_count = int((original.notna() & values.isna()).sum())
        conversion_rows.append({"Column": column, "Invalid Values Converted to Missing": invalid_count})
        prepared[column] = values
    return prepared, pd.DataFrame(conversion_rows)


# Create a documentation quality score for each row in the dataframe.
def create_documentation_quality_score(dataframe, metric_columns):
    prepared = dataframe.copy()
    if DOCUMENTATION_QUALITY_SCORE_COLUMN not in prepared.columns:
        numeric, _ = numeric_analysis_columns(prepared, metric_columns)
        prepared[DOCUMENTATION_QUALITY_SCORE_COLUMN] = numeric[metric_columns].mean(axis=1)
    return prepared


# Prepare the dataset for analysis, returning a copy, conversion audit, and score creation status.
def prepare_dataset(dataframe):
    validate_required_columns(dataframe)
    score_created = DOCUMENTATION_QUALITY_SCORE_COLUMN not in dataframe.columns
    columns = [*DOCUMENTATION_METRIC_COLUMNS, MAINTENANCE_SCORE_COLUMN]
    if not score_created:
        columns.append(DOCUMENTATION_QUALITY_SCORE_COLUMN)
    prepared, conversions = numeric_analysis_columns(dataframe, columns)
    prepared = create_documentation_quality_score(prepared, DOCUMENTATION_METRIC_COLUMNS)
    return prepared, conversions, score_created


# Missing value summary for every column
def missing_value_summary(dataframe):
    counts = dataframe.isna().sum()
    return pd.DataFrame({
        "Column": counts.index,
        "Missing Count": counts.values,
        "Missing Percentage": counts.values / max(len(dataframe), 1) * 100,
    }).sort_values("Missing Count", ascending=False, kind="stable", ignore_index=True)
