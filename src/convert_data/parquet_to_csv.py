import pandas as pd
from pathlib import Path


# Truncate text columns to a specified maximum character length, clean whitespace, and replace newlines and tabs with spaces
def truncate_text_columns(df, max_char_length):
    df = df.copy()

    text_columns = [
        column for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace("\r\n", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.strip()
        )

        if max_char_length is not None:
            df[column] = df[column].str.slice(0, max_char_length)

    return df


# Convert Parquet to CSV with optional text truncation
def convert_parquet_to_csv(parquet_file_path, max_char_length=None):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    output_file_path = parquet_file_path.with_suffix(".csv")

    print(f"Reading Parquet file: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    df = truncate_text_columns(df, max_char_length)

    df.to_csv(output_file_path, index=False, encoding="utf-8")

    print("\nCSV created successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if max_char_length is not None:
        print(f"Text columns truncated to maximum {max_char_length} characters.")

    print("\nColumns:")
    print(list(df.columns))

    return output_file_path


# Convert Parquet to sample CSV with optional text truncation
def convert_parquet_to_sample_csv(parquet_file_path, sample_size, max_char_length=None):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    output_file_path = parquet_file_path.with_name(
        f"{parquet_file_path.stem}_sample_{sample_size}.csv"
    )

    print(f"Reading first {sample_size} rows from: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    sample_df = df.head(sample_size)

    sample_df = truncate_text_columns(sample_df, max_char_length)

    sample_df.to_csv(output_file_path, index=False, encoding="utf-8")

    print("\nSample CSV created successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Sample rows: {len(sample_df)}")
    print(f"Columns: {len(sample_df.columns)}")

    if max_char_length is not None:
        print(f"Text columns truncated to maximum {max_char_length} characters.")

    print("\nColumns:")
    print(list(sample_df.columns))

    return output_file_path


# Get maximum character length for text columns from user input
def get_max_char_length_from_user():
    max_char_input = input(
        "Enter maximum character length for text columns, or press Enter for no limit: "
    ).strip()

    if max_char_input == "":
        return None

    if not max_char_input.isdigit():
        raise ValueError("Maximum character length must be a positive number.")

    max_char_length = int(max_char_input)

    if max_char_length <= 0:
        raise ValueError("Maximum character length must be greater than 0.")

    return max_char_length


# Get sample size from user input
def get_sample_size_from_user():
    sample_size_input = input("Enter sample size: ").strip()

    if not sample_size_input.isdigit():
        raise ValueError("Sample size must be a positive number.")

    sample_size = int(sample_size_input)

    if sample_size <= 0:
        raise ValueError("Sample size must be greater than 0.")

    return sample_size


if __name__ == "__main__":
    file_path = input("Enter the parquet file path: ").strip()

    file_path = file_path.strip('"').strip("'")

    print("\nChoose an option:")
    print("1. Convert full Parquet file to CSV")
    print("2. Create sample CSV")

    choice = input("Enter your choice 1 or 2: ").strip()

    if choice == "1":
        max_char_length = get_max_char_length_from_user()

        convert_parquet_to_csv(
            parquet_file_path=file_path,
            max_char_length=max_char_length
        )

    elif choice == "2":
        sample_size = get_sample_size_from_user()
        max_char_length = get_max_char_length_from_user()

        convert_parquet_to_sample_csv(
            parquet_file_path=file_path,
            sample_size=sample_size,
            max_char_length=max_char_length
        )

    else:
        print("Invalid choice. Please enter 1 or 2.")