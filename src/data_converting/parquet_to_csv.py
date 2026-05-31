import pandas as pd
from pathlib import Path


def convert_parquet_to_csv(parquet_file_path):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    # Save CSV in the same folder with the same file name
    output_file_path = parquet_file_path.with_suffix(".csv")

    print(f"Reading Parquet file: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    df.to_csv(output_file_path, index=False, encoding="utf-8")

    print("\nCSV created successfully.")
    print(f"Output file: {output_file_path}")
    print("\nColumns:")
    print(list(df.columns))

    return output_file_path


def convert_parquet_to_sample_csv(parquet_file_path, sample_size):
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if parquet_file_path.suffix.lower() != ".parquet":
        raise ValueError("Input file must be a .parquet file")

    # Save sample CSV in the same folder
    output_file_path = parquet_file_path.with_name(
        f"{parquet_file_path.stem}_sample_{sample_size}.csv"
    )

    print(f"Reading first {sample_size} rows from: {parquet_file_path}")

    # Read parquet file
    df = pd.read_parquet(parquet_file_path)

    # Get only first sample_size rows
    sample_df = df.head(sample_size)

    sample_df.to_csv(output_file_path, index=False, encoding="utf-8")

    print("\nSample CSV created successfully.")
    print(f"Output file: {output_file_path}")
    print(f"Sample rows: {len(sample_df)}")
    print(f"Columns: {len(sample_df.columns)}")
    print("\nColumns:")
    print(list(sample_df.columns))

    return output_file_path


if __name__ == "__main__":
    file_path = input("Enter the parquet file path: ").strip()

    # Remove quotes if user copied path with quotes
    file_path = file_path.strip('"').strip("'")

    print("\nChoose an option:")
    print("1. Convert full Parquet file to CSV")
    print("2. Create sample CSV")

    choice = input("Enter your choice 1 or 2: ").strip()

    if choice == "1":
        convert_parquet_to_csv(file_path)

    elif choice == "2":
        sample_size_input = input("Enter sample size: ").strip()

        if not sample_size_input.isdigit():
            print("Invalid sample size. Please enter a positive number.")
        else:
            sample_size = int(sample_size_input)

            if sample_size <= 0:
                print("Sample size must be greater than 0.")
            else:
                convert_parquet_to_sample_csv(file_path, sample_size=sample_size)

    else:
        print("Invalid choice. Please enter 1 or 2.")