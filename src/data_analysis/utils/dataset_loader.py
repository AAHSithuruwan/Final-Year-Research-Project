from io import BytesIO
from pathlib import Path
import pandas as pd

# Load dataset from file bytes and filename, supporting .parquet and .csv formats
def load_dataset(file_bytes, filename):
    extension = Path(filename).suffix.lower()
    if extension not in {".parquet", ".csv"}:
        raise ValueError("Unsupported file format. Upload a .parquet or .csv dataset.")
    try:
        source = BytesIO(file_bytes)
        if extension == ".parquet":
            return pd.read_parquet(source, engine="pyarrow")
        return pd.read_csv(source, encoding="utf-8-sig")
    except Exception as error:
        raise ValueError(
            "The dataset could not be read. Check that it is a valid Parquet or UTF-8 CSV "
            "file, is not corrupted, and has a header row for CSV."
        ) from error
