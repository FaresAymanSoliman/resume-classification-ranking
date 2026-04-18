from pathlib import Path
from typing import Union 
import pandas as pd 
from .config import (
RAW_RESUME_FILE,
    PROCESSED_RESUME_FILE,
    REQUIRED_SOURCE_COLUMNS,
    SOURCE_ID_COLUMN,
    SOURCE_TEXT_COLUMN,
    SOURCE_LABEL_COLUMN,
    ID_COLUMN,
    TEXT_COLUMN,
    LABEL_COLUMN,
)

def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file exists and return it as a Path object 

    Args: 
        file_Path: Path to the target file. 

    Returns: 
        Path object for the validated file. 
    
    Raises: 
        FileNotFoundError: If the file does not exist 
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path 


def read_csv_with_fallbacks(file_path: Union[str,Path]) -> pd.DataFrame:
    """
    Read CSV file using a small set of encoding fallbacks. 

    This helps with datasets that may contain unusual characters. 

    Args: 
        file_path: Path to the CSV file. 
    
    Reurns: 
        loaded pandas DataFrame. 
    
    Raises: 
        ValueError: If the file cannot be decoded with supported encodings. 
    """

    path = validate_file_exists(file_path)

    encodings = ["utf-8" , "utf-8-sig" , "latin-1"]
    last_exception = None 

    for encoding in encodings: 
        try: 
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc: 
            last_exception = exc 
    
    raise ValueError(
        f"Unable to read CSV file with supported encodings: {encodings}."
        f"Please check the file encoding for: {path}."
    ) from last_exception


def validate_required_columns(
        df: pd.DataFrame,
        required_columns: list[str] = REQUIRED_SOURCE_COLUMNS, 
    ) -> None: 
    """
    Validate that the input dataframe contains all required source columns. 
    
    Args: 
        df: Input dataframe loaded from from the raw dataset. 
        required_columns: Required columns expected in the dataframe. 

    Raises: 
        ValueError: If one or more required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns: 
        raise ValueError(
            f"Missing required coulunms: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    

def standardize_resume_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename source dataset columns into the project's internal schema. 

    Source schems example: 
        ID , Resume_str, Category

    Standardized schema: 
        resume_id, resume_text, category

    Args: 
        df: Raw input dataframe.
    
    Returns: 
        Standardized dataframe with only the required baseline column. 
    """

    validate_required_columns(df)

    column_mapping = {
        SOURCE_ID_COLUMN: ID_COLUMN,
        SOURCE_TEXT_COLUMN: TEXT_COLUMN, 
        SOURCE_LABEL_COLUMN: LABEL_COLUMN,
    }

    standarized_df = df.rename(columns =column_mapping) .copy()

    # Keep only the core columns needed for the baseline pipeline 
    standarized_df = standarized_df[[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]]

    return standarized_df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are invalid for the baseline classification pipeline. 

    Rules: 
    - Drop rows where resume text or category is missing
    - Strip whitespace 
    - Remove empty strings after stripping
    - Remove duplicate resume IDs if present 

    Args: 
        df: Standarized dataframe

    Returns: 
        Cleaned dataframe with valid rows only. 
    """

    cleaned_df = df.copy()

    # Drop rows missing required fields 
    cleaned_df = cleaned_df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    # Normalize text fields as strings 
    cleaned_df[TEXT_COLUMN] = cleaned_df[TEXT_COLUMN].astype(str).str.strip()
    cleaned_df[LABEL_COLUMN] = cleaned_df[LABEL_COLUMN].astype(str).str.strip()

    # Remove empty values after stripping 
    cleaned_df = cleaned_df[cleaned_df[TEXT_COLUMN] != ""]
    cleaned_df = cleaned_df[cleaned_df[LABEL_COLUMN] != ""]

    # Drop duplicate IDs if they exist 
    cleaned_df = cleaned_df.drop_duplicates(subset=[ID_COLUMN])

    # Reset index for clean downstream usage 
    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df


def load_raw_resume_data(
        file_path: Union[str, Path] = RAW_RESUME_FILE,
) -> pd.DataFrame: 
    """
    Load the raw resume dataset from disk. 

    Args: 
        file_path: Path to the raw CSV dataset.

    Returns: 
        Raw dataframe.  
    """

    return read_csv_with_fallbacks(file_path)

    
def load_and_prepare_resume_data(
        file_path: Union[str, Path] = RAW_RESUME_FILE, 
) -> pd.DataFrame: 
    """
    Full loading pipeline for the baseline resume dataset. 

    Steps: 
    1. Load raw CSV 
    2. Validate required columns 
    3. Standardize schema 
    4. Remove invalid rows 

    Args: 
        file_path: Path to the raw CSV dataset. 

    Returns: 
        Prepared dataframe ready for preprocessing. 
    """

    raw_df = load_raw_resume_data(file_path)
    standarized_df = standardize_resume_dataframe(raw_df)
    prepared_df = remove_invalid_rows(standarized_df)

    return prepared_df


def save_dataframe(
        df: pd.DataFrame,
        output_path: Union[str, Path] = PROCESSED_RESUME_FILE,
) -> None: 
    """
    Save a dataframe to CSV, Creating parent directories if needed. 

    Args: 
        df: Dataframe to save. 
        output_path: Destination CSV path. 
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Dataframe saved to: {output_path}")
