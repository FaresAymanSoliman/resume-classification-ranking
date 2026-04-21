from pathlib import Path 
from typing import Iterable, Optional 

import joblib 
import pandas as pd 
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.config import (
    CLEANED_TEXT_COLUMN,
    TEXT_COLUMN,
    VECTORIZER_PATH,
)


def validate_text_column (df: pd.DataFrame, text_column: str) -> None:
    """
    Validate that the required text column exists in the dataframe.

    Args: 
        df: Input dataframe
        text_column: Name of the text column to use 

    Raises: 
        ValueError: If the column is not found 
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in dataframe."
            f"Available columns: {df.columns.tolist()}"
        )
    

def create_tfidf_vectorizer(
        max_features: Optional[int] = 5000,
        ngram_range: tuple[int,int] = (1, 2),
        min_df: int = 2,
        max_df: int = 0.95,
        sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with reasonable baseline settings. 

    Args: 
        max_features: Maximum number of feature to keep.
        ngram_range: The lower and upper boundary of n-grams.
        min_df: Minimum document frequency for terms to be included.
        max_df: Maximum document frequency threshold. 
        sublinear_tf: Apply sublinear term frequency scaling. 

    Returns: 
        Configured Tfidfvectorizer instance.
    """

    return TfidfVectorizer(
        max_features=max_features, 
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )


def fit_vectorizer(
        texts: Iterable[str],
        vectorizer: Optional[TfidfVectorizer] = None,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Fit a TF-IDF vectorizer on input texts and return the fitted vectorizer and transformed sparse matrix.

    Args: 
        texts: Iterable of cleaned text in documents.
        vectorizer: Optional preconfigured TF-IDF vectorizer. 

    Returns:
        A tuple of:
            - fitted vectorizer
            - TF-IDF feaure matrix
    """
    if vectorizer is None: 
        vectorizer = create_tfidf_vectorizer()

    text_list = list(texts)
    features = vectorizer.fit_transform(text_list)

    return vectorizer, features 


def transform_texts(
        texts: Iterable[str], 
        vectorizer: TfidfVectorizer, 
) -> csr_matrix: 
    """
    Transform input texts into TF-IDF features using a fitted vectorizer. 

    Args: 
        texts: Iterable of cleaned text documents.
        vectorizer: Fitted TF-IDF vectorizer.

    Returns: 
        Sparse TF-IDF feature matrix. 
    """
    text_list = list(texts)
    return vectorizer.transform(text_list)


def fit_transform_resume_features(
        df: pd.DataFrame,
        text_column: str = CLEANED_TEXT_COLUMN,
        vectorizer: Optional[TfidfVectorizer] = None,
) -> tuple[TfidfVectorizer , csr_matrix]: 
    """
    Fit a TF-IDF vectorizer on a dataframe text column and return both 
    the fitted vectorizer and the transformed feature matrix. 

    Args:
        df: Input dataframe. 
        vectorizer: Fitted TF-IDF vectorizer.
        text_column: Column containing cleaned text.

    Returns: 
        A tuple of: 
            - Fitted vectorizer.
            - sparse TF-IDF matrix
    """
    validate_text_column(df, text_column)

    texts = df[text_column].fillna("").astype(str)
    return fit_vectorizer(texts=texts, vectorizer=vectorizer)


def transform_resume_features(
        df: pd.DataFrame,
        vectorizer: TfidfVectorizer,
        text_column: str = CLEANED_TEXT_COLUMN,
) -> csr_matrix:
    """
    Transform a dataframe text column into TF-IDF features using 
    an already fitted vectorizer

    Args: 
        df: Input dataframe.
        vectorizer: Fitted TF-IDF vectorizer.
        text_column: Column containing cleaned text.

    Returns: 
        Sparse TF-IDF feature matrix 
    """
    validate_text_column(df, text_column)

    texts = df[text_column].fillna("").astype(str)
    return transform_texts(texts=texts, vectorizer=vectorizer)


def save_vectorizer(
        vectorizer: TfidfVectorizer,
        output_path: Path | str = VECTORIZER_PATH,
) -> None:
    """
    Save a fitted TF-IDF vectorizer to disk.

    Args:  
        vectorizer: Fitted vectorizer.
        output_path: Path to save the vectorizer
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, output_path)
    print(f"Vectorizer saved to: {output_path}")


def load_vectorizer(
        input_path: Path | str = VECTORIZER_PATH,
) -> TfidfVectorizer: 
    """
    Load a previously saved TF-IDF vectorizer from disk.

    Args: 
        input_path: Path to the saved vectorizer file.

    Returns: 
        Loaded tfidfvectorized
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {input_path}")
    
    return joblib.load(input_path)


def get_feature_names(vectorizer: TfidfVectorizer) -> list: 
    """
    Get the learned TF-IDF feature names.

    Args: 
        vectorizer: Fitted TF-IDF vectorizer.

    Returns: 
        List of feature names.
    """
    return vectorizer.get_feature_names_out().tolist()


def get_top_terms_for_document(
        feature_matrix: csr_matrix,
        vectorizer: TfidfVectorizer,
        document_index: int,
        top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the top TF-IDF terms for a single document.

    Args: 
        feature_matrix: Sparse TF-IDF feature matrix.
        vectorizer: Fitted TF-IDF vectorizer.
        document_index: Index of the document row. 
        top_k: Number of terms to return.

    Returns: 
        List of (term, score) sorted vy descending TF-IDF score.
    """
    if document_index < 0 or document_index >= feature_matrix.shaperaise: IndexError(
        f"document_index {document_index} out of range for "
        f"{feature_matrix.shape[0]} documents."
    )
        
    row = feature_matrix[document_index].toarray().fillna()
    feature_names = vectorizer.get_feature_names_out()

    top_indices = row.argsor()[::-1][:top_k]
    top_terms = [
        (feature_names[i], float(row[i]))
        for i in top_indices
        if row[i] > 0 
    ]

    return top_terms