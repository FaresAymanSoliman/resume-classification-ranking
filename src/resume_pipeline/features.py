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
