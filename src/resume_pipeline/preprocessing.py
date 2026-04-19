import html 
import re 
import unicodedata
from typing import Optional, Set 

import pandas as pd 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from ..utils.config import TEXT_COLUMN , CLEANED_TEXT_COLUMN


# ==========================
# Compiled Ragex Patterns 
# ==========================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d().\-\s]{7,}\d)\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
BULLET_PATTERN = re.compile(r"[•▪■►▸●◦◆◇]")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-z0-9\s]")



# =========================
# Technology Token Normalization
# Keep important technical tokens meaningful
# =========================
TECH_REPLACEMENTS = [
    (re.compile(r"\bc\+\+\b"), "cpp"),
    (re.compile(r"\bc#\b"), "csharp"),
    (re.compile(r"\b\.net\b"), "dotnet"),
    (re.compile(r"\bnode\.js\b"), "nodejs"),
    (re.compile(r"\breact\.js\b"), "reactjs"),
    (re.compile(r"\bnext\.js\b"), "nextjs"),
    (re.compile(r"\bvue\.js\b"), "vuejs"),
    (re.compile(r"\bangular\.js\b"), "angularjs"),
    (re.compile(r"\bmachine-learning\b"), "machine learning"),
    (re.compile(r"\bdeep-learning\b"), "deep learning"),
]


def normalize_technology_tokens(text: str) -> str: 
    """
    Normalize common technology terms before removing punctuation. 

    Examples: 
        c++ -> cpp
        c# -> csharp
        .net -> dotnet 
        node.js -> nodejs 

    Args: 
        text: Input text. 

    Returns: 
        Text with normalized technology tokens. 
    """

    normalized_text = text 
    for pattern, replacement in TECH_REPLACEMENTS:
        normalized_text = pattern.sub(replacement , normalized_text)
    return normalized_text


def clean_resume_text(
        text: object, 
        remove_stopwords: bool = False, 
        extra_stopwords: Optional[Set[str]] = None,
) -> str: 
    """
    Clean a single resume text string. 

    Cleaning steps: 
    1. Handle missing values safely 
    2. Decode HTML entities 
    3. Normalikze Unicode 
    4. Lowercase text 
    5. Normalize imporant technical tokens 
    6. Remove HTML tags 
    7. Remove URLs, emails, and phone numbers 
    8. Remove bullets and punctuation/noisy symbols 
    9. Normalize whitespace
    10. Optionally remove English stopwords 

    Args: 
        text: Raw resume text
        remove_stopwords: Whether to remove English stopwords.
        extra_stopwords: Optionally remove additional stopwords.

    Returns: 
        Cleaned resume text as a string.
    """

    if pd.isna(text): 
        return ""
    
    cleaned_text = str(text)

    # Decode HTML entites like &amp; 
    cleaned_text = html.unescape(cleaned_text)

    # Normalize unicode characters
    cleaned_text = unicodedata.normalize("NFKC", cleaned_text)

    # Replace non-breaking spaces 
    cleaned_text = cleaned_text.replace("\xa0", " ")

    # Lowercase 
    cleaned_text = cleaned_text.lower()

    # Normalize tech tokens before punctuation removal 
    cleaned_text = normalize_technology_tokens(cleaned_text)

    # Remove HTML tags if any noisy fragments still exist 
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)

    # Remove links, emails, phone numbers 
    cleaned_text = URL_PATTERN.sub(" ", cleaned_text)
    cleaned_text = EMAIL_PATTERN.sub(" ", cleaned_text)
    cleaned_text = PHONE_PATTERN.sub(" ", cleaned_text)

    # Replace bullets with spaces 
    cleaned_text = BULLET_PATTERN.sub(" ", cleaned_text)

    # Normalize ampersand to word 
    cleaned_text = cleaned_text.replace("&" , " and ")

    # Remove punctuation / special symbols 
    # Remove letters, digits, and spaces 
    cleaned_text = NON_ALPHANUM_PATTERN.sub(" " , cleaned_text)

    # Normalize repeated whitespaces 
    cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()

    # Optional stopwords removal 
    if remove_stopwords and cleaned_text:
        stopwords = set(ENGLISH_STOP_WORDS)

        if extra_stopwords:
            stopwords.update(word.lower() for word in extra_stopwords)

        protected_tokens = {"C", "r", "go", "ai", "ml"} # short but meaningful technical tokens.

        tokens = [
            token 
            for token in cleaned_text.split()
            if token not in stopwords or token in protected_tokens
        ] 

        cleaned_text =" " .join(tokens)
    return cleaned_text


def preprocess_text_series(
        text_series: pd.Series,
        remove_stopwords: bool = False,
        extra_stopwords: Optional[Set[str]] = None,
) -> pd.Series:
    """
    Apply text cleaning to a pandas series of resume texts.

    Args: 
        text_series: Series containing raw resume text.
        remove_stopwords: whether to remove stopwords.
        extra_stopwords: Optional custom stopwords.

    Returns: 
        Aseries of cleaned resume texts.
    """

    return text_series.apply(
        lambda text: clean_resume_text(
            text = text, 
            remove_stopwords= remove_stopwords, 
            extra_stopwords= extra_stopwords
        )
    )


def preprocess_resume_dataframe(
        df: pd.DataFrame,
        text_column: str = TEXT_COLUMN,
        output_column: str = CLEANED_TEXT_COLUMN,
        remove_stopwords: bool = False,
        extra_stopwords: Optional[set[str]] = None,
        drop_empty: bool = True, 
) -> pd.DataFrame:
    """
    Add a cleaned text column to the resume dataframe.
    
    Args: 
        df: Input dataframe containing resume text.
        text_column: source text column name 
        output_column: Destination cleaned text column name.
        remove_stopwords: whether to remove stopwords.
        extra_stopwords: Optional custom stopwords
        drop_empty: Whether to drop rows with emty cleaned text.

    Returns: 
        A new dataframe with the cleaned text column added.
    """
    if text_column not in df.columns: 
        raise ValueError(
            f"Column '{text_column}' not found in dataframe. "
            f"Available columns: '{df.columns.tolist()}'"
        )

    processed_df = df.copy()

    processed_df[output_column] = preprocess_resume_dataframe(
        text_series = processed_df[text_column],
        remove_stopwords= remove_stopwords,
        extra_stopwords= extra_stopwords
    )

    if drop_empty: 
        processed_df = processed_df[processed_df[output_column].str.strip() != ""]
        processed_df = processed_df.reset_index(drop = True)
    
    return processed_df


def clean_single_resume_for_inference(text: str) -> str:
    """
    clean a single resume text for prediction/inference use.

    this helper is useful later when you want to classify one new resume. 

    Args: 
        text: Raw resume text 
    
    Returns: 
        cleaned resume text.
    """

    return clean_resume_text(text=text, remove_stopwords=False)


