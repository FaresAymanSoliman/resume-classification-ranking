from typing import Dict, Tuple

import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ..utils.file_loader import (
    load_and_prepare_resume_data,
    save_dataframe,
)
from .preprocessing import preprocess_resume_dataframe
from .features import (
    fit_transform_resume_features,
    save_vectorizer,
)
from .model import (
    encode_labels,
    split_data,
    train_classifier,
    evaluate_classifier,
    save_model,
)
from ..utils.config import (
    CLEANED_TEXT_COLUMN,
    PROCESSED_RESUME_FILE, 
)


# =========================
# Full Training Pipeline
# =========================

def train_resume_classifier_pipeline(
        save_outputs: bool = True,
        remove_stopwords: bool = False, 
) -> Dict:
    """
    Full end-to-end pipeline for resume classification.

    Steps: 
    1. Load raw resume dataset. 
    2. Preprocess text.
    3. Extract TF-IDF features. 
    4. Encode labels.
    5. Split into train/test. 
    6. Train classifier.
    7. Evaluate model.
    8. Save model & vectorize (optional)
    9. Save processed dataset (optional)
    
    Args: 
        save_outputs: Wether to save model, vectorizer, and processed data.
        remove_stopwords: Wether to remove stopwords during preprocessing.

    Returns: 
        Dictionary containing trained objects and evaluation metrics
    """
    
    print(" Loading resume data...")
    df = load_and_prepare_resume_data()
    print(f"  Loaded {len(df)} resumes")

    print(" Preprocessing resumes...")
    df_processed = preprocess_resume_dataframe(
        df,
        remove_stopwords=remove_stopwords,
    )
    print("  Preprocessing completed")

    print(" Extracting TF-IDF features...")
    vectorizer, X = fit_transform_resume_features(
        df_processed,
        text_column=CLEANED_TEXT_COLUMN,
    )
    print(f"  Feature matrix shape: {X.shape}")

    print(" Encoding labels...")
    y_encoded, label_encoder = encode_labels(
        df_processed["category"].values
    )

    print(" Splitting data into train/test...")
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    print(" Training classifier...")
    model = train_classifier(X_train, y_train)

    print(" Evaluating classifier...")
    metrics = evaluate_classifier(
        model=model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
    )

    print(f" Accuracy: {metrics['accuracy']:.4f}")

    if save_outputs:
        print(" Saving model and vectorizer...")
        save_model(model, label_encoder)
        save_vectorizer(vectorizer)

        print(" Saving processed dataset...")
        save_dataframe(df_processed, PROCESSED_RESUME_FILE)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "processed_dataframe": df_processed,
    }


# =========================
# Helper Function
# =========================

def quick_train() -> None:
    """
    Quick wrapper to train the resume classification pipeline.
    Useful for experiments or CLI execution.
    """
    train_resume_classifier_pipeline()
