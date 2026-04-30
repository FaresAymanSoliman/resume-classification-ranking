from pathlib import Path
from typing import Tuple

import joblib
import numpy as np 
from scipy.sparse import csr_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..utils.config import (
    LABEL_COLUMN,
    CLASSIFIER_PATH,
    RANDOM_STATE,
    TEST_SIZE
)



# =========================
# Label Encoding
# =========================

def encode_labels (labels: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode target lablels into numeric form .

    Args: 
        labels: Array-like of category labels.

    Returns: 
        Encoded labels and fitted LabelEncoder.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder


# =========================
# Train / Test Split
# =========================

def split_data(
        x: csr_matrix,
        y: np.ndarray,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
): 
    """
    Split data into train and test sets. 

    Args:
        x: Feature matrix.
        y: Target labels.
        test_size: Proportion of test data. 
        random_state: Random seed.

    Returns:  
        X_train , Y_train , X_test , Y_test 
    """
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# =========================
# Model Creation
# =========================

def create_classifier() -> LogisticRegression: 
    """
    Create a logistic regression classifier with good baseline settings.

    Returns: 
        LogisticRegression instance.
    """
    return LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


# =========================
# Training
# =========================

def train_classifier(
        X_train: csr_matrix,
        y_train: np.ndarray,
        model: LogisticRegression | None = None,
) -> LogisticRegression:
    """
    Train a classifier on feature matrix and labels.

    Args: 
        X_train: Training features. 
        y_train: Training labels. 
        model: Optional existing model. 

    Returns: 
        Trained classifier
    """
    if model is None:
        model = create_classifier()

    model.fit(X_train,y_train)
    return model


# =========================
# Evaluation
# =========================

def evaluate_classifier(
        model: LogisticRegression,
        X_test: csr_matrix,
        y_test: np.ndarray,
        label_encoder: LabelEncoder,
) -> dict:
    """
    Evaluate classifier performance on test data.

    Args: 
        model: Trained classifier.
        X_test: Test features.
        y_test: True test labels. 
        label_encoder: Label encoder for inverse mapping. 

    Returns: 
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    conf_matrix = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
    }


# =========================
# Save / Load Model
# =========================

def save_model(
        model: LogisticRegression,
        label_encoder: LabelEncoder,
        output_path: Path | str = CLASSIFIER_PATH,
) -> None:
    """
    Save trained model and label encoder 

    Args: 
        model: Trained classifier. 
        label_encoder: Fitted LabelEncoder.
        output_path: Path to save model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True , exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "Label_encoder": label_encoder,
        },
        output_path,
    )

    print(f"Model saved to {output_path}")


def load_model(
        input_path: Path | str = CLASSIFIER_PATH,
) -> tuple[LogisticRegression, LabelEncoder]:
    """
    Load a saved model and label encoder.
    
    Args:
        input_path: Path to saved file. 
    
    Returns: 
        tuple of (model, label encoder).
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
    
    data = joblib.load(input_path)
    return data["model"], data["label_encoder"]


# =========================
# Prediction
# =========================

def predict(
    model: LogisticRegression,
    X: csr_matrix,
    label_encoder: LabelEncoder,
) -> np.ndarray:
    """
    Predict category labels from features.

    Args:
        model: Trained classifier.
        X: Feature matrix.
        label_encoder: Label encoder.

    Returns:
        Array of predicted category names.
    """
    encoded_preds = model.predict(X)
    return label_encoder.inverse_transform(encoded_preds)
