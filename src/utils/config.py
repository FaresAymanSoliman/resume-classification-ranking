from pathlib import Path

# =========================
# Project Root Directories
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# =========================
# Raw Dataset File
# =========================
RAW_RESUME_FILE = RAW_DATA_DIR / "Resume.csv"

# =========================
# Processed Output Files
# =========================
PROCESSED_RESUME_FILE = PROCESSED_DATA_DIR / "processed_resumes.csv"

# =========================
# Model Artifact Paths
# =========================
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"

# =========================
# Original Kaggle Column Names
# Dataset columns:
# ID, Resume_str, Resume_html, Category
# =========================
SOURCE_ID_COLUMN = "ID"
SOURCE_TEXT_COLUMN = "Resume_str"
SOURCE_HTML_COLUMN = "Resume_html"
SOURCE_LABEL_COLUMN = "Category"

# =========================
# Standardized Internal Column Names
# =========================
ID_COLUMN = "resume_id"
TEXT_COLUMN = "resume_text"
LABEL_COLUMN = "category"
CLEANED_TEXT_COLUMN = "cleaned_resume_text"

# =========================
# Required Columns for Baseline Pipeline
# =========================
REQUIRED_SOURCE_COLUMNS = [
    SOURCE_ID_COLUMN,
    SOURCE_TEXT_COLUMN,
    SOURCE_LABEL_COLUMN,
]

# =========================
# Misc Settings
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.2