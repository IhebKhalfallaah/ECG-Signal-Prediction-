"""Application settings. Credentials come from environment variables (defaults are for local demos only)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config():
    return {
        "SECRET_KEY": os.environ.get("SECRET_KEY", "your_secret_key"),
        "USERNAME": os.environ.get("ECG_USERNAME", "admin"),
        "PASSWORD": os.environ.get("ECG_PASSWORD", "1234"),
        "MODEL_PATH": str(BASE_DIR / "models" / "svm_model_C60_gamma001.pkl"),
        "UPLOAD_FOLDER": str(BASE_DIR / "uploads"),
        "PLOTS_DIR": str(BASE_DIR / "app" / "static" / "plots"),
    }
