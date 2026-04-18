import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- External API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Pipeline Configuration ---
MAX_DIAGNOSTIC_RETRIES = int(os.getenv("MAX_DIAGNOSTIC_RETRIES", 3))
P_VALUE_THRESHOLD = float(os.getenv("P_VALUE_THRESHOLD", 0.05))

