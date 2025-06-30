from pathlib import Path
import os
import logging

LOGGING_LEVEL = logging.DEBUG

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "daily_trackers"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_TESTING_OUTPUT_DIR = DATA_DIR / "model_testing_output"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist (optional, but good practice for project setup)
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODEL_TESTING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)