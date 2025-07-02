from pathlib import Path
import os
import logging
import streamlit as st

LOGGING_LEVEL = logging.DEBUG

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "classifier_data"
RAW_DATA_DIR = DATA_DIR / "daily_trackers"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_OUTPUT_DIR = DATA_DIR / "model_output"
LOGS_DIR = DATA_DIR / "logs"
PROCESSED_CSV_NAME = "processed.csv"
FT_MODEL_OUTPUT_CSV_NAME = "output.csv"
DEFAULT_MODEL_OUTPUT_CSV_NAME = "output_default.csv"
COLUMN_NAMES = ["Title", "User Selection", "AI Selection"]

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = st.secrets['openai_api']['API_KEY']
PROMPT_TEMPLATE = st.secrets['openai_api']['TITLE_CLASSIFICATION_INSTRUCTION']
FT_MODEL = "ft:gpt-4o-mini-2024-07-18:utilizeai:title-classification-balanced:Bmcz4fNK"
DEFAULT_MODEL = "gpt-4o-mini"
BATCH_SIZE = 100  # Number of records to process before saving to CSV
CONCURRENT_REQUESTS = 10 # Number of API requests to make concurrently within a batch
MAX_RETRIES = 5   # Maximum number of retries for an API call per request
RETRY_DELAY_SECONDS = 5 # Initial delay for retries (will increase exponentially)