import streamlit as st
import pandas as pd
import os
import shutil
import asyncio
import io

# Assuming these imports are available in the environment
from classifier_app import extract_data, process_data
from classifier_config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_CSV_NAME, 
    MODEL_OUTPUT_DIR, FT_MODEL_OUTPUT_CSV_NAME, DEFAULT_MODEL_OUTPUT_CSV_NAME
)
from project_logger import setup_project_logger

# Initialize logger for the helper
logger = setup_project_logger("streamlit_helper")

def save_uploaded_file_and_extract(uploaded_file):
    """
    Saves the uploaded Streamlit file to RAW_DATA_DIR and then
    triggers the data extraction process.
    """
    if uploaded_file is not None:
        # Create RAW_DATA_DIR if it doesn't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        for f in os.listdir(RAW_DATA_DIR):
            os.remove(os.path.join(RAW_DATA_DIR, f))
        logger.info("Cleared existing raw files.")
        
        # Define the path where the file will be saved
        file_path = os.path.join(RAW_DATA_DIR, uploaded_file.name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file to {file_path}")
        
        # Trigger the extraction process
        try:
            # # Clear existing processed files to ensure fresh processing
            # for f in os.listdir(PROCESSED_DATA_DIR):
            #     os.remove(os.path.join(PROCESSED_DATA_DIR, f))
            # logger.info("Cleared existing processed data files.")

            extract_data() # This will process all excel files in RAW_DATA_DIR
            # st.success("Excel data extracted and processed into CSV.")
            return True
        except Exception as e:
            st.error(f"Error during data extraction: {e}")
            logger.exception("Error during data extraction")
            return False
    return False

def run_classification_and_load_output(is_test, is_default_model):
    """
    Runs the data classification process and loads the appropriate output CSV.
    """
    try:
        # Ensure model output directory exists and is clear for new outputs
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        # # Optional: Clear previous model outputs if you want only the latest
        # for f in os.listdir(MODEL_OUTPUT_DIR):
        #     os.remove(os.path.join(MODEL_OUTPUT_DIR, f))
        # logger.info("Cleared existing model output files.")

        with st.spinner("Running title classification... This may take a moment."):
            # The process_data function itself uses asyncio.run, so it handles the event loop.
            # Calling asyncio.run multiple times in the same thread can be problematic in some environments.
            # For a long-running production app, consider using Streamlit's @st.cache_resource
            # or a separate thread/process if asyncio conflicts arise.
            process_data(is_test=is_test, is_default_model=is_default_model)
        st.success("Title classification completed!")
        
        output_csv_name = DEFAULT_MODEL_OUTPUT_CSV_NAME if is_default_model else FT_MODEL_OUTPUT_CSV_NAME
        output_file_path = os.path.join(MODEL_OUTPUT_DIR, output_csv_name)
        
        if os.path.exists(output_file_path):
            df_output = pd.read_csv(output_file_path)
            # st.info(f"Loaded classified data from {output_csv_name}.")
            return df_output
        else:
            st.warning(f"Output file '{output_csv_name}' not found after processing. Please check logs for errors.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error during data classification: {e}")
        logger.exception("Error during data classification")
        return pd.DataFrame()

def to_excel_bytes(df):
    """Converts a pandas DataFrame to an in-memory Excel file (bytes)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data