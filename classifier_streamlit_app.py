import streamlit as st
import pandas as pd
import io
import os
from classifier_streamlit_helper import (
    save_uploaded_file_and_extract, run_classification_and_load_output, 
    to_excel_bytes, clear_raw_data_directory)
from classifier_config import COLUMN_NAMES
from streamlit.runtime.scriptrunner.script_runner import RerunException

st.set_page_config(layout="wide", page_title="Classifier App")

# Initialize session state for data persistence
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = pd.DataFrame()
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame()
if 'extracted_file_name' not in st.session_state:
    st.session_state['extracted_file_name'] = None
if 'is_test_selected' not in st.session_state:
    st.session_state['is_test_selected'] = False

st.title("AI News Curation")

# Function to reset the application state
def reset_app_state():
    clear_raw_data_directory()
    st.session_state['processed_df'] = pd.DataFrame()
    st.session_state['edited_df'] = pd.DataFrame()
    st.session_state['extracted_file_name'] = None
    st.session_state['is_test_selected'] = False
    try:
        st.rerun() # Rerun the app to reflect the changes
    except RerunException:
        pass

disable_buttons = not st.session_state['edited_df'].empty

# --- 1. User uploads an Excel file and triggers extraction ---
st.header("1. Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"], key="excel_uploader")

if uploaded_file is not None:
    if st.session_state['extracted_file_name'] != uploaded_file.name:
        if st.button(f"Extract '{uploaded_file.name}'", 
                     key="extract_button_new",
                     disabled=disable_buttons):
            with st.spinner(f"Extracting titles from {uploaded_file.name}..."):
                extraction_success = save_uploaded_file_and_extract(uploaded_file)
                if extraction_success:
                    st.session_state['extracted_file_name'] = uploaded_file.name
                    st.session_state['processed_df'] = pd.DataFrame()
                    st.session_state['edited_df'] = pd.DataFrame()
                    st.success("You can now classify the titles.")
                else:
                    st.error("Data extraction failed.")
    else:
        st.info(f"'{uploaded_file.name}' is ready for classification.")
        if st.button(f"Re-extract '{uploaded_file.name}'",
                     key="extract_button_re",
                     disabled=disable_buttons):
            with st.spinner(f"Re-extracting titles from {uploaded_file.name}..."):
                extraction_success = save_uploaded_file_and_extract(uploaded_file)
                if extraction_success:
                    st.session_state['extracted_file_name'] = uploaded_file.name
                    st.session_state['processed_df'] = pd.DataFrame()
                    st.session_state['edited_df'] = pd.DataFrame()
                    st.success("Data re-extraction complete. You can now configure and process the data.")
                else:
                    st.error("Data re-extraction failed.")

# --- 2. User selects model and report options, then triggers classification ---
if st.session_state['extracted_file_name'] is not None:
    st.header("2. Classify Titles")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.radio(
            "Select Model Type:",
            ("Base Model", "Fine-tuned Model"),
            key="model_selection",
            index=0,
            help="Choose 'Base Model' or 'Fine-tuned Model'",
            disabled=disable_buttons
        )
    with col2:
        generate_report = st.checkbox(
            "Evaluate",
            value=st.session_state['is_test_selected'],
            key="generate_report_checkbox",
            help="Check this if the uploaded file is a test file and you want a metrics report.",
            disabled=disable_buttons
        )

    is_base_model = (model_choice == "Base Model")
    st.session_state['is_test_selected'] = generate_report
    
    if st.button("Classify", key="process_button", disabled=disable_buttons):
        st.session_state['processed_df'] = run_classification_and_load_output(is_test=st.session_state['is_test_selected'], is_base_model=is_base_model)
        st.session_state['edited_df'] = st.session_state['processed_df'].copy()

# --- 3. Display, Filter, Search, and Edit Data ---
if not st.session_state['edited_df'].empty:
    st.header("3. Review and Edit Classified Titles")

    base_df = st.session_state['edited_df']

    st.subheader("Search and Filter Options")

    search_term = st.text_input("Search in Titles:", "")

    user_filter = st.multiselect(
        "Filter by User Selection",
        options=base_df[COLUMN_NAMES[1]].dropna().unique().tolist()
        if COLUMN_NAMES[1] in base_df.columns else [],
        default=[]
    )

    ai_filter = st.multiselect(
        "Filter by AI Selection",
        options=base_df[COLUMN_NAMES[2]].dropna().unique().tolist()
        if COLUMN_NAMES[2] in base_df.columns else [],
        default=[]
    )

    # Filter view only
    view_df = base_df.copy()

    if search_term.strip():
        view_df = view_df[view_df[COLUMN_NAMES[0]].str.contains(search_term, case=False, na=False)]

    if user_filter:
        view_df = view_df[view_df[COLUMN_NAMES[1]].isin(user_filter)]

    if ai_filter:
        view_df = view_df[view_df[COLUMN_NAMES[2]].isin(ai_filter)]

    # Display columns
    display_columns = [COLUMN_NAMES[0], COLUMN_NAMES[2]]
    if st.session_state['is_test_selected'] and COLUMN_NAMES[1] in view_df.columns:
        display_columns.insert(1, COLUMN_NAMES[1])

    st.subheader("Editable Data Table (Filtered View)")

    # Show filtered editable table
    edited_view = st.data_editor(
        view_df[display_columns],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=False,
        key="data_editor_main"
    )

    # Update only the changed cells back into the master dataset
    if not edited_view.equals(view_df[display_columns]):
        for col in display_columns:
            st.session_state['edited_df'].loc[edited_view.index, col] = edited_view[col]

# --- 4. Save Edited Data ---
if not st.session_state['edited_df'].empty:
    st.header("4. Save Edited Data")
    excel_data_bytes = to_excel_bytes(st.session_state['edited_df'])

    st.download_button(
        label="Download Edited Excel File",
        data=excel_data_bytes,
        file_name="classified_and_edited_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Click to download the currently displayed and edited data as an Excel file."
    )

    # --- New: Process Another File Button ---
    st.markdown("---") # Add a separator for better visual grouping
    st.subheader("Process Another File")
    if st.button("Process Another File", key="process_another_button"):
        reset_app_state()
        st.success("Application reset. Please upload a new file.")