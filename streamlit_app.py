import streamlit as st
import pandas as pd
import io
import os 
from streamlit_helper import save_uploaded_file_and_extract, run_classification_and_load_output, to_excel_bytes

st.set_page_config(layout="wide", page_title="Classifier App")

# Initialize session state for data persistence
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = pd.DataFrame() # DataFrame after classification
if 'edited_df' not in st.session_state:
    st.session_state['edited_df'] = pd.DataFrame() # DataFrame after user edits
if 'extracted_file_name' not in st.session_state:
    st.session_state['extracted_file_name'] = None # To track which Excel file was last extracted to CSV
if 'is_test_selected' not in st.session_state:
    st.session_state['is_test_selected'] = False # To track the state of 'generate metrics report'

st.title("AI News Curation")

# --- 1. User uploads an Excel file and triggers extraction ---
st.header("1. Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"], key="excel_uploader")

if uploaded_file is not None:
    # st.info(f"File selected: **{uploaded_file.name}**")
    
    # Check if the currently uploaded file is different from the one last extracted
    if st.session_state['extracted_file_name'] != uploaded_file.name:
        if st.button(f"Extract '{uploaded_file.name}'", key="extract_button_new"):
            with st.spinner(f"Extracting titles from {uploaded_file.name}..."):
                extraction_success = save_uploaded_file_and_extract(uploaded_file)
                if extraction_success:
                    st.session_state['extracted_file_name'] = uploaded_file.name
                    # Reset processed_df and edited_df if a new file is extracted
                    st.session_state['processed_df'] = pd.DataFrame() 
                    st.session_state['edited_df'] = pd.DataFrame()
                    st.success("You can now classify the titles.")
                else:
                    st.error("Data extraction failed.")
    else:
        st.info(f"'{uploaded_file.name}' is ready for classification.")
        if st.button(f"Re-extract '{uploaded_file.name}'", key="extract_button_re"):
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
# Only show this section if an excel file has been successfully extracted to processed CSV
if st.session_state['extracted_file_name'] is not None:
    st.header("2. Classify Titles")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.radio(
            "Select Model Type:",
            ("Base Model", "Fine-tuned Model"),
            key="model_selection",
            index=0, # Default to Default Model
            help="Choose 'Base Model' or 'Fine-tuned Model'"
        )
    with col2:
        generate_report = st.checkbox(
            "Evaluate",
            value=st.session_state['is_test_selected'], # Set initial value from session state
            key="generate_report_checkbox",
            help="Check this if the uploaded file is a test file and you want a metrics report."
        )

    is_default_model = (model_choice == "Default Model (output_default.csv)")
    # Store the current state of generate_report checkbox in session_state
    st.session_state['is_test_selected'] = generate_report 

    if st.button("Classify", key="process_button"):
        st.session_state['processed_df'] = run_classification_and_load_output(is_test=st.session_state['is_test_selected'], is_default_model=is_default_model)
        st.session_state['edited_df'] = st.session_state['processed_df'].copy() # Initialize edited_df

# --- 3. Display, Filter, Search, and Edit Data ---
if not st.session_state['edited_df'].empty:
    st.header("3. Review and Edit Classified Titles")

    current_df = st.session_state['edited_df'].copy()

    # Determine which columns to display
    display_columns = ['title', 'model']
    if st.session_state['is_test_selected'] and 'user' in current_df.columns:
        display_columns.insert(1, 'user') # Insert 'user' after 'title'

    # Filter current_df to only include display_columns
    current_df_for_display = current_df[display_columns]

    # --- Search Bar ---
    search_query = st.text_input("Search across all columns (visible):", "", key="search_input")
    if search_query:
        # Filter rows where any displayed column contains the search query (case-insensitive)
        current_df_for_display = current_df_for_display[
            current_df_for_display.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)
        ]

    # --- Dynamic Filters ---
    st.subheader("Column Filters")
    
    # Define columns for which filters should be shown
    filter_cols = ['model']
    if st.session_state['is_test_selected'] and 'user' in current_df.columns:
        filter_cols.append('user')
    
    for col in filter_cols:
        if col in current_df_for_display.columns: # Ensure the column exists in the displayed data
            if current_df_for_display[col].dtype == 'object' or current_df_for_display[col].dtype == 'category':
                unique_values = current_df_for_display[col].dropna().unique().tolist()
                if unique_values: # Only show filter if there are unique values
                    selected_options = st.multiselect(
                        f"Filter by {col}", 
                        sorted(unique_values), 
                        default=sorted(unique_values),
                        key=f"filter_{col}"
                    )
                    current_df_for_display = current_df_for_display[current_df_for_display[col].isin(selected_options)]
            elif pd.api.types.is_numeric_dtype(current_df_for_display[col]):
                numeric_series = current_df_for_display[col].dropna()
                if not numeric_series.empty:
                    col_min, col_max = float(numeric_series.min()), float(numeric_series.max())
                    if col_min == col_max:
                        st.write(f"Column '{col}' has a single value: {col_min}")
                    else:
                        selected_range = st.slider(
                            f"Filter by {col} range", 
                            col_min, 
                            col_max, 
                            (col_min, col_max),
                            key=f"range_filter_{col}"
                        )
                        current_df_for_display = current_df_for_display[(current_df_for_display[col] >= selected_range[0]) & (current_df_for_display[col] <= selected_range[1])]

    st.subheader("Editable Data Table")
    # Display the filtered data in an editable format
    edited_data = st.data_editor(
        current_df_for_display, # Pass the column-filtered DataFrame here
        num_rows="dynamic",  # Allows adding/deleting rows
        use_container_width=True,
        key="data_editor_main" # A unique key for this widget
    )
    # Update the session state with the latest edited data
    # IMPORTANT: Ensure that edits to `current_df_for_display` correctly update `st.session_state['edited_df']`
    # This part can be tricky if columns are dropped. A safer approach might be to allow `data_editor` to
    # operate on the full `edited_df` and then project columns for display *only*.
    # However, for simplicity and direct control over what's edited visually, we'll keep it this way.
    # If a user adds a row, it will only have the displayed columns.
    st.session_state['edited_df'] = edited_data

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