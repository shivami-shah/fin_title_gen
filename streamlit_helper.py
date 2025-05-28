from typing import List
import streamlit as st
from helper import generate_titles, generate_titles_with_rogue_scores, save_to_database, get_content_from_url

def handle_generate_button_click(input_content: str, input_title: str, generate_type: str, api_key: str, selected_model: str):
    # Clear previous results and states
    st.session_state.generated_titles = []
    st.session_state.titles_generated_flag = False
    st.session_state.selected_title = None
    st.session_state.editable_selected_title = ""

    # Set the current tab type
    st.session_state.current_tab_type = generate_type

    if generate_type == "summary_only":
        if not input_content:
            st.warning("Please enter some content in the summary textbox to generate titles.")
            return
        with st.spinner("Generating titles from summary..."):
            st.session_state.generated_titles = generate_titles(
                summary=input_content,
                model=selected_model, # Use the passed selected_model
                api_key=api_key,
                is_url_content=False
            )
            st.session_state.titles_generated_flag = True
    elif generate_type == "summary_rouge":
        if not input_content:
            st.warning("Please enter some content in the summary textbox to generate titles.")
            return
        if not input_title:
            st.warning("Please provide a 'User Title' to generate titles with Rouge Scores.")
            return
        with st.spinner("Generating titles with Rouge Scores from summary..."):
            st.session_state.generated_titles = generate_titles_with_rogue_scores(
                summary=input_content,
                model=selected_model, # Use the passed selected_model
                user_title=input_title,
                api_key=api_key
            )
            st.session_state.titles_generated_flag = True
    elif generate_type == "url":
        if not input_content:
            st.warning("Please enter a URL to generate titles.")
            return
        with st.spinner("Fetching content and generating titles from URL..."):
            # First, fetch content from URL
            url_content = get_content_from_url(input_content)
            if not url_content:
                st.error("Could not retrieve content from the provided URL. Please check the URL or try again.")
                return
            
            # Then generate titles using the extracted content
            st.session_state.generated_titles = generate_titles(
                summary=url_content, # Pass the fetched content as summary
                model=selected_model, # Use the passed selected_model
                api_key=api_key,
                is_url_content=True # Indicate that the content is from a URL
            )
            st.session_state.titles_generated_flag = True

    if st.session_state.generated_titles:
        st.session_state.selected_title = st.session_state.generated_titles[0]
        st.session_state.editable_selected_title = st.session_state.generated_titles[0]
    else:
        st.session_state.selected_title = None
        st.session_state.editable_selected_title = ""

def handle_save_button_click(source_type: str, selected_model_for_save: str):
    if st.session_state.editable_selected_title:
        with st.spinner("Saving selected title..."):
            # Determine which input field was used for the source
            source_value = ""
            if source_type == "summary_only" or source_type == "summary_rouge":
                source_value = st.session_state.input_summary_text_area if 'input_summary_text_area' in st.session_state else ""
            elif source_type == "url":
                source_value = st.session_state.input_url_text_input if 'input_url_text_input' in st.session_state else ""

            data = [
                source_value, # This will be either summary or URL
                st.session_state.input_title_text_input if st.session_state.input_title_text_input and source_type == "summary_rouge" else "",
                selected_model_for_save, # Use the passed model
                st.session_state.editable_selected_title
            ]
            status = save_to_database(data=data)
            if status:
                st.success(f"Successfully saved: '{st.session_state.editable_selected_title}'")
                # Clear generated titles and inputs after saving
                st.session_state.generated_titles = []
                st.session_state.titles_generated_flag = False
                st.session_state.selected_title = None
                st.session_state.editable_selected_title = ""
                
                # Clear the specific input fields used for the current session
                if 'input_summary_text_area' in st.session_state:
                    st.session_state.input_summary_text_area = ""
                if 'input_title_text_input' in st.session_state:
                    st.session_state.input_title_text_input = ""
                if 'input_url_text_input' in st.session_state:
                    st.session_state.input_url_text_input = ""
            else:
                st.error("Failed to save the selected title. Please try again.")
    else:
        st.warning("No title selected to save.")