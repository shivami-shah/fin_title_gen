from typing import List
import streamlit as st
from helper import generate_titles, save_to_database, get_content_from_url, read_from_database

open_ai_api_key = st.secrets['OPENAI_API_KEY']
gemini_api_key = st.secrets['GEMINI_API_KEY']
instruction = st.secrets['INSTRUCTION']

def handle_generate_button_click():
    model = st.session_state.model_selector
    is_url = True if st.session_state.selected_input_source == "url" else False
    is_rouge = True if st.session_state.selected_scoring_method == "rouge" else False
    input_content = st.session_state.input_url_text_input if is_url else st.session_state.input_summary_text_area
    user_title = st.session_state.input_title_text_area if is_rouge else ""
    if model == "Open AI":
        api_key = open_ai_api_key
    elif model == "Gemini":
        api_key = gemini_api_key
    else:
        api_key = None
    
    if not input_content:
        st.warning("Please enter some content to generate titles.")
        return
    if is_rouge and not user_title:
        st.warning("Please provide a 'User Title' to generate titles with Rouge Scores.")
        return
    
    with st.spinner("Generating titles..."):
        st.session_state.generated_flag = True
        models = {"Open AI": "gpt-4o", "Gemini": "gemini-1.0", "Model 3": "model-3"}
        if model in models:
            model = models[model]
        else:
            st.error("Unsupported model selected.")
            return                       
        st.session_state.generated_titles, st.session_state.rouge_scores = generate_titles(
                summary=input_content,
                user_title=user_title,
                model=model,
                api_key=api_key,
                is_url_content=is_url,
                is_rouge=is_rouge,
                instruction=instruction
            )
        if not st.session_state.generated_titles:
            st.error("Failed to connect to the model. Please try again later.")
        

def handle_save_button_click():    
    with st.spinner("Saving selected title..."):
        data_to_save = [
            st.session_state.model_selector,
            st.session_state.input_summary_text_area if st.session_state.selected_input_source == "summary" else "",
            st.session_state.input_url_text_input if st.session_state.selected_input_source == "url" else "",
            st.session_state.editable_selected_title,
            st.session_state.input_title_text_area if st.session_state.selected_scoring_method == "rouge" else "",
            st.session_state.rouge_scores[st.session_state.selected_title_index] if st.session_state.selected_scoring_method == "rouge" else ""
        ]
        success = save_to_database(data=data_to_save)
        if success:
            st.success("Title saved successfully!")
            for key in ["input_summary_text_area", "input_url_text_input", 
                         "input_title_text_area", "generated_titles", 
                         "rouge_scores", "selected_title_index", 
                         "editable_selected_title", "generated_flag"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.session_state.input_summary_text_area = ""
            st.session_state.input_url_text_input = ""
            st.session_state.input_title_text_area = ""
        else:
            st.error("Failed to save the title. Try again later.")
            
            
def read_from_db():
    with st.spinner("Loading history from database..."):
        df = read_from_database()
        if df.empty:
            st.warning("No history found in the database.")
        else:
            st.subheader("Titles from Database:")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("---")