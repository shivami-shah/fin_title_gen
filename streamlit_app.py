from typing import List
import streamlit as st
from helper import read_from_database
from streamlit_helper import handle_generate_button_click, handle_save_button_click


# --- Streamlit App ---
open_ai_api_key = st.secrets['OPENAI_API_KEY']
st.set_page_config(page_title="LLM Title Generator POC", layout="centered")
st.title("LLM-Powered Title Generator")

# --- Initialize session state variables ---
if 'generated_titles' not in st.session_state:
    st.session_state.generated_titles = []
if 'titles_generated_flag' not in st.session_state:
    st.session_state.titles_generated_flag = False
if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None
if 'editable_selected_title' not in st.session_state:
    st.session_state.editable_selected_title = ""
if 'input_summary_text_area' not in st.session_state:
    st.session_state.input_summary_text_area = ""
if 'input_title_text_input' not in st.session_state:
    st.session_state.input_title_text_input = ""
if 'input_url_text_input' not in st.session_state:
    st.session_state.input_url_text_input = ""
if 'current_tab_type' not in st.session_state:
    st.session_state.current_tab_type = "summary_only"

tab1, tab2, tab3, tab4 = st.tabs(
    ["From Summary", "From Summary With Rouge Score", "From URL", "History"]
)

with tab1: # Generate Title From Summary
    st.header("Generate Titles From Summary")
    st.markdown("Enter your summary below and let the LLM suggest titles!")

    st.subheader("Model Selection", divider="red")
    model_options = ["Model 1", "Model 2", "Model 3"]
    selected_model_tab1 = st.selectbox(
        "Choose a model for title generation:",
        options=model_options,
        index=0,
        key="model_selector_tab1"
    )

    st.subheader("Summary", divider="red")
    input_summary_tab1 = st.text_area(
        "Paste your summary here:",
        height=150,
        placeholder="Enter your summary here...",
        key="input_summary_text_area"
    )

    st.subheader("Generate Titles", divider="red")
    if st.button(
        "Generate Titles",
        type="primary",
        on_click=handle_generate_button_click,
        args=(input_summary_tab1, "", "summary_only", open_ai_api_key, selected_model_tab1,),
        key="generate_button_tab1"
    ):
        pass

    # Common display logic after generation
    if st.session_state.titles_generated_flag and st.session_state.current_tab_type == "summary_only":
        if st.session_state.generated_titles:
            st.markdown("---")
            st.subheader("Generated Titles:", divider="red")
            
            try:
                default_index = st.session_state.generated_titles.index(st.session_state.selected_title)
            except ValueError:
                default_index = 0

            st.session_state.selected_title = st.radio(
                "Select your preferred title:",
                options=st.session_state.generated_titles,
                index=default_index,
                key="title_selector_tab1"
            )
            st.session_state.editable_selected_title = st.session_state.selected_title

            st.markdown("---")
            st.subheader("Edit Selected Title (Optional):")
            st.session_state.editable_selected_title = st.text_input(
                "Make any desired edits to the selected title:",
                value=st.session_state.editable_selected_title,
                key="editable_title_input_tab1"
            )
            
            if st.session_state.selected_title:
                st.button(
                    "Save Selected Title",
                    type="secondary",
                    on_click=handle_save_button_click,
                    args=("summary_only", selected_model_tab1,),
                    key="save_button_tab1"
                )
        else:
            st.warning("No titles were generated for your input. Please try again.")

with tab2:
    st.header("Generate Titles From Summary (Rouge Score)")
    st.markdown("Enter your summary and a user title below. The LLM will suggest titles, and their ROUGE scores against your user title will be calculated.")

    st.subheader("Model Selection", divider="red")
    selected_model_tab2 = st.selectbox(
        "Choose a model for title generation:",
        options=model_options,
        index=0,
        key="model_selector_tab2"
    )

    st.subheader("Summary", divider="red")
    input_summary_tab2 = st.text_area(
        "Paste your summary here:",
        height=150,
        placeholder="Enter your summary here...",
        key="input_summary_text_area_tab2"
    )

    st.subheader("User Title (Mandatory)", divider="red")
    input_title_tab2 = st.text_input(
        "Enter a title if rouge scores are to be generated:",
        placeholder="Enter your title here...",
        key="input_title_text_input"
    )

    st.subheader("Generate Titles", divider="red")
    if st.button(
        "Generate Titles with Rouge Scores",
        type="primary",
        on_click=handle_generate_button_click,
        args=(input_summary_tab2, input_title_tab2, "summary_rouge", open_ai_api_key, selected_model_tab2,),
        key="generate_button_tab2"
    ):
        pass

    if st.session_state.titles_generated_flag and st.session_state.current_tab_type == "summary_rouge":
        if st.session_state.generated_titles:
            st.markdown("---")
            st.subheader("Generated Titles:", divider="red")
            
            try:
                default_index = st.session_state.generated_titles.index(st.session_state.selected_title)
            except ValueError:
                default_index = 0

            st.session_state.selected_title = st.radio(
                "Select your preferred title:",
                options=st.session_state.generated_titles,
                index=default_index,
                key="title_selector_tab2"
            )
            st.session_state.editable_selected_title = st.session_state.selected_title


            st.markdown("---")
            st.subheader("Edit Selected Title (Optional):")
            st.session_state.editable_selected_title = st.text_input(
                "Make any desired edits to the selected title:",
                value=st.session_state.editable_selected_title,
                key="editable_title_input_tab2"
            )
            
            if st.session_state.selected_title:
                st.button(
                    "Save Selected Title",
                    type="secondary",
                    on_click=handle_save_button_click,
                    args=("summary_rouge", selected_model_tab2,),
                    key="save_button_tab2"
                )
        else:
            st.warning("No titles were generated for your input. Please try again.")

with tab3:
    st.header("Generate Titles From URL")
    st.markdown("Enter a URL below to extract content and generate titles!")

    st.subheader("Model Selection", divider="red")
    selected_model_tab3 = st.selectbox(
        "Choose a model for title generation:",
        options=model_options,
        index=0,
        key="model_selector_tab3"
    )

    st.subheader("URL Input", divider="red")
    input_url_tab3 = st.text_input(
        "Enter the URL here:",
        placeholder="e.g., https://example.com/article",
        key="input_url_text_input"
    )

    st.subheader("Generate Titles", divider="red")
    if st.button(
        "Generate Titles From URL",
        type="primary",
        on_click=handle_generate_button_click,
        args=(input_url_tab3, "", "url", open_ai_api_key, selected_model_tab3,),
        key="generate_button_tab3"
    ):
        pass

    if st.session_state.titles_generated_flag and st.session_state.current_tab_type == "url":
        if st.session_state.generated_titles:
            st.markdown("---")
            st.subheader("Generated Titles:", divider="red")
            
            try:
                default_index = st.session_state.generated_titles.index(st.session_state.selected_title)
            except ValueError:
                default_index = 0

            st.session_state.selected_title = st.radio(
                "Select your preferred title:",
                options=st.session_state.generated_titles,
                index=default_index,
                key="title_selector_tab3"
            )
            st.session_state.editable_selected_title = st.session_state.selected_title


            st.markdown("---")
            st.subheader("Edit Selected Title (Optional):")
            st.session_state.editable_selected_title = st.text_input(
                "Make any desired edits to the selected title:",
                value=st.session_state.editable_selected_title,
                key="editable_title_input_tab3"
            )
            
            if st.session_state.selected_title:
                st.button(
                    "Save Selected Title",
                    type="secondary",
                    on_click=handle_save_button_click,
                    args=("url", selected_model_tab3,),
                    key="save_button_tab3"
                )
        else:
            st.warning("No titles were generated for your input. Please try again.")


with tab4:
    st.header("Generated Titles")
    
    data = read_from_database()
    if data.empty:
        st.warning("No titles found in the database.")
    else:
        st.subheader("Titles from Database:")
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.markdown("---")