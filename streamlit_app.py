from typing import List
import streamlit as st
from helper import generate_titles, generate_titles_with_rogue_scores, save_to_database, read_from_database


def handle_generate_button_click(rougue_scores: bool, api_key: str):
    if not input_summary:
        st.warning("Please enter some content in the summary textbox to generate titles.")
        st.session_state.generated_titles = []
        st.session_state.titles_generated_flag = False
        st.session_state.selected_title = None # Clear selected title
        return
    if not rougue_scores:
        with st.spinner("Generating titles..."):
            st.session_state.generated_titles = generate_titles(summary=input_summary, model=st.session_state.model_selector, api_key=api_key)
            st.session_state.titles_generated_flag = True
            if st.session_state.generated_titles:
                st.session_state.selected_title = st.session_state.generated_titles[0]
            else:
                st.session_state.selected_title = None
    else:
        if not input_title:
            st.warning("Please provide a 'User Title' to generate titles with Rouge Scores.")
            st.session_state.generated_titles = []
            st.session_state.titles_generated_flag = False
            st.session_state.selected_title = None
            return
        with st.spinner("Generating titles with Rouge Scores..."):
            st.session_state.generated_titles = generate_titles_with_rogue_scores(summary=input_summary, model=st.session_state.model_selector, user_title=input_title, api_key=api_key) # Use session_state for selected_model
            st.session_state.titles_generated_flag = True
            if st.session_state.generated_titles:
                st.session_state.selected_title = st.session_state.generated_titles[0]
            else:
                st.session_state.selected_title = None

def handle_save_button_click():
    if st.session_state.selected_title:
        with st.spinner("Saving selected title..."):
            data = [st.session_state.input_summary_text_area,
                    st.session_state.input_title_text_input if st.session_state.input_title_text_input else "",
                    st.session_state.model_selector,
                    st.session_state.selected_title
                    ]
            status = save_to_database(data=data)
            if status:
                st.success(f"Successfully saved: '{st.session_state.selected_title}'")
                # Clear generated titles after saving
                st.session_state.generated_titles = []
                st.session_state.titles_generated_flag = False
                st.session_state.selected_title = None
                st.session_state.input_summary_text_area = ""
                st.session_state.input_title_text_input = ""
            else:
                st.error("Failed to save the selected title. Please try again.")
    else:
        st.warning("No title selected to save.")
            
            
# --- Streamlit App ---
open_ai_api_key = st.secrets['OPENAI_API_KEY']
st.set_page_config(page_title="LLM Title Generator POC", layout="centered")
st.title("LLM-Powered Title Generator")
tab1, tab2 = st. tabs(["Generate New Title", "Generated Titles"])

# --- Initialize session state variables ---
# These variables will persist across reruns
if 'generated_titles' not in st.session_state:
    st.session_state.generated_titles = []
if 'titles_generated_flag' not in st.session_state:
    st.session_state.titles_generated_flag = False
if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = tab1

with tab1:
    st.header("Generate New Titles")
    st.markdown("Enter your summary and title (optional) below, choose a model, and let the LLM suggest titles!")

    # 1. Dropdown List
    st.subheader("Model Selection", divider="red")
    model_options = ["Model 1", "Model 2", "Model 3"]
    selected_model = st.selectbox(
        "Choose a model for title generation:",
        options=model_options,
        index=0,
        key="model_selector"
    )


    # 2. Input Textbox - Summary
    st.subheader("Summary", divider="red")
    input_summary = st.text_area(
        "Paste your summary here:",
        height=150,
        placeholder="Enter your summary here...",
        key="input_summary_text_area"
    )

    # 3. Input Textbox - User Title (Optional)
    st.subheader("User Title (Optional)", divider="red")
    input_title = st.text_input(
        "Enter a title if rouge scores are to be generated:",
        placeholder="Enter your title here...",
        key="input_title_text_input"
    )

    # --- Determine Button States ---
    disable_button1 = bool(input_title)
    disable_button2 = not bool(input_title)

    # 3. Buttons
    st.subheader("Generate Titles", divider="red")

    # --- Create Two Columns for Side-by-Side Buttons ---
    col1, col2 = st.columns(2)            
            
    with col1:
        st.button(
            "Generate Titles",
            type="primary",
            disabled=disable_button1,
            help="This button is enabled when the 'User Title' is empty.",
            on_click=handle_generate_button_click,
            args=(False,open_ai_api_key,)
        )

    with col2:
        st.button(
            "Generate Titles with Rouge Scores",
            type="primary",
            disabled=disable_button2,
            help="This button is enabled when a 'User Title' is provided.",
            on_click=handle_generate_button_click,
            args=(True,open_ai_api_key,)
        )

    # The condition now checks the flag in session_state
    if st.session_state.titles_generated_flag:
        if st.session_state.generated_titles:
            st.markdown("---")
            st.subheader("Generated Titles:", divider="red")
            
            # Find the index of the selected_title in the generated_titles list
            try:
                default_index = st.session_state.generated_titles.index(st.session_state.selected_title)
            except ValueError:
                default_index = 0

            # Update st.session_state.selected_title when a radio button is chosen
            st.session_state.selected_title = st.radio(
                "Select your preferred title:",
                options=st.session_state.generated_titles,
                index=default_index,
                key="title_selector"
            )

            st.markdown("---")
            st.subheader("Selected Title:")
            st.success(st.session_state.selected_title)
            
            # --- Save Button (appears only if a title is selected) ---
            if st.session_state.selected_title:
                st.button(
                    "Save Selected Title",
                    type="secondary",
                    on_click=handle_save_button_click,
                    help="Saves the selected title and summary."
                )
        else:
            st.warning("No titles were generated for your input. Please try again.")
        
        
with tab2:
    st.header("Generated Titles")
    
    data = read_from_database()
    if data.empty:
        st.warning("No titles found in the database.")
    else:
        st.subheader("Titles from Database:")
        st.dataframe(data, use_container_width=True)
        st.markdown("---")