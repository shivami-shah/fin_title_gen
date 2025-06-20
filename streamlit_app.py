from typing import List
import streamlit as st
import streamlit_helper as sh

# --- Streamlit App ---
st.set_page_config(page_title="LLM Title Generator POC", layout="centered")
st.title("LLM-Powered Title Generator")


def on_model_selection_change():
    if st.session_state.model_selector == "Open AI - Finnovate Research":
        st.session_state.fine_tuned_model = True
        st.session_state.selected_input_source = "summary"
    else:
        st.session_state.fine_tuned_model = False


def start_page():
    if 'selected_input_source' not in st.session_state:
        st.session_state.selected_input_source = "summary"
    if 'selected_scoring_method' not in st.session_state:
        st.session_state.selected_scoring_method = "no_rouge"
    if 'generated_flag' not in st.session_state:
        st.session_state.generated_flag = False
    if 'fine_tuned_model' not in st.session_state:
        st.session_state.fine_tuned_model = False
        
    input_options = ["summary", "url"]
    scoring_methods = ["no_rouge", "rouge"]

    st.subheader("Inputs For Title Generation", divider="red")
    
    model_options = ["Open AI", "Gemini", "Perplexity", "Open AI - Finnovate Research"]
    selected_model = st.selectbox(
        "Choose a model for title generation:",
        options=model_options,
        index=0,
        key="model_selector",
        on_change=on_model_selection_change,
        disabled=st.session_state.generated_flag,
    )
       
    st.radio(
        "Select the input source for title generation:",
        options=input_options,
        index=0,
        format_func=lambda x: "From Summary" if x == "summary" else "From URL",
        key="selected_input_source",
        horizontal=True,
        disabled=st.session_state.generated_flag or st.session_state.fine_tuned_model,
    )
    
    st.radio(
        "Select the scoring method for title generation:",
        options=["no_rouge", "rouge"],
        index=0,
        format_func=lambda x: "No Rouge Score" if x == "no_rouge" else "With Rouge Score",
        key="selected_scoring_method",
        horizontal=True,
        disabled=st.session_state.generated_flag,
    )


def handle_input_selection():   
    
    if st.session_state.selected_input_source == "summary":
        st.text_area("Enter Summary:", key="input_summary_text_area")
    elif st.session_state.selected_input_source == "url":
        st.text_input("Enter URL:", key="input_url_text_input")
        
    if st.session_state.selected_scoring_method == "rouge":
        st.text_area("Enter User Title (for Rouge Scoring):", key="input_title_text_area")
        
    if st.session_state.model_selector == "Open AI - Finnovate Research":
        st.markdown(
            "Note: The 'Open AI - Finnovate Research' model will generate only one title based on the summary. URL content is not supported for this model.",
        )
        
    st.button(
        "Generate Titles",
        type="primary",
        on_click=sh.handle_generate_button_click,
        disabled=st.session_state.generated_flag,
    )
    
    if st.session_state.generated_flag:
        st.button(
            "Reset",
            type="secondary",
            on_click=lambda: st.session_state.clear(),
        )
    

def display_generated_titles():
    if 'generated_titles' in st.session_state and st.session_state.generated_titles:
        st.markdown("---")
        st.subheader("Generated Titles:", divider="red")
        
        options = []        
        if st.session_state.rouge_scores != []:
            options = [
                f"{title} (Rouge Score: {rouge_score})"
                for title, rouge_score in zip(st.session_state.generated_titles, st.session_state.rouge_scores)
            ]
        else:
            options = [title for title in st.session_state.generated_titles]
            
        st.session_state.selected_title_index = st.radio(
            "Select your preferred title:",
            options=list(range(len(options))),
            format_func=lambda i: options[i] ,
            index=0,
        )
        
        
def save_editable_title():
    if 'selected_title_index' in st.session_state:
        st.markdown("---")
        st.subheader("Edit Selected Title (Optional):")
        st.session_state.editable_selected_title = st.text_area(
            "Make any desired edits to the selected title:",
            value=st.session_state.generated_titles[st.session_state.selected_title_index],
            key="editable_title_input"
        )
        
        st.button(
            "Save Selected Title",
            type="primary",
            on_click=sh.handle_save_button_click,
        )


def main():
    tab1, tab2 = st.tabs(["Title Generation", "Generated Titles"])
    
    with tab1:
        start_page()
        handle_input_selection()
        display_generated_titles()
        save_editable_title()
        
    with tab2:
        st.header("Generated Titles")    
        sh.read_from_db()

        
if __name__ == "__main__":
    main()