import streamlit as st

st.set_page_config(
    page_title="Unified LLM-Powered News Tool",
    page_icon="📰", # Optional: A nice icon for your app
    layout="wide"
)

st.title("Finnovate Research AI Systems")

st.markdown("---") # Add a separator for better visual grouping
st.write(
    "Use the sidebar navigation to switch between **Title Generation** system and the "
    "**Title Curation** system."
)
