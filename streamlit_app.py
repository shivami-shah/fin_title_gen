import streamlit as st

st.set_page_config(
    page_title="Unified LLM-Powered News Tool",
    page_icon="📰", # Optional: A nice icon for your app
    layout="wide"
)

st.title("Welcome to the Unified LLM-Powered News Tool")
st.write(
    "Use the sidebar navigation to switch between the **News Classifier** and the "
    "**News Title Generator**. Data may be lost if switched between these tools."
)
st.markdown("---") # Add a separator for better visual grouping
