from typing import List
from rouge_score import rouge_scorer
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI
from google import genai

# Google Sheets API setup
import streamlit as st
scope = ["https://spreadsheets.google.com/feeds", 
         "https://www.googleapis.com/auth/drive",
        ]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["google_service_account"], scope
)
client = gspread.authorize(credentials)

spreadsheet = client.open_by_key(st.secrets["google_service_account"]["spreadsheet_id"])
worksheet = spreadsheet.worksheet("Titles")

# Model names
OPENAI_MODEL = "chatgpt-4o-latest"
OPENAI_FT_MODEL = "ft:gpt-40-2024-08-06:utilizeai:title-generation-v1:BdYmZT3t"
GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"
PERPLEXITY_MODEL = "sonar-pro"
url_instruction = "The summary is in the following URL. Please extract the content first."


def get_content_from_url(url: str) -> str:
    """
    Placeholder function to fetch content from a given URL.
    In a real application, this would involve web scraping or an API call.
    
    Args:
        url (str): The URL to fetch content from.
        
    Returns:
        str: The extracted content from the URL.
    """
    print(f"Fetching content from URL: {url}")
    # TODO: Implement actual URL content extraction (e.g., using requests and BeautifulSoup)
    return "This is a placeholder for content extracted from the URL."


def make_prompt_for_llm(instruction: str, text_content: str, is_url_content: bool = False) -> str:
    """
    Create a prompt for the LLM based on the provided summary or URL content.
    
    Args:
        text_content (str): The summary or content extracted from URL to include in the prompt.
        is_url_content (bool): True if text_content is from a URL, False if it's a summary.
        
    Returns:
        str: The generated prompt.
    """
    if is_url_content:
        instruction += url_instruction + "URL: "
    prompt = instruction + text_content + "Summary: "
    return prompt    


def get_llm_response(prompt: str, model: str, api_key: str) -> str:
    """
    Get a response from the LLM (Language Model) based on the provided prompt.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The model to use for generating the response.
        
    Returns:
        str: The response from the LLM.
    """
    if model == OPENAI_MODEL: # Using OpenAI's API for gpt-4o model
        try:
            client = OpenAI(api_key=api_key)
            response_stream = client.chat.completions.create(
                model=model,
                stream=True,
                messages=[{"role": "user", "content": prompt}]
            )
            response = ""
            for chunk in response_stream:
                response += chunk.choices[0].delta.content or ""
            return response
        except Exception as e:
            return
    
    if model == GEMINI_MODEL: # Using Gemini's API for gemini-2.0-flash model
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt)
            return response.text
        except Exception as e:
            return
    
    if model == PERPLEXITY_MODEL: 
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            response = client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": ("")},
                    {"role": "user", "content": (prompt)},
                    ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return


def clean_response(response: str, model: str) -> List[str]:
    """
    Clean the response from the LLM to extract the relevant information.
    
    Args:
        response (str): The response from the LLM.
        
    Returns:
        List[str]: A list of cleaned titles.
    """
    if model == OPENAI_MODEL or model == GEMINI_MODEL or model == PERPLEXITY_MODEL:
        lines = response.strip().split('\n')
        lines = [line for line in lines if line.strip()]
        generated_titles = []
        
        if model == GEMINI_MODEL:
            lines = lines[1:]
        
        for line in lines:
            line = line.strip()        
            if line[0].isdigit():
                line = line[1:].strip()
            if line[0] == '.':
                line = line[1:].strip()
            if line.startswith("Here are "):
                continue
            if line.startswith("**"):
                line = line.split("**")[-1].strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].strip()
            if line.startswith("'") and line.endswith("'"):
                line = line[1:-1].strip()
            generated_titles.append(line)        
        
        if len(generated_titles) == 10 and model == GEMINI_MODEL:
            generated_titles = [item for index, item in enumerate(generated_titles) if index % 2 != 0]
            
        return generated_titles
    
    return []


def calculate_rouge_scores(title: str, generated_title: str) -> float:
    """
    Calculate the ROUGE score between the original title and the generated title.
    
    Args:
        title (str): The title given by the user.
        generated_title (str): The title generated by the model.
        
    Returns:
        float: The ROUGE-1 F-measure score.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(title, generated_title)
    score = round(scores['rouge1'].fmeasure, 3)
    return (score)


def generate_titles(instruction: str, summary: str, user_title: str, model: str, api_key:str, is_url_content: bool = False, is_rouge: bool = False) -> List[str]:
    """
    Generate titles based on the provided summary/URL content using the specified model.
    
    Args:
        summary (str): The summary text or URL content to generate titles for.
        model (str): The model to use for title generation.
        is_url_content (bool): True if summary is from a URL, False if it's a direct summary.
        
    Returns:
        List[str]: A list of generated titles.
    """
    prompt = make_prompt_for_llm(text_content=summary, is_url_content=is_url_content, instruction=instruction)
    response = get_llm_response(prompt=prompt, model=model, api_key=api_key)
    
    if not response:
        return [], []
    
    generated_titles = clean_response(response=response, model=model)
    
    rouge_scores = []
    if is_rouge:
        for generated_title in generated_titles:
            rouge_score = calculate_rouge_scores(title=user_title, generated_title=generated_title)
            rouge_scores.append(rouge_score)
            
    return generated_titles, rouge_scores


def save_to_database(data: List) -> bool:
    """
    Save the generated titles and their ROUGE scores to a database.
    
    Args:
        data (List[Any]): The data to save to the database.
                Expected format: [model, summary, generated title, user title (optional), rouge_score (optional)]
        
    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        df = read_from_database()
        if not df.empty:
            df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
            new_id = df["ID"].max() + 1
        else:
            new_id = 1
            
        selected_title = data[3] if len(data) > 3 else ""
        if " \t- ROUGE Score: " in selected_title:
            title, score = selected_title.split(" \t- ROUGE Score: ", 1)
            data[3] = title.strip()
            data.append(score.strip())
        else:
            data.append("")            
            
        data.insert(0, new_id)
        worksheet.append_row(data)
        return True
    except Exception as e:
        return False


def read_from_database() -> pd.DataFrame:
    """
    Read data from the database.
    
    Returns:
        pd.DataFrame: The data read from the database.
    """
    df = get_as_dataframe(worksheet, evaluate_formulas=True).dropna(how="all")
    return df