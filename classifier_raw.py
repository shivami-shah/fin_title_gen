import asyncio
import os
import csv
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from classifier_config import MAX_RETRIES, RETRY_DELAY_SECONDS, BATCH_SIZE, CONCURRENT_REQUESTS, MODEL_OUTPUT_DIR, OPENAI_API_KEY, PROMPT_TEMPLATE, PROCESSED_DATA_DIR, MODEL
from project_logger import setup_project_logger

logger = setup_project_logger("excel_extraction")
client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=30.0) # Set client-level timeout

def read_csv_file(file_path)-> list:
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    (Kept for compatibility with generate_classification_metrics_report if needed directly)

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries representing the rows in the CSV file.
    """
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        return []
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV file: {e}")
        return []
    
def read_input_csv(file_path):
    """
    Reads a CSV file with headers: source_website, title, url, selected, date.
    Extracts 'title' for model input and 'selected' for actual output.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'title' (for model input) and 'user_output' (actual output).
    """
    extracted_data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Ensure required columns exist
            if not all(col in reader.fieldnames for col in ['title', 'selected']):
                logger.error(f"CSV file '{file_path}' must contain 'title' and 'selected' columns.")
                return []
            for row_num, row in enumerate(reader, 1):
                title = row.get('title', '').strip()
                selected = row.get('selected', '').strip()

                if not title:
                    logger.warning(f"Skipping row {row_num} due to missing 'title'.")
                    continue
                # 'selected' can be empty, we still want to record it as actual output

                extracted_data.append({
                    'title': title,
                    'user_output': selected
                })
        logger.info(f"Successfully extracted {len(extracted_data)} records from '{file_path}'.")
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV file '{file_path}': {e}", exc_info=True)
        return []
    return extracted_data
    
def load_existing_results(output_csv_file):
    """
    Loads already processed results from the output CSV file to enable resumption.

    Args:
        output_csv_file (str): The path to the output CSV file.

    Returns:
        list: A list of dictionaries containing already processed data.
    """
    existing_data = []
    if os.path.exists(output_csv_file):
        try:
            with open(output_csv_file, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Use 'title' to check for existing records
                    if 'title' in row and 'user_output' in row and 'model_output' in row:
                        existing_data.append(row)
                    else:
                        logger.warning(f"Skipping row in existing CSV due to missing 'title', 'user_output', or 'model_output' key: {row}")
            logger.info(f"Loaded {len(existing_data)} existing records from '{output_csv_file}'.")
        except Exception as e:
            logger.error(f"Error loading existing results from '{output_csv_file}': {e}", exc_info=True)
            return []
    return existing_data

def save_processed_data(data, output_file_path):
    """
    Saves the processed data (list of dictionaries) to a CSV file.
    If the file exists, it appends to it; otherwise, it creates a new file.

    Args:
        data (list): A list of dictionaries containing processed data.
        output_file_path (str): The path to the output CSV file.
    """
    if not data:
        logger.warning(f"No data to save to '{output_file_path}'.")
        return

    try:
        file_exists = os.path.exists(output_file_path)
        fieldnames = ['title', 'user_output', 'model_output']

        mode = 'a' if file_exists else 'w'
        with open(output_file_path, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data)
        logger.info(f"Processed data (checkpoint) successfully written/appended to '{output_file_path}'.")
    except IOError as e:
        logger.error(f"Error writing processed data to '{output_file_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving processed data: {e}", exc_info=True)
        
async def chat(message, model):
    """
    Sends a message to the OpenAI chat model asynchronously and returns the response.
    Includes robust error handling and retry mechanism for API calls.

    Args:
        message (str): The input message for the chat model.

    Returns:
        str or None: The model's response string if successful, None otherwise.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model,
                stream=True,
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                timeout=30
            )
            answer = ""
            async for chunk in response:
                res = chunk.choices[0].delta.content or ""
                answer = answer + res
            return answer
        except RateLimitError as e:
            logger.warning(f"OpenAI API Rate Limit Exceeded (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for RateLimitError. Giving up.")
                return None
        except APITimeoutError as e:
            logger.warning(f"OpenAI API Timeout Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for APITimeoutError. Giving up.")
                return None
        except APIError as e:
            logger.warning(f"OpenAI API Error (Code: {e.status_code}, Type: {e.type}) (Attempt {attempt + 1}/{MAX_RETRIES}): {e.message}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for APIError. Giving up.")
                return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during API call (Attempt {attempt + 1}/{MAX_RETRIES}): {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Max retries reached for unexpected error. Giving up.")
                return None
    return None

async def run_tests(model, input_data_list, output_csv_file):
# def run_tests(model, input_data_list, output_csv_file):
    """
    Runs tests by calling the chat model for each item concurrently and collecting results.
    Includes checkpointing to save data periodically and resumption logic.

    Args:
        title_selected_dict (list): List of dictionaries with 'user_content' and 'assistant_response'.
        output_csv_file (str): The path to the CSV file where results will be written.

    Returns:
        list: The combined output after processing.
    """
    existing_results = load_existing_results(output_csv_file)
    processed_titles = {item['title'] for item in existing_results if 'title' in item}

    # Filter out already processed items
    items_to_process = [item for item in input_data_list if item['title'] not in processed_titles]

    if not items_to_process:
        logger.info("All items already processed or no new items to process. Exiting.")
        return existing_results # Return all existing data

    total_new_items = len(items_to_process)
    logger.info(f"Starting to process {total_new_items} new items (total items in source: {len(input_data_list)}).")

    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # Function to process a single item asynchronously
    async def process_single_item(item, original_index):
        async with semaphore:
            title_input = item['title']
            user_output = item['user_output']
            
            # Construct the message for the chat model
            message = PROMPT_TEMPLATE + title_input

            logger.info(f"Processing item {original_index + 1}/{len(input_data_list)}...")
            model_output = await chat(message, model)

            if model_output is None:
                logger.error(f"API call failed for item {original_index + 1} after all retries.")
                return None # Indicate failure for this item
            else:
                logger.info(f"Successfully processed item {original_index + 1}.")
                return {
                    "title": title_input,
                    "user_output": user_output,
                    "model_output": model_output
                }

    all_processed_results = list(existing_results) # Start with existing results

    # The actual_start_index tracks the index in the original title_selected_dict
    for i in range(0, total_new_items, BATCH_SIZE):
        batch_items = items_to_process[i:i + BATCH_SIZE]
        
        # Create a list of coroutines for the current batch
        # Map back to original index for comprehensive logging
        coroutines = []
        for batch_item in batch_items:
            try:
                original_index = input_data_list.index(batch_item)
            except ValueError:
                # Fallback if item is not found (e.g., if input_data_list was filtered differently)
                original_index = -1 # Or handle as appropriate
            coroutines.append(process_single_item(batch_item, original_index))

        logger.info(f"Processing batch {i // BATCH_SIZE + 1}. Number of items in this batch: {len(batch_items)}")
        
        # Run all coroutines in the current batch concurrently
        # gather will run tasks concurrently and return results in the order of coroutines
        batch_results = await asyncio.gather(*coroutines) 
        
        # Add successful results from the batch to the overall list
        successful_batch_results = [res for res in batch_results if res is not None]
        all_processed_results.extend(successful_batch_results)

        # Save the successful results of this batch to CSV
        if successful_batch_results:
            save_processed_data(successful_batch_results, output_csv_file)
        else:
            logger.warning(f"No successful results in batch {i // BATCH_SIZE + 1} to save.")

    return all_processed_results

def generate_classification_metrics_report(output_csv_file):
    logger.info("Generating classification metrics report.")
    # Calculate the number of items in each batch
    # Load the (potentially processed) CSV file into a DataFrame
    try:
        df = pd.read_csv(output_csv_file)
    except FileNotFoundError:
        logger.error(f"Error: Output file '{output_csv_file}' not found for metrics generation.")
        return
    except Exception as e:
        logger.error(f"Error reading CSV for metrics generation from '{output_csv_file}': {e}", exc_info=True)
        return

    # Ensure the necessary columns exist for metrics calculation
    if 'user_output' not in df.columns or 'model_output' not in df.columns:
        logger.error(f"Required columns 'user_output' and 'model_output' not found in '{output_csv_file}'.")
        return

    # Normalize labels and filter
    before_count = len(df)
    df = df[df['user_output'].isin(["Selected", "Not Selected"])]
    df = df[df['model_output'].isin(["Selected", "Not Selected"])]
    after_count = len(df)
    logger.info(f"Filtered data: {before_count} -> {after_count} rows after filtering for 'Selected' and 'Not Selected' labels.")
    
    if after_count == 0:
        logger.warning("No valid data points remaining after filtering for 'Selected' and 'Not Selected' labels. Cannot generate metrics.")
        return

    y_true = df['user_output'].str.strip()
    y_pred = df['model_output'].str.strip()
    
    # Step 3: Compute metrics
    # Handle cases where there's only one class present after filtering
    unique_labels_true = y_true.unique()
    unique_labels_pred = y_pred.unique()
    all_unique_labels = list(set(unique_labels_true) | set(unique_labels_pred))

    if 'Selected' not in all_unique_labels:
        logger.warning("The 'Selected' label is not present in true or predicted labels after filtering. Precision, Recall, F1 for 'Selected' cannot be computed.")
        precision = float('nan')
        recall = float('nan')
        f1 = float('nan')
    else:
        # Only compute if 'Selected' is present, otherwise precision_score will raise an error if pos_label is missing
        precision = precision_score(y_true, y_pred, pos_label='Selected', zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label='Selected', zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label='Selected', zero_division=0)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Ensure all labels expected in the confusion matrix are present in both true and predicted sets
    # This prevents errors if one of the labels ('Selected', 'Not Selected') is missing entirely
    labels_for_matrix = ['Selected', 'Not Selected']
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels_for_matrix)
    class_report = classification_report(y_true, y_pred, labels=labels_for_matrix, zero_division=0)

    # Step 4: Print metrics
    logger.info("Classification Metrics:")
    logger.info(f"Accuracy     : {accuracy:.4f}")
    logger.info(f"Precision    : {precision:.4f}")
    logger.info(f"Recall       : {recall:.4f}")
    logger.info(f"F1 Score     : {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"\nClassification Report:\n{class_report}")

def gather_data_and_run(is_test=False):
    logger.info("Script started execution.")
    output_dir = MODEL_OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: '{output_dir}'")
    
    output_csv_file = os.path.join(output_dir, "output.csv")
    
    today = datetime.date.today().strftime("%Y%m%d")
    input_file_name = PROCESSED_DATA_DIR / f"processed_data_{today}.csv"
    if input_file_name and not os.path.exists(input_file_name):
        logger.error(f"File not found: '{input_file_name}'")

    input_data_list = read_input_csv(input_file_name)
    input_data_list = input_data_list[:105] # Limit to 105 for testing
    
    if not input_data_list:
        logger.error(f"No data extracted from '{input_file_name}'. Please check the file content or path.")
        return False # Indicate failure
    logger.info(f"Total items to consider from source: {len(input_data_list)}")
    
    # Run the asynchronous api call function    
    logger.info(f"Running tests with model '{MODEL}' on {len(input_data_list)} items.")
    asyncio.run(run_tests(MODEL, input_data_list, output_csv_file))
    final_count_after_run = len(load_existing_results(output_csv_file))
    logger.info(f"Total items processed and saved in '{output_csv_file}': {final_count_after_run}")
    
    if is_test:
        generate_classification_metrics_report(output_csv_file)
        logger.info("Classification metrics report generated.")
    
    logger.info("Script finished execution.")
    return True # Indicate success
        

if __name__ == "__main__":
    gather_data_and_run(is_test=True)