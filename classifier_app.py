import asyncio
from classifier_excel_data_extractor import ExcelProcessor
from classifier_data_processor import DataProcessor

def extract_data():
    processor = ExcelProcessor()
    processor.process_all_excel_files_in_directory()
    
def process_data(is_test=False, is_base_model=False):
    processor = DataProcessor()    # Run with is_test=True to generate metrics report
    # Use limit_data for quick testing on a subset of data
    asyncio.run(processor.run(is_test=is_test, is_base_model=is_base_model, limit_data=10)) # For production, remove limit_data

if __name__ == "__main__":
    extract_data()
    process_data(is_test=True, is_base_model=True)