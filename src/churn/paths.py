from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PARENT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PARENT_DIR / 'data'

def create_directories():
    """
    Creates the necessary directories if they do not exist.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f'Folder "data" ensured at "{DATA_DIR}"')
               
    except Exception as e:
        
        logging.error(f"An error occurred while creating directories: {e}")

# Call the function to create directories when the module is run directly
if __name__ == "__main__":
    create_directories()