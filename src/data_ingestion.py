# src/data_ingestion.py
import os
import zipfile
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent # This should be your project's root directory
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
KAGGLE_DATASET_ID = "paultimothymooney/chest-xray-pneumonia"

def download_dataset():
    """Downloads the dataset from Kaggle."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading dataset '{KAGGLE_DATASET_ID}' to '{RAW_DATA_DIR}'...")

    try:
        # Using subprocess to call the kaggle CLI
        # The dataset will be downloaded as a zip file named chest-xray-pneumonia.zip
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_ID, "-p", str(RAW_DATA_DIR), "--unzip"],
            check=True,
            capture_output=True # Suppress kaggle CLI output unless there's an error
        )
        logging.info("Dataset downloaded and unzipped successfully.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download dataset. Error: {e}")
        if e.stderr:
            logging.error(f"Kaggle CLI error: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        logging.error("Kaggle CLI not found. Make sure it's installed and in your PATH.")
        raise

if __name__ == "__main__":
    logging.info("Starting data ingestion process...")
    download_dataset()
    logging.info("Data ingestion process completed.")
    logging.info(f"Please check the contents of: {RAW_DATA_DIR}")