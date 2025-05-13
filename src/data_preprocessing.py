import os
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import pandas as pd
# from sklearn.model_selection import train_test_split # Not currently used

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Loads an image, converts to RGB, resizes, and normalizes."""
    try:
        # Ensure image_path is a Path object
        image_path = Path(image_path)
        # Explicitly check if the file exists before opening
        if not image_path.is_file():
             # Log the path that failed the check
             logging.error(f"File existence check failed for: {image_path}")
             return None

        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(target_size, Image.LANCZOS) # Use a high-quality resizer
        img_array = np.array(img_resized, dtype=np.float32)
        # Normalize to [0, 1] range
        img_array = img_array / 255.0
        return img_array
    # Keep FileNotFoundError just in case, though is_file() should catch it
    except FileNotFoundError:
        logging.error(f"Image file not found (exception): {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

def create_processed_dataframe(raw_data_dir, output_csv_path):
    """
    Scans raw data directories, creates a dataframe with filepaths and labels,
    and saves it to a CSV file.
    """
    base_path = Path(raw_data_dir) # e.g., /Users/hisham/pneumonia-detection-mlops/data/raw/chest_xray

    # We want paths relative to the project root.
    # If base_path = project_root/data/raw/chest_xray
    # then project_root = base_path.parent.parent.parent
    project_root_for_storing_relative_paths = base_path.parent.parent.parent
    data_info = []

    logging.info(f"Scanning for images in {base_path}")
    # This log message should now point to your actual project root
    logging.info(f"Storing paths relative to: {project_root_for_storing_relative_paths}")

    for split in ['train', 'val', 'test']:
        split_dir = base_path / split
        if not split_dir.is_dir():
            logging.warning(f"Directory not found for split '{split}': {split_dir}")
            continue

        for label in ["NORMAL", "PNEUMONIA"]:
            class_dir = split_dir / label
            if not class_dir.is_dir():
                logging.warning(f"Directory not found for label '{label}' in split '{split}': {class_dir}")
                continue

            image_files = list(class_dir.rglob('*.jp*g'))
            # logging.info(f"Found {len(image_files)} images in {class_dir}") # Can be verbose

            for filepath in image_files:
                try:
                    # Store relative path from the true project root
                    relative_path = filepath.relative_to(project_root_for_storing_relative_paths)
                    data_info.append({
                        'relative_filepath': str(relative_path).replace('\\', '/'),
                        'label': label,
                        'split': split
                    })
                except ValueError as e:
                     logging.error(f"Could not compute relative path for {filepath} regarding {project_root_for_storing_relative_paths}: {e}")


    if not data_info:
        logging.error("No image files found or processed into data_info. Check raw data directory structure and logs.")
        return None

    df = pd.DataFrame(data_info)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Processed dataframe saved to {output_path}. Total entries: {len(df)}")
    return df

# Example usage / testing block
if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray" # Adjust 'chest_xray' if needed
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_CSV = PROCESSED_DATA_DIR / "metadata.csv"

    logging.info("--- Running Data Preprocessing Script ---")
    logging.info(f"Project Root (calculated): {PROJECT_ROOT}")
    logging.info(f"Raw Data Dir (calculated): {RAW_DATA_DIR}")
    logging.info(f"Output CSV (calculated): {OUTPUT_CSV}")

    logging.info("Creating processed metadata dataframe...")
    # This will now just create the CSV and print log messages from within the function.
    create_processed_dataframe(RAW_DATA_DIR, OUTPUT_CSV)

    logging.info("--- Data Preprocessing Script Completed ---")