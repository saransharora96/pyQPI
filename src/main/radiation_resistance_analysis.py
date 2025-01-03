import os
import time
import logging
from install_modules import upgrade_pip, install_packages
from config.config_radiation_resistance import INSTALLATION_NEEDED, RESUME_PROCESSING
import utils.dir_utils as file_utils
from main.process_dataset import process_directory
import pandas as pd
from config.config_radiation_resistance import dataset_location, processing_log_path, output_csv_path

logging.basicConfig(
    filename="../pyQPI/src/logs/error_record.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def execute_code():

    start_time = time.time()  # Record the start time
    os.system('cls' if os.name == 'nt' else 'clear')
    
    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    file_utils.remove_files(dataset_location, r"", ".png")  # Delete .png thumbnails
    file_utils.remove_files(dataset_location, r"T\d{3}_", ".tiff")  # Delete T***_ (un-stitched)

    if RESUME_PROCESSING:
        processed_files = set()
        processed_features = {}
        if os.path.exists(output_csv_path):
            existing_data = pd.read_csv(output_csv_path)
            for _, row in existing_data.iterrows():
                file_path = row["file_path"]
                features = [col for col in row.index if col not in ["file_path", "error"] and not pd.isna(row[col])]
                processed_features[file_path] = features
            processed_files.update(existing_data['file_path'].tolist())
    else:
        file_utils.reset_processing_environment(dataset_location, processing_log_path, output_csv_path)
        processed_files = set()
        processed_features = {}

    logging.info("Starting processing...")
    try:
        process_directory(
            base_dir=dataset_location,
            output_csv_path=output_csv_path,
        )
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error(f"Processing failed with error: {e}", exc_info=True)
        raise

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    execute_code()





