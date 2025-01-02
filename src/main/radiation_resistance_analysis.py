import os
import time
import logging
from install_modules import upgrade_pip, install_packages
from config.config_radiation_resistance import INSTALLATION_NEEDED, RESUME_PROCESSING
import utils.dir_utils as file_utils
from main.process_dataset import process_directory

logging.basicConfig(
    filename="../pyQPI/src/logs/error_record.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def execute_code():

    start_time = time.time()  # Record the start time
    os.system('cls' if os.name == 'nt' else 'clear')
    log_paths = {'processed':"../pyQPI/src/logs/feature_processing_completed.log",'skipped':"../pyQPI/src/logs/skipped_files.log"}
    
    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    dataset_location = (
        "D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\OracleQPI\pyQPI\data"
    )
    # dataset_location = (
    #     "E:\radiation_resistance_dataset_export"
    # )

    file_utils.remove_files(dataset_location, r"", ".png")  # Delete .png thumbnails
    file_utils.remove_files(dataset_location, r"T\d{3}_", ".tiff")  # Delete T***_ (un-stitched)

    if RESUME_PROCESSING:
        processed_files = file_utils.read_processed_files(log_paths["processed"])
    else:
        file_utils.reset_processing_environment(dataset_location, log_paths, "extracted_parameters.csv")
        processed_files=set()

    logging.info("Starting processing...")
    try:
        process_directory(
            base_dir=dataset_location,
            output_csv="extracted_parameters.csv",
            processed_files=processed_files,
            log_paths=log_paths
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





