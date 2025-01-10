import os
import time
import logging
from install_modules import upgrade_pip, install_packages
from config.config_radiation_resistance import INSTALLATION_NEEDED, RESUME_PROCESSING
import utils.dir_utils as file_utils
from main.process_dataset import process_directory
import pandas as pd
from config.config_radiation_resistance import dataset_location, output_csv_path
from multiprocessing import Manager
import multiprocessing


def setup_logging():
    """ Setup logging configuration. """
    logging.basicConfig(
        filename="../pyQPI/src/logs/error_record.log",  # Ensure this path is correct and accessible
        filemode='a',  # Append mode
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def execute_code():

    setup_logging()
    start_time = time.time()  # Record the start time
    os.system('cls' if os.name == 'nt' else 'clear')

    manager = Manager()
    csv_lock = manager.Lock()
    
    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    file_utils.remove_files(dataset_location, r"", ".png")  # Delete .png thumbnails
    file_utils.remove_files(dataset_location, r"T\d{3}_", ".tiff")  # Delete T***_ (un-stitched)

    os.system('cls' if os.name == 'nt' else 'clear')
    logging.info("Starting processing...")
    file_utils.count_cells_in_dishes(dataset_location)

    try:
        process_directory(
            base_dir=dataset_location,
            output_csv_path=output_csv_path,
            csv_lock = csv_lock
        )
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error(f"Processing failed with error: {e}", exc_info=True)
        raise

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")  # compatibility for debugging
    execute_code()