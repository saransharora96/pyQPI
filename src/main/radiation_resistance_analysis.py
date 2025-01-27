import os
import time
import logging
from multiprocessing import Manager, Queue
import warnings
from install_modules import upgrade_pip, install_packages
import utils.dir_utils as file_utils
from process_dataset import process_directory
from progress_logging import setup_logging, configure_listener
from config.config_radiation_resistance import (
    INSTALLATION_NEEDED,
    dataset_location,
    output_csv_path,
    log_file_path
)
import cupy as cp


def execute_code():
    """
    Main function to execute the processing workflow. It sets up logging, manages
    dataset processing, and handles the cleanup of temporary files.
    """

    # Suppress specific PerformanceWarning from CuPy
    warnings.filterwarnings("ignore", message=".*Using synchronous transfer as pinned memory.*")  # Specific warning message

    # Setup logging
    os.remove(log_file_path) if os.path.exists(log_file_path) else None
    log_queue = Queue()
    setup_logging(log_file_path, log_queue)
    listener = configure_listener(log_file_path, log_queue)
    if listener:
        listener.start()

    start_time = time.time()  # Record the start time
    os.system('cls' if os.name == 'nt' else 'clear')

    # Initialize shared manager for multiprocessing
    manager = Manager()
    csv_lock = manager.Lock()

    # Install required Python packages if specified
    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    # Remove temporary files
    # file_utils.remove_files(dataset_location, r"", ".png")  # Delete .png thumbnails
    # file_utils.remove_files(dataset_location, r"T\d{3}_", ".tiff")  # Delete T***_ (un-stitched)
    os.system('cls' if os.name == 'nt' else 'clear')

    logging.info("Starting processing...")

    # Count initial files in dataset
    file_utils.count_cells_in_dishes(dataset_location)

    try:
        # Process the dataset
        process_directory(
            base_dir=dataset_location,
            output_csv_path=output_csv_path,
            csv_lock=csv_lock,
            log_queue=log_queue
        )
        logging.info("Processing completed successfully.")
    except Exception as e:
        # Log any exceptions that occur
        logging.error(f"Processing failed with error: {e}", exc_info=True)
        raise
    finally:
        # Log and display elapsed processing time
        end_time = time.time()  
        elapsed_time = end_time - start_time 
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Stop the logging listener
        if listener:
            listener.stop()

if __name__ == "__main__":
    execute_code()