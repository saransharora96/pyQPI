import os
import time
import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager, Queue, Process
import multiprocessing
from install_modules import upgrade_pip, install_packages
from config.config_radiation_resistance import (
    INSTALLATION_NEEDED, 
    ENABLE_LOGGING,
    dataset_location, 
    output_csv_path, 
    log_file_path
)
import utils.dir_utils as file_utils
from main.process_dataset import process_directory


def setup_logging(log_file_path: str, log_queue: Queue):
    """
    Set up logging to use a Queue for multiprocessing-safe logging.
    """
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if ENABLE_LOGGING:
        # Add a QueueHandler to pass logs to the main process
        queue_handler = QueueHandler(log_queue)
        logging.root.addHandler(queue_handler)
        logging.root.setLevel(logging.INFO)
    else:
        # Suppress logging by setting a high log level
        logging.root.setLevel(logging.CRITICAL)

def configure_listener(log_file_path: str, log_queue: Queue):
    """
    Configure the QueueListener to write logs from the queue to a file.
    """
    if ENABLE_LOGGING:
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        listener = QueueListener(log_queue, file_handler)
        return listener
    return None

def configure_listener(log_file_path: str, log_queue: Queue):
    """Configure the QueueListener to write logs from the queue to a file."""
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    listener = QueueListener(log_queue, file_handler)
    return listener

def execute_code():

    # Set up logging
    log_queue = Queue()
    setup_logging(log_file_path, log_queue)
    listener = configure_listener(log_file_path, log_queue)

    if listener:
        listener.start()
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
            csv_lock = csv_lock,
            log_queue = log_queue
        )
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error(f"Processing failed with error: {e}", exc_info=True)
        raise
    finally:
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        if listener:
            listener.stop() 

if __name__ == "__main__":

    execute_code()