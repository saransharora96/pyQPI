from config.config_radiation_resistance import ENABLE_LOGGING, output_csv_path, files_with_errors
from logging.handlers import QueueHandler, QueueListener
import logging
import time
import os
import utils.dir_utils as file_utils
from multiprocessing import Queue
from typing import Any, Dict

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

def log_result(file_path: str, result: Dict[str, Any], csv_lock,worker_index):
    """
    Logs results using the global logging queue.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if "error" in result:
        with open(files_with_errors, "a") as log_file:
            log_file.write(f"{timestamp} - Error processing file {file_path}: {result['error']}\n")
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        logging.info(f"{worker_index} - Successfully completed {file_path}")
        with csv_lock:
            file_utils.update_csv(result, output_csv_path)  # Update the CSV with results