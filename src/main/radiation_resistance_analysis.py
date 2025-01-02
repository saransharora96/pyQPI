import os
import time
import logging
from tqdm import tqdm
import pandas as pd
from typing import Any, Dict
from memory_profiler import profile
from src.classes.Cell import Cell
from src.install_modules import upgrade_pip, install_packages
from src.config.config_radiation_resistance import INSTALLATION_NEEDED, RESUME_PROCESSING
import src.utils.dir_utils as file_utils
from src.classes.AuxiliaryDataGeneration import AuxiliaryDataGeneration
from src.config.config_radiation_resistance import background_ri, alpha, pixel_x, pixel_y, pixel_z, wavelength, resistance_mapping
from src.classes.FeatureExtraction import FeatureExtraction


logging.basicConfig(
    filename="../logs/error_record.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_cell(file_path: str, resistance_label: str, dish_number: int, directories: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a single cell: calculate dry mass, generate MIP and phase images, and segment tomograms.
    """
    # try:

    cell = Cell(file_path, resistance_label, dish_number)  # create an instance of a cell
    auxiliary_generator = AuxiliaryDataGeneration(cell, directories, pixel_x=pixel_x, wavelength=wavelength, background_ri=background_ri)
    auxiliary_generator.generate_and_save_auxiliary_data()
    dry_mass = FeatureExtraction.calculate_dry_mass(cell.load_data('tomogram_segmented'), background_ri, alpha, pixel_x, pixel_y, pixel_z)
    cell.unload_data('tomogram_segmented')  # Unload data after processing
    return {
        "file_path": file_path,
        "radiation_resistance": resistance_label,
        "dish_number": dish_number,
        "dry_mass": dry_mass
    }
    # except Exception as general_exc:
    #     logging.error(f"Unexpected error while processing cell {file_path}: {general_exc}", exc_info=True)
    #     return {
    #         "file_path": file_path,
    #         "radiation_resistance": resistance_label,
    #         "dish_number": dish_number,
    #         "error": str(general_exc)
    #     }



def process_dish(dish_path: str, resistance_label: str, dish_number: int, progress_bar: tqdm, processed_files: set, tracker_path: str, output_csv_path: str) -> None:
    """
    Process all cells in a dish directory.
    """
    results = []
    dish_path = os.path.normpath(dish_path)  # Normalize the dish path
    directories = file_utils.get_output_directories(dish_path)
    skipped_files_log = os.path.join("src", "logs", "skipped_files.log")

    files = [
        file for file in os.listdir(dish_path)
        if file.endswith((".tiff", ".tif")) and
           "MIP" not in file and "phase" not in file and
           "mask" not in file and "segment" not in file and
           os.path.isfile(os.path.join(dish_path, file)) and
           os.path.join(dish_path, file) not in processed_files
    ]

    # Ensure log directory exists
    if not os.path.exists("src/logs"):
        os.makedirs("src/logs")

    for file in files:
        file_path = os.path.normpath(os.path.join(dish_path, file))
        result = process_cell(file_path, resistance_label, dish_number, directories)
        if "error" in result:
            with open(skipped_files_log, "a") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error processing file {file_path}: {result['error']}\n")
        else:
            file_utils.append_results_to_csv([result], output_csv_path)
            file_utils.update_processed_file(tracker_path, file_path)
        progress_bar.update(1)


def process_resistance_folder(resistance_folder: str, base_dir: str, processed_files: set, tracker_path: str,
                              progress_bar: tqdm, output_csv_path: str) -> None:
    """
    Process all dishes in a resistance folder. Each dish is processed if it's a directory and it starts with 'dish'.

    Args:
    resistance_folder (str): The folder name representing a specific resistance category.
    base_dir (str): The base directory where resistance folders are located.
    processed_files (set): Set of file paths that have been processed.
    tracker_path (str): Path to the file tracking processed files.
    progress_bar (tqdm): Progress bar instance for visual feedback.
    output_csv_path (str): Path to the CSV file where results are appended after processing each cell.

    Returns:
    None: Results are saved directly to the CSV file after each cell is processed.
    """
    resistance_path = os.path.join(base_dir, resistance_folder)
    if not os.path.isdir(resistance_path):
        return  # If the path is not a directory, there's nothing to process.

    for dish_folder in os.listdir(resistance_path):
        dish_path = os.path.join(resistance_path, dish_folder)
        if os.path.isdir(dish_path) and dish_folder.startswith("dish"):
            resistance_label, dish_number = file_utils.get_resistance_label_and_dish(
                resistance_mapping, resistance_path, dish_folder
            )
            progress_bar.set_description(f"Processing: cell Line = {resistance_folder}, dish = {dish_folder}")
            process_dish(dish_path, resistance_label, dish_number, progress_bar, processed_files, tracker_path, output_csv_path)


def process_directory(base_dir: str, output_csv: str, processed_files: set, tracker_path: str) -> None:
    """
    Process all resistance folders and dishes in the base directory and save results to a CSV file.
    """
    output_csv_path = os.path.join(base_dir, output_csv)
    if os.path.exists(output_csv_path):
        existing_data = pd.read_csv(output_csv_path)
        processed_files.update(existing_data['file_path'].tolist())

    total_files = file_utils.count_total_files(base_dir, resistance_mapping)
    already_processed = file_utils.count_processed_files(base_dir, processed_files)

    with tqdm(total=total_files, initial=already_processed, desc="Processing Files", unit="file", dynamic_ncols=True, bar_format="{l_bar}\033[93m{bar}\033[0m{r_bar}") as progress_bar:
        for resistance_folder in resistance_mapping.keys():
            resistance_path = os.path.join(base_dir, resistance_folder)
            if os.path.isdir(resistance_path):
                process_resistance_folder(resistance_folder, base_dir, processed_files, tracker_path, progress_bar, output_csv_path)


def execute_code():

    start_time = time.time()  # Record the start time
    tracker_path = "../logs/feature_processing_completed.log"
    processed_files = set()

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    # dataset_location = (
    #     r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\OracleQPI\pyQPI\data"
    # )
    dataset_location = (
        r"E:\radiation_resistance_dataset_export"
    )

    if RESUME_PROCESSING:
        processed_files = file_utils.read_processed_files(tracker_path)
    else:
        file_utils.reset_processing_environment(dataset_location, tracker_path, "extracted_parameters.csv")

    logging.info("Starting processing...")
    try:
        process_directory(
            base_dir=dataset_location,
            output_csv="extracted_parameters.csv",
            processed_files=processed_files,
            tracker_path=tracker_path
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





