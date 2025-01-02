import os
import logging
import time
from tqdm import tqdm
import pandas as pd
from typing import Any, Dict
from memory_profiler import profile
from classes.Cell import Cell
from classes.AuxiliaryDataGeneration import AuxiliaryDataGeneration
from config.config_radiation_resistance import background_ri, alpha, pixel_x, pixel_y, pixel_z, wavelength, resistance_mapping
from classes.FeatureExtraction import FeatureExtraction
import utils.dir_utils as file_utils


def log_result(file_path: str, result: Dict[str, Any], log_paths: Dict[str, str]):
    """
    Log the result of processing, either appending to the processed log or the skipped log based on the presence of an error.
    """
    if "error" in result:
        with open(log_paths['skipped'], "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error processing file {file_path}: {result['error']}\n")
            os.system('cls' if os.name == 'nt' else 'clear')
    else:
        with open(log_paths['processed'], "a") as log_file:
            log_file.write(f"{file_path}\n")


def process_cell(file_path: str, resistance_label: str, dish_number: int, directories: Dict[str, str], log_paths: Dict[str, str]) -> Dict[str, Any]:
    try:
        cell = Cell(file_path, resistance_label, dish_number)
        auxiliary_generator = AuxiliaryDataGeneration(cell, directories, pixel_x=pixel_x, wavelength=wavelength, background_ri=background_ri)
        auxiliary_generator.generate_and_save_auxiliary_data()
        dry_mass = FeatureExtraction.calculate_dry_mass(cell.load_data('tomogram_segmented'), background_ri, alpha, pixel_x, pixel_y, pixel_z)
        cell.unload_data('tomogram_segmented')
        result = {
            "file_path": file_path,
            "radiation_resistance": resistance_label,
            "dish_number": dish_number,
            "dry_mass": dry_mass
        }
        log_result(file_path, result, log_paths)
        return result
    except Exception as general_exc:
        logging.error(f"Unexpected error while processing cell {file_path}: {general_exc}", exc_info=True)
        result = {
            "file_path": file_path,
            "radiation_resistance": resistance_label,
            "dish_number": dish_number,
            "error": str(general_exc)
        }
        log_result(file_path, result, log_paths)
        return result


def process_dish(dish_path: str, resistance_label: str, dish_number: int, progress_bar: tqdm, processed_files: set, log_paths: Dict[str, str], output_csv_path: str):
    dish_path = os.path.normpath(dish_path)
    directories = file_utils.get_output_directories(dish_path)
    files = [
        file for file in os.listdir(dish_path)
        if file.endswith((".tiff", ".tif")) and "MIP" not in file and "phase" not in file and
           "mask" not in file and "segment" not in file and os.path.isfile(os.path.join(dish_path, file)) and
           os.path.join(dish_path, file) not in processed_files
    ]
    for file in files:
        file_path = os.path.normpath(os.path.join(dish_path, file))
        result = process_cell(file_path, resistance_label, dish_number, directories, log_paths)
        if "error" not in result:
            file_utils.append_results_to_csv([result], output_csv_path)
        progress_bar.update(1)


def process_resistance_folder(resistance_folder: str, base_dir: str, processed_files: set, log_paths: Dict[str, str],
                              progress_bar: tqdm, output_csv_path: str) -> None:
    """
    Process all dishes in a resistance folder. Each dish is processed if it's a directory and it starts with 'dish'.

    Args:
    resistance_folder (str): The folder name representing a specific resistance category.
    base_dir (str): The base directory where resistance folders are located.
    processed_files (set): Set of file paths that have been processed.
    completed_files (str): Path to the file tracking processed files.
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
            process_dish(dish_path, resistance_label, dish_number, progress_bar, processed_files, log_paths, output_csv_path)


def process_directory(base_dir: str, output_csv: str, processed_files: set, log_paths: Dict[str, str]) -> None:
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
                process_resistance_folder(resistance_folder, base_dir, processed_files, log_paths, progress_bar, output_csv_path)
