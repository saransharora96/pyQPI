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
import cupy as cp
from config.config_radiation_resistance import processing_log_path, output_csv_path
from multiprocessing import Pool, Manager, Lock


def log_result(file_path: str, result: Dict[str, Any], csv_lock):
    if "error" in result:
        with open(processing_log_path, "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error processing file {file_path}: {result['error']}\n")
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        # Update the CSV with results
        with csv_lock:
            file_utils.update_csv(result, output_csv_path)
                                     

def process_cell(file_path: str, resistance_label: str, dish_number: int,
                 directories: Dict[str, str], processed_features: Dict[str, Any], csv_lock):
    results = {"file_path": file_path, "radiation_resistance": resistance_label, "dish_number": dish_number}
    try:
        cell = Cell(file_path, resistance_label, dish_number)
        auxiliary_generator = AuxiliaryDataGeneration(cell, directories, pixel_x=pixel_x, wavelength=wavelength, background_ri=background_ri)
        auxiliary_generator.generate_and_save_auxiliary_data()

        # Process only unprocessed features
        unprocessed_features = set(FeatureExtraction.FEATURE_METHODS.keys()) - set(processed_features.get(file_path, []))
        for feature_name in unprocessed_features:
            feature_info = FeatureExtraction.FEATURE_METHODS[feature_name]
            try:
                required_data = cell.load_data(feature_info["data_type"])
                if required_data is None:
                    raise ValueError(f"Required data '{feature_info['data_type']}' is missing for feature '{feature_name}'.")

                method = feature_info["method"].__func__
                args = feature_info.get("args", [])
                kwargs = feature_info.get("kwargs", {})
                feature_result = method(required_data, *args, **kwargs)

                if isinstance(feature_result, list) or isinstance(feature_result, cp.ndarray):
                    for idx, value in enumerate(feature_result):
                        results[f"{feature_name}_{idx + 1}"] = value
                else:
                    results[feature_name] = feature_result

            except Exception as e:
                logging.warning(f"Feature {feature_name} failed for file {file_path}: {e}")
                results[f"error"] = str(e)

        cell.unload_all_data()
        log_result(file_path, results, csv_lock)
        return results

    except Exception as e:
        logging.error(f"Error processing cell {file_path}: {e}", exc_info=True)
        results["error"] = str(e)
        log_result(file_path, results, csv_lock)
        return results


def process_file_worker(args):
    """
    Worker function to process individual files.
    """
    file_path, resistance_label, dish_number, directories, processed_features, csv_lock = args

    # Process the cell and update processed features
    result = process_cell(file_path, resistance_label, dish_number, directories, processed_features, csv_lock)

    # Update processed features in the shared dictionary
    if "error" not in result:
        processed_features[file_path] = list(result.keys())  # Store the processed features

    return result


def process_dish(dish_path: str, resistance_label: str, dish_number: int, progress_bar: tqdm, processed_files: list, output_csv_path: str, processed_features: Dict[str, Any], csv_lock):
    """
    Process all files in a dish directory. Files are processed if they are not in the processed files log
    or if they have missing features.
    """
    dish_path = os.path.normpath(dish_path)
    directories = file_utils.get_output_directories(dish_path)
    processed_files_set = set(processed_files)

    # List all valid files in the dish directory
    files = [
        file for file in os.listdir(dish_path)
        if file.endswith((".tiff", ".tif"))
        and "MIP" not in file
        and "phase" not in file
        and "mask" not in file
        and "segment" not in file
        and os.path.isfile(os.path.join(dish_path, file))
    ]

    # Prepare arguments for worker processes

    args_list = [
        (
            os.path.join(dish_path, file),
            resistance_label,
            dish_number,
            directories,
            processed_features,
            csv_lock
        )
        for file in files
        if file not in processed_files_set or set(FeatureExtraction.FEATURE_METHODS.keys()) - set(processed_features.get(file, []))
    ]

    with Pool(processes=os.cpu_count()) as pool:
        for _ in pool.imap_unordered(process_file_worker, args_list):
            progress_bar.update(1)

    # Update processed_files after processing
    for result in args_list:
        processed_files.append(result[0])  # Append file path


def process_resistance_folder(resistance_folder: str, base_dir: str, processed_files: set,
                              progress_bar: tqdm, output_csv_path: str, processed_features: Dict[str, Any], csv_lock) -> None:
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
            process_dish(dish_path, resistance_label, dish_number, progress_bar, processed_files, output_csv_path, processed_features, csv_lock)


def process_directory(base_dir: str, output_csv_path: str, csv_lock) -> None:
    """
    Process all resistance folders and dishes in the base directory and save results to a CSV file.
    """
    manager = Manager()
    processed_files = manager.list()
    processed_features = manager.dict()
    
    if os.path.exists(output_csv_path):
        existing_data = pd.read_csv(output_csv_path)
        for _, row in existing_data.iterrows():
            file_path = row["file_path"]
            features = [col for col in row.index if col not in ["file_path", "error"] and not pd.isna(row[col])]
            processed_features[file_path] = features
        processed_files.update(existing_data['file_path'].tolist())

    # Identify unprocessed features
    all_features = set(FeatureExtraction.FEATURE_METHODS.keys())
    processed_files_set = set(processed_files)  # Convert to set for membership checks
    files_to_reprocess = {
        f: list(all_features - set(processed_features.get(f, [])))
        for f in processed_files_set
        if not all_features.issubset(set(processed_features.get(f, [])))
    }

    total_files = file_utils.count_total_files(base_dir, resistance_mapping)
    files_not_processed = total_files - len(processed_files_set)
    already_processed = total_files - len(files_to_reprocess) - files_not_processed

    with tqdm(total=total_files, initial=already_processed, desc=f"Processing Files: ", unit="file", dynamic_ncols=True, colour="yellow") as progress_bar:
        for resistance_folder in resistance_mapping.keys():
            resistance_path = os.path.join(base_dir, resistance_folder)
            if os.path.isdir(resistance_path):
                process_resistance_folder(resistance_folder, base_dir, processed_files, progress_bar, output_csv_path, processed_features, csv_lock)
    os.system('cls' if os.name == 'nt' else 'clear')