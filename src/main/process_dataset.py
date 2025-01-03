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


def log_result(file_path: str, result: Dict[str, Any], progress_bar: tqdm):
    if "error" in result:
        with open(processing_log_path, "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error processing file {file_path}: {result['error']}\n")
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        # Update the CSV with results
        file_utils.update_csv(result, output_csv_path)
    progress_bar.update(1)


def process_cell(file_path: str, resistance_label: str, dish_number: int, directories: Dict[str, str], processed_features: Dict[str, Any], progress_bar: tqdm) -> Dict[str, Any]:
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
                feature_result = method(required_data, *feature_info.get("args", []), **feature_info.get("kwargs", {}))

                if isinstance(feature_result, list) or isinstance(feature_result, cp.ndarray):
                    for idx, value in enumerate(feature_result):
                        results[f"{feature_name}_{idx + 1}"] = value
                else:
                    results[feature_name] = feature_result

            except Exception as e:
                logging.warning(f"Feature {feature_name} failed for file {file_path}: {e}")
                results[f"error"] = str(e)
                       
        cell.unload_all_data()
        log_result(file_path, results, progress_bar)
        return results
    
    except Exception as e:
        logging.error(f"Error processing cell {file_path}: {e}", exc_info=True)
        results["error"] = str(e)
        log_result(file_path, results, progress_bar)
        return results


def process_dish(dish_path: str, resistance_label: str, dish_number: int, progress_bar: tqdm, processed_files: set, output_csv_path: str, processed_features: Dict[str, Any]):
    """
    Process all files in a dish directory. Files are processed if they are not in the processed files log
    or if they have missing features.
    """
    dish_path = os.path.normpath(dish_path)
    directories = file_utils.get_output_directories(dish_path)

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

    for file in files:
        file_path = os.path.normpath(os.path.join(dish_path, file))
        if file_path not in processed_files or set(FeatureExtraction.FEATURE_METHODS.keys()) - set(processed_features.get(file_path, [])):
            result = process_cell(file_path, resistance_label, dish_number, directories, processed_features, progress_bar)
            if "error" not in result:
                processed_files.add(file_path)


def process_resistance_folder(resistance_folder: str, base_dir: str, processed_files: set,
                              progress_bar: tqdm, output_csv_path: str, processed_features: Dict[str, Any]) -> None:
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
            process_dish(dish_path, resistance_label, dish_number, progress_bar, processed_files, output_csv_path, processed_features)


def process_directory(base_dir: str, output_csv_path: str) -> None:
    """
    Process all resistance folders and dishes in the base directory and save results to a CSV file.
    """
    processed_files = set()
    processed_features = {}
    
    if os.path.exists(output_csv_path):
        existing_data = pd.read_csv(output_csv_path)
        for _, row in existing_data.iterrows():
            file_path = row["file_path"]
            features = [col for col in row.index if col not in ["file_path", "error"] and not pd.isna(row[col])]
            processed_features[file_path] = features
        processed_files.update(existing_data['file_path'].tolist())

    # Identify unprocessed features
    all_features = set(FeatureExtraction.FEATURE_METHODS.keys())
    files_to_reprocess = {
        f: list(all_features - set(processed_features.get(f, [])))
        for f in processed_files
        if not all_features.issubset(set(processed_features.get(f, [])))
    }

    total_files = file_utils.count_total_files(base_dir, resistance_mapping)
    files_not_processed = total_files - len(processed_files)
    already_processed = total_files - len(files_to_reprocess) - files_not_processed

    with tqdm(total=total_files, initial=already_processed, desc=f"Processing Files: ", unit="file", dynamic_ncols=True, colour="yellow") as progress_bar:
        for resistance_folder in resistance_mapping.keys():
            resistance_path = os.path.join(base_dir, resistance_folder)
            if os.path.isdir(resistance_path):
                process_resistance_folder(resistance_folder, base_dir, processed_files, progress_bar, output_csv_path, processed_features)
    os.system('cls' if os.name == 'nt' else 'clear')