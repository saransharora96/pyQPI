import os
from tqdm import tqdm
from src.utils.data_classes import Cell
import pandas as pd
import src.utils.dir_utils as file_utils
import src.utils.feature_utils as qpi_utils
from src.config.config_radiation_resistance import background_ri, alpha, pixel_x, pixel_y, pixel_z, wavelength, resistance_mapping
from typing import Any, Dict, List


def process_cell(file_path: str, resistance_label: str, dish_number: int, mip_dir: str, mip_scaled_dir: str, phase_dir: str, phase_scaled_dir: str) -> Dict[str, Any]:
    """
    Process a single cell: calculate dry mass, generate MIP and phase images.
    """
    cell = Cell(file_path, resistance_label, dish_number)
    dry_mass = cell.calculate_dry_mass( background_ri, alpha, pixel_x, pixel_y, pixel_z    )
    qpi_utils.generate_and_save_mip(cell, file_path, mip_dir, mip_scaled_dir)
    qpi_utils.generate_and_save_phase(cell, file_path, phase_dir, phase_scaled_dir, pixel_x, wavelength, background_ri)

    return {
        "file_path": file_path,
        "radiation_resistance": resistance_label,
        "dish_number": dish_number,
        "dry_mass": dry_mass
    }

def process_dish(dish_path: str, resistance_label: str, dish_number: int, progress_bar: tqdm) -> List[Dict[str, Any]]:
    """
    Process all cells in a dish directory.
    """
    results = []
    files = [
        file for file in os.listdir(dish_path)
        if file.endswith(".tiff") or file.endswith(".tif")
           and "MIP" not in file and "phase" not in file  # Exclude old output files
           and os.path.isfile(os.path.join(dish_path, file))
    ]
    # Create output directories with full permissions
    directories = file_utils.get_output_directories(dish_path)
    for dir_type in directories.values():
        file_utils.create_directory_with_permissions(dir_type)

    for file in files:
        file_path = os.path.join(dish_path, file)
        result = process_cell(
            file_path,
            resistance_label,
            dish_number,
            directories["mip"],
            directories["mip_scaled"],
            directories["phase"],
            directories["phase_scaled"]
        )
        results.append(result)
        progress_bar.update(1)
    return results

def process_resistance_folder(resistance_folder: str, base_dir: str, progress_bar: tqdm) -> List[Dict[str, Any]]:
    """
    Process all dishes in a resistance folder.
    """
    results = []
    resistance_path = os.path.join(base_dir, resistance_folder)
    if not os.path.isdir(resistance_path):
        return results

    for dish_folder in os.listdir(resistance_path):
        dish_path = os.path.join(resistance_path, dish_folder)
        if os.path.isdir(dish_path) and dish_folder.startswith("dish"):
            resistance_label, dish_number = file_utils.get_resistance_label_and_dish(
                resistance_mapping, resistance_path, dish_folder
            )
            results.extend(process_dish(dish_path, resistance_label, dish_number, progress_bar))
    return results

def process_directory(base_dir: str, output_csv: str) -> None:
    """
    Process all resistance folders and dishes in the base directory and save results to CSV.
    """
    total_files = file_utils.count_total_files(base_dir, resistance_mapping)
    results = []

    with tqdm(total=total_files, desc="Processing Files", unit="file", dynamic_ncols=True) as progress_bar:
        for resistance_folder in resistance_mapping.keys():
            resistance_path = os.path.join(base_dir, resistance_folder)
            if not os.path.isdir(resistance_path):
                continue

            for dish_folder in os.listdir(resistance_path):
                dish_path = os.path.join(resistance_path, dish_folder)
                if not os.path.isdir(dish_path) or not dish_folder.startswith("dish"):
                    continue

                resistance_label, dish_number = file_utils.get_resistance_label_and_dish(resistance_mapping, resistance_path, dish_folder)

                # Update progress bar description with current cell line and dish
                progress_bar.set_description(f"Processing: Cell Line={resistance_folder}, Dish={dish_folder}")

                results.extend(process_dish(dish_path, resistance_label, dish_number, progress_bar))

    if os.path.exists(output_csv):
        print(f"Warning: {output_csv} already exists, data will not be saved.")
    else:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")