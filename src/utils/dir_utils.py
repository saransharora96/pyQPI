import os
from typing import Dict, List
import stat
import shutil


# utilities for data reading and directory processing
def get_resistance_label_and_dish(resistance_mapping, resistance_path, dish_folder):
    """
    Get the resistance label and dish number from the resistance mapping, resistance path, and dish folder.
    """
    resistance_label = resistance_mapping[os.path.basename(resistance_path)]
    dish_number = int(dish_folder.replace("dish", ""))
    return resistance_label, dish_number

def create_directory(path: str) -> None:
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)

def get_output_directories(dish_path: str) -> Dict[str, str]:
    """
    Get or create output directories for MIP and Phase images.
    """
    directories = {
        "mip": os.path.join(dish_path, "MIP"),
        "mip_scaled": os.path.join(dish_path, "MIP_scaled"),
        "phase": os.path.join(dish_path, "Phase"),
        "phase_scaled": os.path.join(dish_path, "Phase_scaled"),
    }
    for dir_path in directories.values():
        create_directory(dir_path)
    return directories

def count_total_files(base_dir: str, resistance_mapping: dict) -> int:
    return sum(
        len([
            file for file in os.listdir(os.path.join(base_dir, folder, dish))
            if file.endswith(".tiff") or file.endswith(".tif")
            and os.path.isfile(os.path.join(base_dir, folder, dish, file))  # Top-level only
        ])
        for folder in resistance_mapping.keys()
        for dish in os.listdir(os.path.join(base_dir, folder))
        if os.path.isdir(os.path.join(base_dir, folder, dish))
    )


def create_directory_with_permissions(path: str) -> None:
    """
    Create a directory and set full control permissions for the user.
    """
    try:
        os.makedirs(path, exist_ok=True)
        # Set full control permissions for user
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")


def on_rm_error(func, path, exc_info):
    """
    Error handler for `shutil.rmtree` to handle permission issues.
    """
    if not os.access(path, os.W_OK):
        print(f"Changing permissions for: {path}")
        os.chmod(path, stat.S_IWUSR)  # Grant write permissions
        func(path)
    else:
        print(f"Failed to remove {path}")
        raise exc_info[1]

def remove_directories(base_dir: str, directories_to_remove: List[str]) -> None:
    """
    Remove specified directories and their contents recursively from the dataset.

    Args:
        base_dir (str): The base directory containing datasets.
        directories_to_remove (List[str]): List of directory names to remove.
    """
    for root, dirs, _ in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            if dir_name in directories_to_remove:
                dir_path = os.path.join(root, dir_name)
                try:
                    print(f"Removing directory and its contents: {dir_path}")
                    shutil.rmtree(dir_path, onerror=on_rm_error)
                except OSError as e:
                    print(f"Error removing {dir_path}: {e}")
