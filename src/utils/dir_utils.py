import os
from typing import Dict, List, Any
import stat
import shutil
import subprocess
import re
import pandas as pd
import psutil
import logging


# utilities for data reading and directory processing
def get_resistance_label_and_dish(resistance_mapping, resistance_path, dish_folder):
    """
    Get the resistance label and dish number from the resistance mapping, resistance path, and dish folder.
    """
    resistance_label = resistance_mapping[os.path.basename(resistance_path)]
    dish_number = int(dish_folder.replace("dish_", ""))
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
        "image_binary_mask": os.path.join(dish_path, "Image_binary_mask"),
        "mip_segmented": os.path.join(dish_path, "MIP_segmented"),
        "mip_segmented_scaled": os.path.join(dish_path, "MIP_segmented_scaled"),
        "tomogram_binary_mask": os.path.join(dish_path, "Tomogram_binary_mask"),
        "tomogram_segmented": os.path.join(dish_path, "Tomogram_segmented"),
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


def force_delete(path):
    if os.name == 'nt':  # Windows
        subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', path], check=False)
    else:  # macOS/Linux
        subprocess.run(['rm', '-rf', path], check=False)
    print(f"Force deleted: {path}")


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
                    shutil.rmtree(dir_path, onerror=on_rm_error)
                except Exception as e:
                    print(f"Normal deletion failed for {dir_path}, forcing deletion.")
                    force_delete(dir_path)


def remove_files(base_dir, pattern, file_extension):
    """
    Remove files that match a specific pattern and extension from the dataset.

    Args:
        base_dir (str): The base directory to search.
        pattern (str): Regex pattern to match files.
        file_extension (str): Extension of the files to remove.
    """
    regex = re.compile(pattern)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(file_extension) and (regex.search(file) or file_extension == ".png"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


def count_cells_in_dishes(base_dir):
    """
    Count the number of files in each dish and in each cell line, print the results with totals,
    and add spacing for better readability.

    Args:
        base_dir (str): The base directory containing cell line folders.

    Returns:
        dict: A nested dictionary with cell line and dish counts, and prints summary with totals.
    """

    ok_header = '\033[95m'  # Purple
    ok_blue = '\033[94m'  # Blue
    ok_green = '\033[92m'  # Green

    cell_line_counts = {}
    for cell_line in os.listdir(base_dir):
        cell_line_path = os.path.join(base_dir, cell_line)
        if os.path.isdir(cell_line_path):  # Check if it's a directory
            dish_counts = {}
            total_files = 0
            for dish in os.listdir(cell_line_path):
                dish_path = os.path.join(cell_line_path, dish)
                if os.path.isdir(dish_path):  # Check if it's a directory
                    count = len(
                        [file for file in os.listdir(dish_path) if os.path.isfile(os.path.join(dish_path, file))])
                    dish_counts[dish] = count
                    total_files += count
            cell_line_counts[cell_line] = {'dishes': dish_counts, 'total': total_files}

    print("")
    for cell_line, data in cell_line_counts.items():
        print(f"{ok_header}{cell_line}")
        for dish, count in data['dishes'].items():
            print(f"{ok_blue}{dish} has {count} cells")
        print(f"{ok_green}total cells in {cell_line}: {data['total']}\n")

    return cell_line_counts


def read_processed_files(tracker_path):
    """Read processed file paths from a tracker file and normalize them."""
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as file:
            # Normalize each path as it's read from the file
            processed_files = {os.path.normpath(line.strip()) for line in file}
        return processed_files
    return set()


def update_processed_file(tracker_path, file_path):
    """Append a new processed file path to the tracker file after normalizing it."""
    with open(tracker_path, 'a') as file:
        # Normalize the file path before writing to ensure consistency
        normalized_path = os.path.normpath(file_path)
        file.write(normalized_path + '\n')


def reset_processing_environment(dataset_location, processing_log_path, output_csv_path):
    # Remove directories and specific files
    generated_dir = [
        "MIP", "MIP_scaled", "Phase", "Phase_scaled", "Image_binary_mask",
        "MIP_segmented", "MIP_segmented_scaled", "Tomogram_binary_mask",
        "Tomogram_segmented"
    ]
    remove_directories(dataset_location, generated_dir)

    # Remove logs and output files
    for file_path in [processing_log_path, output_csv_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
    print("")


def count_processed_files(base_dir, processed_files):
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if os.path.join(root, file) in processed_files:
                count += 1
    return count

def update_csv(result: Dict[str, Any], output_csv_path: str):
    """
    Update an existing row in the CSV or append a new row if the file does not exist.
    """
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        file_path = result["file_path"]

        # Check if the file already exists in the CSV
        if file_path in df["file_path"].values:
            # Update the existing row
            index = df[df["file_path"] == file_path].index[0]
            for key, value in result.items():
                df.at[index, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True) # Append a new row
    else:
        df = pd.DataFrame([result]) # Create a new CSV if it doesn't exist

    df.to_csv(output_csv_path, index=False)

def get_gpu_usage():
    """
    Retrieve GPU utilization using nvidia-smi.
    Returns:
        float: GPU utilization as a percentage. Returns 0 if no GPUs are detected or an error occurs.
    """
    try:
        # Run the nvidia-smi command to query GPU usage
        result = subprocess.run(
            [
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                '--format=csv,noheader,nounits'
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            return {"gpu_percent": 0.0, "memory_used": 0, "memory_total": 0, "temperature": 0}

        # Parse the output (e.g., "34 %" -> 34)
        gpu_stats_lines = result.stdout.strip().split('\n')
        gpu_stats = []
        for line in gpu_stats_lines:
            gpu_util, mem_used, mem_total, temp = map(str.strip, line.split(','))
            gpu_stats.append({
                "gpu_percent": float(gpu_util),
                "memory_used": float(mem_used),
                "memory_total": float(mem_total),
                "temperature": float(temp)
            })
        return gpu_stats
    except Exception as e:
        logging.error("Error retrieving GPU usage", exc_info=True)
        return [{"gpu_percent": 0.0, "memory_used": 0, "memory_total": 0, "temperature": 0}]

def get_resource_usage():
    """
    Get current resource usage for CPU, memory, disk, and GPU.
    Returns:
        dict: A dictionary containing usage stats.
    """
    gpu_stats = get_gpu_usage()  # Get GPU stats from a function that returns a list of dictionaries

    # Get CPU usage statistics
    cpu_usage = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }

    # Create a dictionary to hold all usage statistics, including CPU and each individual GPU
    usage_stats = {
        "cpu": cpu_usage
    }

    # Iterate over each GPU's statistics and add to the dictionary
    for index, gpu in enumerate(gpu_stats):
        usage_stats[f"gpu_{index + 1}"] = {
            "percent": gpu['gpu_percent'],
            "memory_used": gpu['memory_used'],
            "memory_total": gpu['memory_total'],
            "temperature": gpu['temperature']
        }

    return usage_stats