import os
import logging
import time
import pandas as pd
from typing import Any, Dict
import cupy as cp
from multiprocessing import Pool, Manager, Semaphore 
import gc
from threading import Thread
import threading

from rich.layout import Layout
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.live import Live

from classes.Cell import Cell
from classes.AuxiliaryDataGeneration import AuxiliaryDataGeneration
from config.config_radiation_resistance import background_ri, pixel_x,wavelength, resistance_mapping, memory_thresholds, max_workers, processing_log_path, output_csv_path, RESUME_PROCESSING, dataset_location, max_tasks_per_child
from classes.FeatureExtraction import FeatureExtraction
import utils.dir_utils as file_utils


def worker_initializer(semaphore):
    """
    Initialize global semaphore for workers.
    """
    global global_semaphore
    global_semaphore = semaphore

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
        del auxiliary_generator  # Free auxiliary generator

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
        del cell  # Explicitly delete the object
        cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory blocks
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()  # Trigger garbage collection
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
    global global_semaphore  # Use the global semaphore
    file_path, resistance_label, dish_number, directories, processed_features_in_each_file, csv_lock = args

    global_semaphore.acquire()
    try:
        result = process_cell(file_path, resistance_label, dish_number, directories, processed_features_in_each_file, csv_lock)
        if "error" not in result:
            processed_features_in_each_file[file_path] = list(result.keys())  # Store the processed features
            log_result(file_path, result, csv_lock)
        else:
            logging.error(f"Error in result from process_cell for file {file_path}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}", exc_info=True)
        result = {"file_path": file_path, "error": str(e)}
    finally:    
        global_semaphore.release()
        cp.get_default_memory_pool().free_all_blocks() 
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
    return result


def resource_manager(progress_layout, current_workers, max_workers, semaphore, worker_lock, stop_event):
    """
    Monitors system resources, updates the Rich layout, and throttles workers based on memory usage.
    """
    upper_threshold, lower_threshold, critical_threshold = memory_thresholds

    while not stop_event.is_set():
        try:
            # Fetch resource usage
            usage = file_utils.get_resource_usage()  # Assume this returns a dictionary of resource metrics
            memory_usage = usage['memory_percent']

            try:
                # Update worker counts based on memory usage
                with worker_lock:
                    if memory_usage > critical_threshold:
                        global pool
                        restart_counter += 1
                        logging.warning(f"Memory above critical threshold ({critical_threshold}%). Restarting pool... (Restart #{restart_counter})")
                        pool.close()
                        pool.join()
                        pool = Pool(processes=max_workers, maxtasksperchild=5)
                        current_workers.value = max_workers  # Reset worker count
                    elif memory_usage > upper_threshold and current_workers.value > 1:
                        current_workers.value -= 1  # Decrease workers
                        semaphore.acquire()
                    elif memory_usage < lower_threshold and current_workers.value < max_workers:
                        current_workers.value += 1  # Increase workers
                        semaphore.release()
            except:
                logging.error("Unable to update current_workers", exc_info=True)


            # Update the Rich layout
            monitor_panel = Panel(
                f"Workers: {current_workers.value}/{max_workers}"
                f" | CPU: {usage['cpu_percent']:.1f}%"
                f" | Memory: {usage['memory_percent']:.1f}%"
                f" | GPU: {usage['gpu_percent']:.1f}%"
            )
            progress_layout["monitor"].update(monitor_panel)

        except Exception as e:
            logging.error(f"Error in resource manager: {e}")
            break
        finally:
            time.sleep(2)  # Delay to avoid tight looping


def process_dish(dish_path: str, resistance_label: str, dish_number: int, completely_processed_files: list, 
                 processed_features_for_each_file: Dict[str, Any], csv_lock, progress, progress_task, 
                 start_time, semaphore, already_processed):
    """
    Process all files in a dish directory. Files are processed if they are not in the completely_processed_files set 
    for the features that are missing in processed_features_for_each_file
    """
    dish_path = os.path.normpath(dish_path)
    directories = file_utils.get_output_directories(dish_path)
    files_in_directory = [os.path.join(dish_path, filename) for filename in os.listdir(dish_path)]
    completely_processed_files = set(completely_processed_files)

    # List all valid files in the dish directory
    files = [
        file for file in files_in_directory
        if file.endswith((".tiff", ".tif"))
        and "MIP" not in file
        and "phase" not in file
        and "mask" not in file
        and "segment" not in file
        and os.path.isfile(file)
        and file not in completely_processed_files
    ]

    args_list = [
        (
            file,
            resistance_label,
            dish_number,
            directories,
            processed_features_for_each_file,
            csv_lock
        )
        for file in files
    ]

    try:
        with Pool(processes=max_workers, maxtasksperchild=max_tasks_per_child, initializer=worker_initializer, initargs=(semaphore,)) as pool:
            for _ in pool.imap_unordered(process_file_worker, args_list):
                
                elapsed_time = time.time() - start_time
                completed_files = progress.tasks[progress_task].completed - already_processed
                seconds_per_file = elapsed_time / completed_files if completed_files > 0 else 0.0
                
                progress.update(
                    progress_task, 
                    advance=1, 
                    seconds_per_file=seconds_per_file, 
                    current_cell_line=resistance_label, 
                    current_dish=dish_number
                    )
    except:
        logging.error("Error with pool", exc_info=True)
    finally:
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def process_resistance_folder(resistance_folder: str, base_dir: str, completely_processed_files: list,
                              processed_features_for_each_file: Dict[str, Any], csv_lock, progress, 
                              progress_task, start_time, semaphore, already_processed) -> None:
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
            process_dish(
                dish_path, 
                resistance_label, 
                dish_number, 
                completely_processed_files, 
                processed_features_for_each_file, 
                csv_lock, progress, 
                progress_task, 
                start_time, 
                semaphore, 
                already_processed
                )


def process_directory(base_dir: str, output_csv_path: str, csv_lock) -> None:
    """
    Process all resistance folders and dishes in the base directory and save results to a CSV file.
    Initializes and manages processed files and features to avoid redundant processing.
    """

     # Setup multiprocessing managers for state sharing across processes
    manager = Manager()
    stop_event = threading.Event()
    start_time = time.time()  # Record the start time
    worker_lock = manager.Lock()

    processed_features_for_each_file = manager.dict()
    completely_processed_files = list() # Files where all features are processed
    files_in_csv = set()
    
    if RESUME_PROCESSING and os.path.exists(output_csv_path):
        existing_data = pd.read_csv(output_csv_path)
        files_in_csv.update(existing_data['file_path'].tolist())
        all_features = set(FeatureExtraction.FEATURE_METHODS.keys()) # Define all possible features

        for _, row in existing_data.iterrows():
            file_path = row["file_path"]
            features = [col for col in row.index if col not in ["file_path", "error"] and not pd.isna(row[col])]
            processed_features_for_each_file[file_path] = set(features)

            # Check if all features are processed for this file
            if all_features.issubset(features):
                if file_path not in completely_processed_files:  # Check for uniqueness before appending
                    completely_processed_files.append(file_path)

    else:
        file_utils.reset_processing_environment(dataset_location, processing_log_path, output_csv_path)

    total_files = file_utils.count_total_files(base_dir, resistance_mapping)
    already_processed = len(completely_processed_files)

    # Multiprocessing setup
    current_workers = manager.Value('i', max_workers)

    # Define progress bar
    progress = Progress(
        TextColumn("[bold blue]Overall Progress:"),
        TextColumn(" | [cyan]Cell Line: {task.fields[current_cell_line]}"),
        TextColumn(" | [cyan]Dish: {task.fields[current_dish]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.completed}/{task.total} files"),
        TextColumn(" | [green]{task.fields[seconds_per_file]:.2f} sec/file")
    )
    task = progress.add_task("Processing Files", total=total_files, completed=already_processed, seconds_per_file=0.0, current_dish="Starting...", current_cell_line="Starting...")

    # Define rich layout
    progress_layout = Layout()
    progress_layout.split(
        Layout(name="monitor", size=3),
        Layout(progress, name="progress", ratio=1)
    )

    semaphore = Semaphore(max_workers)
    resource_manager_thread = Thread(
        target=resource_manager,
        args=(
            progress_layout, 
            current_workers, 
            max_workers, 
            semaphore, 
            worker_lock, 
            stop_event),
        daemon=True
    )
    resource_manager_thread.start()
    
    # Render live dashboard
    try:
        with Live(progress_layout, refresh_per_second=1):
            for resistance_folder in resistance_mapping.keys():
                resistance_path = os.path.join(base_dir, resistance_folder)
                if os.path.isdir(resistance_path):
                    try:
                        process_resistance_folder(
                            resistance_folder, 
                            base_dir, 
                            completely_processed_files, 
                            processed_features_for_each_file, 
                            csv_lock, 
                            progress, 
                            task, 
                            start_time, 
                            semaphore,
                            already_processed
                            )
                    except Exception as e:
                        error_message = f"Error processing resistance folder {resistance_folder}: {e}"
                        logging.error(error_message, exc_info=True)
    finally:
        stop_event.set() # stop threads
        resource_manager_thread.join()
        os.system("cls" if os.name == "nt" else "clear") # clear terminal