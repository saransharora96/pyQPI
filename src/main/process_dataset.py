import os
import logging
import time
import pandas as pd
from typing import Any, Dict
import cupy as cp
import gc
from multiprocessing import Pool, Manager, Semaphore, Queue
from threading import Thread
import threading
from logging.handlers import QueueHandler
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from classes.Cell import Cell
from classes.AuxiliaryDataGeneration import AuxiliaryDataGeneration
from classes.FeatureExtraction import FeatureExtraction
import utils.dir_utils as file_utils
from config.config_radiation_resistance import (
    background_ri,
    pixel_x,
    wavelength,
    resistance_mapping,
    files_with_errors,
    output_csv_path,
    dataset_location,
    memory_thresholds,
    max_workers,
    initial_workers,
    max_tasks_per_child,
    resource_check_frequency,
    queue_chunk_size,
    RESUME_PROCESSING,
    ENABLE_LOGGING
)
import asyncio

pool = None

def worker_initializer(semaphore, log_queue, ENABLE_LOGGING):
    """
    Initialize global semaphore for workers.
    """
    global global_semaphore, global_log_queue
    global_semaphore = semaphore
    global_log_queue = log_queue

    # Remove existing handlers
    logging.root.handlers = []

    if ENABLE_LOGGING:
        # Add QueueHandler to pass logs to the main process
        queue_handler = QueueHandler(log_queue)
        logging.root.addHandler(queue_handler)
        logging.root.setLevel(logging.INFO)
    else:
        # Suppress logging by setting a high log level
        logging.root.setLevel(logging.CRITICAL)

def log_result(file_path: str, result: Dict[str, Any], csv_lock):
    """
    Logs results using the global logging queue.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if "error" in result:
        with open(files_with_errors, "a") as log_file:
            log_file.write(f"{timestamp} - Error processing file {file_path}: {result['error']}\n")
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        logging.info(f"{timestamp} - Successfully completed {file_path}")
        with csv_lock:
            file_utils.update_csv(result, output_csv_path)  # Update the CSV with results

async def process_cell(file_path: str, resistance_label: str, dish_number: int,
                 directories: Dict[str, str], processed_features: Dict[str, Any], csv_lock,):
    results = {"file_path": file_path, "radiation_resistance": resistance_label, "dish_number": dish_number}
    try:
        cell = Cell(file_path, resistance_label, dish_number)
        auxiliary_generator = AuxiliaryDataGeneration(cell, directories, pixel_x=pixel_x, wavelength=wavelength, background_ri=background_ri)
        await auxiliary_generator.generate_and_save_auxiliary_data()
        del auxiliary_generator  # Free auxiliary generator

        # Process only unprocessed features
        unprocessed_features = set(FeatureExtraction.FEATURE_METHODS.keys()) - set(processed_features.get(file_path, []))
        for feature_name in unprocessed_features:
            feature_info = FeatureExtraction.FEATURE_METHODS[feature_name]
            try:
                required_data = await cell.load_data(feature_info["data_type"])
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


def process_file_worker(task):
    """
    Worker function to process individual files.
    """
    global global_semaphore, global_log_queue  # Use the global semaphore
    global_semaphore.acquire()
    try:

        (
            file_path,
            resistance_label,
            dish_number,
            directories,
            processed_features,
            csv_lock,
        ) = task

        result = asyncio.run(
            process_cell(
                file_path,
                resistance_label,
                dish_number,
                directories,
                processed_features,
                csv_lock,
                )
            )
        return result
    except Exception as e:
        logging.error(f"Worker encountered an error: {e}", exc_info=True)
        return {"file_path": file_path, "radiation_resistance": "Unknown", "dish_number": "Unknown", "error": str(e)}
    finally:    
        global_semaphore.release()
        cp.get_default_memory_pool().free_all_blocks() 
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

def restart_pool_ungracefully(num_workers):
    # not graceful, emergency hard cut for memory followed by restart 

    global pool
    if pool is not None:
        pool.terminate()  # Safely terminate the existing pool
        pool.join()       # Wait for the pool's worker processes to exit
    pool = Pool(processes=num_workers, maxtasksperchild=max_tasks_per_child)  # Create a new pool with fewer workers
    logging.info(f"Pool restarted with {num_workers} workers.")

def recreate_pool(worker_count):
    global pool
    if pool is not None:
        pool.close()
        pool.join()
    pool = Pool(processes=worker_count)

def update_ui(progress_layout, usage, current_workers):
    cpu_usage = usage["cpu"]
    panel_text = (
        f"CPU || Workers: {current_workers.value}/{max_workers}"
        f" | Utilization: {cpu_usage['cpu_percent']:.1f}%"
        f" | Memory: {cpu_usage['memory_percent']:.1f}%"
    )
    for key, gpu in usage.items():
        if "gpu" in key:
            panel_text += (
                f" || GPU: Utilization: {gpu['percent']:.1f}%"
                f" | Memory: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB"
                f" | Temperature: {gpu['temperature']:.1f}°C"
            )
    monitor_panel = Panel(panel_text)
    progress_layout["monitor"].update(monitor_panel)

def resource_manager(progress_layout, current_workers, semaphore, worker_lock, stop_event):
    """
    Monitors system resources, updates the Rich layout, and throttles workers based on memory usage.
    """
    upper_threshold, lower_threshold, critical_threshold = memory_thresholds
    restart_counter = 0

    while not stop_event.is_set():
        try:
            usage = file_utils.get_resource_usage()
            cpu_usage = usage["cpu"]
            cpu_memory_usage = cpu_usage["memory_percent"]

            try:
                # Update worker counts based on memory usage
                with worker_lock:
                    if cpu_memory_usage > critical_threshold:
                        logging.warning(f"Memory above critical threshold ({critical_threshold}%). Restarting pool... (Restart #{restart_counter})")
                        restart_counter += 1
                        semaphore.acquire()  # Block new tasks
                        current_workers.value = max(min(current_workers.value - 1, max_workers),1)
                        restart_pool_ungracefully(current_workers.value)
                        semaphore.release()  # Allow tasks again

                    elif cpu_memory_usage > upper_threshold and current_workers.value > 1:
                        current_workers.value -= 1  # Decrease workers
                        semaphore.acquire()
                        recreate_pool(current_workers.value)

                    elif cpu_memory_usage < lower_threshold and current_workers.value < max_workers:
                        current_workers.value += 1  # Increase workers
                        semaphore.release()
                        recreate_pool(current_workers.value)

            except:
                logging.error("Unable to update current_workers", exc_info=True)
            update_ui(progress_layout, usage, current_workers)
        except Exception as e:
            logging.error(f"Error in resource manager: {e}", exc_info=True)
            break

        finally:
            time.sleep(resource_check_frequency)  # Delay to avoid tight looping

def populate_queue(base_dir, resistance_mapping, task_queue, completely_processed_files, processed_features_for_each_file, csv_lock):
    """Populate a multiprocessing queue with all file processing tasks."""
    
    for resistance_folder in resistance_mapping.keys():
        resistance_path = os.path.join(base_dir, resistance_folder)
        if not os.path.isdir(resistance_path):
            continue
        for dish_folder in os.listdir(resistance_path):
            dish_path = os.path.join(resistance_path, dish_folder)
            if os.path.isdir(dish_path) and dish_folder.startswith("dish"):
                resistance_label, dish_number = file_utils.get_resistance_label_and_dish(
                    resistance_mapping, resistance_path, dish_folder
                )
            directories = file_utils.get_output_directories(dish_path)
            files_in_directory = [os.path.join(dish_path, filename) for filename in os.listdir(dish_path)]
            completely_processed_files = set(completely_processed_files)

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
            for file_path in files:
                task_queue.put(
                    (
                    file_path, 
                    resistance_label, 
                    dish_number, 
                    directories,
                    processed_features_for_each_file,
                    csv_lock
                    )
                )

def process_directory(base_dir: str, output_csv_path: str, csv_lock, log_queue) -> None:
    """
    Process all resistance folders and dishes in the base directory and save results to a CSV file.
    Initializes and manages processed files and features to avoid redundant processing.
    """
     # Setup multiprocessing managers for state sharing across processes
    manager = Manager()
    task_queue = manager.Queue()
    stop_event = threading.Event()
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
        file_utils.reset_processing_environment(dataset_location, files_with_errors, output_csv_path)

    total_files = file_utils.count_total_files(base_dir, resistance_mapping)
    already_processed = len(completely_processed_files)
    
    try:
        populate_queue(base_dir, resistance_mapping, task_queue, completely_processed_files, processed_features_for_each_file, csv_lock)
    except:
        logging.error("Error populating queue", exc_info=True)
        
    # Define progress bar
    progress = Progress(
        TextColumn("[bold blue]OVERALL PROGRESS:"),
        TextColumn(" | [cyan]Cell Line: {task.fields[current_cell_line]}"),
        TextColumn(" | [cyan]Dish: {task.fields[current_dish]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.completed}/{task.total} files"),
        TextColumn(" | [green]{task.fields[seconds_per_file]:.2f} sec/file")
    )
    progress_task = progress.add_task("Processing Files", total=total_files, completed=already_processed, seconds_per_file=0.0, current_dish="Starting...", current_cell_line="Starting...")

    # Define rich layout
    progress_layout = Layout()
    progress_layout.split(
        Layout(name="monitor", size=3),
        Layout(progress, name="progress", ratio=1)
    )
    current_workers = manager.Value('i', initial_workers)
    semaphore = Semaphore(max_workers)
    resource_manager_thread = Thread(
        target=resource_manager,
        args=(
            progress_layout, 
            current_workers,  
            semaphore, 
            worker_lock, 
            stop_event),
        daemon=True
    )
    resource_manager_thread.start()
    start_time = time.time()  # Record the start time
    with Live(progress_layout, refresh_per_second=1):
        try:
            with Pool(
                processes=initial_workers, 
                maxtasksperchild=max_tasks_per_child, 
                initializer=worker_initializer, 
                initargs=(semaphore,log_queue, ENABLE_LOGGING)
            ) as pool:
                while not task_queue.empty():
                    chunk = [task_queue.get() for _ in range(min(queue_chunk_size, task_queue.qsize()))]

                    for result in pool.imap_unordered(process_file_worker, chunk):
                        # Update progress bar in the main process
                        elapsed_time = time.time() - start_time
                        completed_files = progress.tasks[progress_task].completed + 1
                        seconds_per_file = elapsed_time / completed_files if completed_files > 0 else 0.0

                        progress.update(
                            progress_task,
                            advance=1,
                            seconds_per_file=seconds_per_file,
                            current_cell_line=result.get("radiation_resistance", "Unknown"),
                            current_dish=result.get("dish_number", "Unknown"),
                        )

            logging.info("All tasks completed.")
        except:
            logging.error("Error with pool", exc_info=True)
        finally:
            pool.close()
            pool.join()
            stop_event.set() # stop threads
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            resource_manager_thread.join()
            os.system("cls" if os.name == "nt" else "clear") # clear terminal