import os
import logging
import time
from time import sleep
import pandas as pd
from typing import Any, Dict
import cupy as cp
import gc
import asyncio
from multiprocessing import Pool, Manager, Semaphore, Value
from threading import Thread
import threading
from collections import deque
from collections import deque
from logging.handlers import QueueHandler
from rich.live import Live
from classes.Cell import Cell
from classes.AuxiliaryDataGeneration import AuxiliaryDataGeneration
from classes.FeatureExtraction import FeatureExtraction
import utils.dir_utils as file_utils
from resource_management import resource_manager
from progress_bar import initialize_progress_bar, update_progress_bar
from progress_logging import log_result 
from config.config_radiation_resistance import (
    background_ri,
    pixel_x,
    wavelength,
    resistance_mapping,
    files_with_errors,
    dataset_location,
    max_workers,
    initial_workers,
    max_tasks_per_child,
    queue_chunk_size,
    max_large_tasks,
    RESUME_PROCESSING,
    ENABLE_LOGGING
)


def worker_initializer(semaphore, log_queue, active_workers, worker_lock, active_large_tasks, ENABLE_LOGGING):
    """
    Initialize global semaphore for workers.
    """
    global global_semaphore, global_log_queue, global_active_workers, global_worker_lock, global_active_large_tasks
    global_semaphore = semaphore
    global_log_queue = log_queue
    global_active_workers = active_workers 
    global_worker_lock = worker_lock
    global_active_large_tasks = active_large_tasks

    # Clear existing logging handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if ENABLE_LOGGING:
        # Set up logging to pass logs to the main process
        queue_handler = QueueHandler(log_queue)
        logging.root.addHandler(queue_handler)
        logging.root.setLevel(logging.DEBUG)
        logging.debug("Worker initialized with logging enabled.")
    else:
        # Suppress logging
        logging.root.setLevel(logging.CRITICAL)
        logging.debug("Worker initialized with logging suppressed.")

async def process_cell(file_path: str, resistance_label: str, dish_number: int,
                 directories: Dict[str, str], processed_features: Dict[str, Any], csv_lock, worker_index: int):
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

    except Exception as e:
        logging.error(f"Error processing cell {file_path}: {e}", exc_info=True)
        results["error"] = str(e)
        log_result(file_path, results, csv_lock, worker_index)
        return results
    
    finally:
        cell.unload_all_data()
        del cell  # Explicitly delete the object
        cp.get_default_memory_pool().free_all_blocks()  # Free CuPy memory blocks
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()  # Trigger garbage collection
        log_result(file_path, results, csv_lock, worker_index)
        return results

def process_file_worker(task, worker_index=-1):
    """
    Worker function to process individual files.
    """
    global global_semaphore, global_log_queue, global_worker_lock  # Use the global semaphore

    task_id = id(task)
    file_path, resistance_label, dish_number, directories, processed_features, csv_lock = task

    try:
        global_semaphore.acquire()
        
        logging.debug(f"[Task {task_id}] Worker {worker_index} started.")
        start_time = time.time()

        _ = asyncio.run(process_cell(file_path, resistance_label, dish_number, directories, processed_features, csv_lock, worker_index))
        elapsed_time = time.time() - start_time
        return {
            "task_id": task_id,
            "worker_index": worker_index,
            "file_path": file_path,
            "dish_number": dish_number,
            "radiation_resistance": resistance_label,
            "elapsed_time": elapsed_time
        }
    
    except Exception as e:
        logging.error(f"[Task {task_id}] Worker encountered an error: {e}", exc_info=True)
        return {
            "task_id": task_id,
            "worker_index": worker_index,
            "file_path": task[0] if isinstance(task, tuple) else None,
            "dish_number": task[2] if isinstance(task, tuple) else None,
            "error": str(e)
        }
    
    finally:    
        global_semaphore.release()
        cp.get_default_memory_pool().free_all_blocks() 
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        logging.debug(f"[Task {task_id}] Worker {worker_index} finished in {elapsed_time:.2f} seconds.")

def populate_queue(base_dir, resistance_mapping, task_queue, completely_processed_files, processed_features_for_each_file, csv_lock):
    """Populate a multiprocessing queue with all file processing tasks."""
    
    tasks = []
    tasks = []
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
                file_size = os.path.getsize(file_path)
                tasks.append((file_size,(file_path, resistance_label, dish_number, directories, processed_features_for_each_file, csv_lock)))
    tasks.sort(reverse=True, key=lambda x: x[0])
    task_queue.extend([task for _, task in tasks]) 

def process_directory(base_dir: str, output_csv_path: str, csv_lock, log_queue) -> None:

    # Setup multiprocessing managers for state sharing across processes
    manager = Manager()
    task_queue = manager.Queue()
    stop_event = threading.Event()
    worker_lock = manager.Lock()
    active_workers = manager.Value('i', 0)  # Tracks the number of active workers
    paused = manager.Value('b', False)
    processed_features_for_each_file = manager.dict()

    completely_processed_files = list()  # Files where all features are processed
    files_in_csv = set()

    if RESUME_PROCESSING and os.path.exists(output_csv_path):
        existing_data = pd.read_csv(output_csv_path)
        files_in_csv.update(existing_data['file_path'].tolist())
        all_features = set(FeatureExtraction.FEATURE_METHODS.keys())  # Define all possible features

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
    progress, progress_layout, progress_task = initialize_progress_bar(total_files, already_processed)

    current_workers = manager.Value('i', initial_workers)
    semaphore = Semaphore(max_workers)
    resource_manager_thread = Thread(
        target=resource_manager,
        args=(
            progress_layout,
            current_workers,
            active_workers,
            semaphore,
            worker_lock,
            paused,
            stop_event
        ),
        daemon=True
    )
    resource_manager_thread.start()

    try:
        task_queue = deque()
        populate_queue(base_dir, resistance_mapping, task_queue, completely_processed_files, processed_features_for_each_file, csv_lock)
    except:
        logging.error("Error populating queue", exc_info=True)

    session_start_time = time.time()
    session_file_count = 0
  
    active_large_tasks = manager.Value('i', 0)
    reserved_workers = [0, 4, 8]
    active_worker_cores = manager.dict({i: False for i in range(current_workers.value)})

    with Live(progress_layout, refresh_per_second=1):
        try:
            with Pool(
                processes=initial_workers,
                maxtasksperchild=max_tasks_per_child,
                initializer=worker_initializer,
                initargs=(semaphore, log_queue, active_workers, worker_lock, active_large_tasks, ENABLE_LOGGING)
            ) as pool:
                # List to track async results of large tasks
                large_task_results = []
                
                while task_queue or large_task_results:
                    if paused.value:
                        sleep(1)
                        continue

                    # Assign large tasks only when memory usage is low and fewer than 2 large tasks are active
                    large_tasks = []
                    with worker_lock:  # Use the shared lock
                        if active_large_tasks.value < max_large_tasks:
                            while not paused.value and task_queue and len(large_tasks) < (max_large_tasks - active_large_tasks.value):
                                large_tasks.append(task_queue.popleft())

                    if large_tasks:
                        logging.debug(f"Submitting {len(large_tasks)} large tasks.")

                        for task in large_tasks:
                            # Find the first free reserved worker
                            free_worker = next(
                                (core for core in reserved_workers if not active_worker_cores.get(core, False)),
                                None
                            )
                            if free_worker is not None:
                                with worker_lock:
                                    active_worker_cores[free_worker] = True  # Mark reserved worker as active
                                    active_large_tasks.value += 1  # Increment active large task count
                                    active_workers.value += 1
                                
                                logging.debug(f"Assigning large task {id(task)} to reserved worker {free_worker}")

                                # Submit the large task
                                async_result = pool.apply_async(
                                    process_file_worker,
                                    args=(task,),
                                    kwds={"worker_index": free_worker}
                                )
                                large_task_results.append(async_result)
                                session_file_count += 1
                            else:
                                logging.warning("No reserved workers available for large tasks.")

                    # Check for completion of large tasks
                    for async_result in large_task_results[:]:
                        if async_result.ready():  # Non-blocking check
                            result = async_result.get()  # Retrieve the result
                            large_task_results.remove(async_result)
                            with worker_lock:
                                # Mark the reserved worker as free
                                worker_index = result.get("worker_index", None)
                                if worker_index in reserved_workers:
                                    active_worker_cores[worker_index] = False  # Free the reserved worker
                                    logging.debug(f"Worker {worker_index} is now free for large tasks.")
                                active_large_tasks.value -= 1  # Decrement active large task count
                                active_workers.value -= 1
                                logging.debug(f"Active large tasks decremented: {active_large_tasks.value}")
                            update_progress_bar(progress, progress_task, session_start_time, session_file_count, result)

                    if active_workers.value < current_workers.value:  # Use remaining workers
                        # Submit smaller tasks for other workers
                        small_tasks = []
                        while (
                            not paused.value and task_queue 
                            and len(small_tasks) < (current_workers.value - active_workers.value) 
                            and len(small_tasks) < queue_chunk_size
                            and len(small_tasks) < max_workers-max_large_tasks
                        ):
                            small_tasks.append(task_queue.pop())  # Get the smallest task
                        
                        if small_tasks:
                            logging.debug(f"Submitting {len(small_tasks)} small tasks.")
                            for task in small_tasks:
                                # Find the first free, non-reserved worker
                                free_worker = next(
                                    (core for core, active in active_worker_cores.items()
                                    if not active and core not in reserved_workers),
                                    None
                                )
                                if free_worker is not None:
                                    with worker_lock:
                                        active_worker_cores[free_worker] = True  # Mark worker as active
                                        active_workers.value += 1

                                    logging.debug(f"Assigning small task {id(task)} to worker {free_worker}")
                                    
                                     # Submit the task with a callback to update the progress bar
                                    def task_complete_callback(result):
                                        with worker_lock:
                                            worker_index = result.get("worker_index")
                                            if worker_index is not None:
                                                active_worker_cores[worker_index] = False
                                                active_workers.value -= 1
                                                logging.debug(f"Worker {worker_index} is now free. Active workers decremented to: {active_workers.value}")
                                            else:
                                                logging.error("Callback result missing 'worker_index'.")
                                        update_progress_bar(progress, progress_task, session_start_time, session_file_count, result)

                                    # Submit the task
                                    async_result = pool.apply_async(
                                        process_file_worker,
                                        args=(task,),
                                        kwds={"worker_index": free_worker},
                                        callback=task_complete_callback
                                    )
                                    session_file_count += 1
                                else:
                                    logging.debug("No free workers available for small tasks.")
                                    break
                                    
                                # Allow some time for tasks to progress (more efficient for CPU)
                                sleep(0.1)

            logging.info("All tasks completed.")
        except:
            logging.error("Error with pool", exc_info=True)
        finally:
            pool.close()
            pool.join()
            stop_event.set()  # stop threads
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            resource_manager_thread.join()
            os.system("cls" if os.name == "nt" else "clear")  # clear terminal
