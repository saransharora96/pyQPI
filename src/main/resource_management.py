import logging
import subprocess
import psutil
import time
from multiprocessing import Pool 
from config.config_radiation_resistance import max_workers, max_tasks_per_child, memory_threshold, resource_check_frequency
from rich.panel import Panel

pool = None

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

def update_ui(progress_layout, usage, current_workers, active_workers):
    cpu_usage = usage["cpu"]
    panel_text = (
        f"CPU || Workers: {active_workers}/{current_workers} active"
        f" | Utilization: {cpu_usage['cpu_percent']:.1f}%"
        f" | Memory: {cpu_usage['memory_percent']:.1f}%"
    )
    for key, gpu in usage.items():
        if "gpu" in key:
            panel_text += (
                f" // GPU: || Utilization: {gpu['percent']:.1f}%"
                f" | Memory: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} MB"
                f" | Temperature: {gpu['temperature']:.1f}°C"
            )
    monitor_panel = Panel(panel_text)
    progress_layout["monitor"].update(monitor_panel)

def resource_manager(progress_layout, current_workers, active_workers, semaphore, worker_lock, paused, stop_event):
    """
    Monitors system resources, updates the Rich layout, and throttles workers based on memory usage.
    Pauses new task submission if memory usage exceeds a critical threshold until memory usage decreases.
    """

    while not stop_event.is_set():
        try:
            usage = get_resource_usage()
            cpu_usage = usage["cpu"]
            cpu_memory_usage = cpu_usage["memory_percent"]

            if cpu_memory_usage > memory_threshold and not paused.value:
                with worker_lock:
                    paused.value = True
                logging.warning(f"Memory above threshold ({memory_threshold}%). Pausing new tasks.")

            elif paused.value and cpu_memory_usage <= memory_threshold:
                with worker_lock:
                    paused.value = False
                logging.info(f"Memory below threshold ({memory_threshold}%). Resuming new tasks.")
    
            # Update UI
            update_ui(
                progress_layout,
                usage,
                current_workers=current_workers.value,
                active_workers=active_workers.value
            )

            # Wait during pause without blocking semaphore unnecessarily
            if paused.value:
                logging.info("Paused: Waiting for memory to stabilize.")
                time.sleep(1)

        except Exception as e:
            logging.error(f"Error in resource manager: {e}", exc_info=True)
            break

        finally:
            time.sleep(resource_check_frequency)  # Delay to avoid tight looping

