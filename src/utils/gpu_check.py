import pycuda.driver as cuda
import pycuda.autoinit  # This is necessary for initializing CUDA driver
from GPUtil import getGPUs

def gpu_info():
    # Check for GPUs and gather their information
    gpus = getGPUs()
    gpu_info_str = ""

    # Iterate through all GPUs found by GPUtil
    for gpu in gpus:
        gpu_details = f"""
        GPU: {gpu.name}
        Total RAM: {round(gpu.memoryTotal / 1024, 2)} GB
        Available RAM: {round(gpu.memoryFree / 1024, 2)} GB
        Temperature: {gpu.temperature} °C
        Load: {round(gpu.load * 100, 2)}%
        """
        gpu_info_str += gpu_details

    # Check if CUDA is installed
    try:
        cuda.init()
        is_cuda_installed = "Yes"
        cuda_version = f"{cuda.get_version()[0]}.{cuda.get_version()[1]}.{cuda.get_version()[2]}"
    except cuda.Error:
        is_cuda_installed = "No"
        cuda_version = "N/A"

    # Prepare the final formatted information
    final_output = f"""
    CUDA Installed: {is_cuda_installed}
    CUDA Version: {cuda_version}
    GPU Details: 
    {gpu_info_str if gpu_info_str else 'No GPU available'}
    """
    return final_output

# Usage
gpu_details = gpu_info()
print(gpu_details)
