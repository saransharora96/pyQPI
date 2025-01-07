import subprocess
import xml.etree.ElementTree as ET

def parse_nvidia_smi_output():
    try:
        # Run nvidia-smi command to get GPU details in XML format
        smi_output = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True)
        smi_xml = smi_output.stdout
        
        # Parse the XML output
        root = ET.fromstring(smi_xml)
        gpu_info_str = ""
        
        # Extract information for each GPU
        for gpu in root.findall('gpu'):
            name = gpu.find('product_name').text
            memory_total = gpu.find('fb_memory_usage/total').text
            memory_free = gpu.find('fb_memory_usage/free').text
            temperature = gpu.find('temperature/gpu_temp').text
            load = gpu.find('utilization/gpu_util').text

            gpu_details = f"""
            GPU: {name}
            Total RAM: {memory_total}
            Available RAM: {memory_free}
            Temperature: {temperature}
            Load: {load}
            """
            gpu_info_str += gpu_details

        return gpu_info_str if gpu_info_str else 'No GPU available'
    except Exception as e:
        return f"Failed to get GPU details: {str(e)}"

# Usage
gpu_details = parse_nvidia_smi_output()
print(gpu_details)
