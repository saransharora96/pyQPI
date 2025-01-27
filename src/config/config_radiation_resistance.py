import os 

INSTALLATION_NEEDED = False
RESUME_PROCESSING = True
ENABLE_LOGGING = True
DEBUG_MODE = False # Removes large files and multiprocessing

# Tomocube Instrument Parameters
wavelength = 532e-9       # wavelength (m)
alpha = 0.2               # refractive index increment (um^3/pg)
pixel_x = 0.095           # pixel size (um)
pixel_y = 0.095
pixel_z = 0.19
background_ri = 1.337
bottom_threshold_factor = 0.75
chunked_save_threshold = 1*1000*1000*1000
chunk_size = 10

resistance_mapping = {
    "cell_line_1": "Sensitive",
    "cell_line_2": "Resistant",
    "cell_line_3": "Intermediate"
}

log_file_path = "../pyQPI/src/logs/log_file.log"
files_with_errors = "../pyQPI/src/logs/files_with_errors_skipped_during_processing.log"

# dataset_location = (
#     r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\OracleQPI\pyQPI\data"
# )
# dataset_location = (
#     "../pyQPI/data"
# )
dataset_location = (
    "../../../mnt/g/radiation_resistance_dataset_export/"
)
disk_mount = '/mnt/g'

output_csv_path = os.path.join(dataset_location, "extracted_parameters.csv")

memory_threshold = 50 # percent
max_tasks_per_child = 3
resource_check_frequency = 3 #seconds
queue_chunk_size = 20
if DEBUG_MODE:
    max_workers = 1
    initial_workers = 1
    max_large_tasks = 0
else:
    max_workers = 20
    initial_workers = 4
    max_large_tasks = 1

reserved_workers_for_large_tasks = [1] # avoid 0, otherwise it messes with DEBUG mode

# Nikon Brillouin has 128 GB RAM, 12 CPU cores, 16 GB GPU RAM
# HT-X has 62.65 GB RAM, 20 CPU cores, 24 GB GPU RAM