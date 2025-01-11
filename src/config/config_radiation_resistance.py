import os 

INSTALLATION_NEEDED = False
RESUME_PROCESSING = False
ENABLE_LOGGING = False

# Tomocube Instrument Parameters
wavelength = 532e-9       # wavelength (m)
alpha = 0.2               # refractive index increment (um^3/pg)
pixel_x = 0.095           # pixel size (um)
pixel_y = 0.095
pixel_z = 0.19
background_ri = 1.337

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
    "../../../mnt/f/radiation_resistance_dataset_export/"
)
disk_mount = '/mnt/f'

output_csv_path = os.path.join(dataset_location, "extracted_parameters.csv")

memory_thresholds = [40, 30, 50] # percentages
max_workers = 12
initial_workers = 12
max_tasks_per_child = 5
resource_check_frequency = 5 #seconds
queue_chunk_size = 24