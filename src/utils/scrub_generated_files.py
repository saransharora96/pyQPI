from utils.dir_utils import remove_directories
from config.config_radiation_resistance import dataset_location


if __name__ == "__main__":
    # Define the base directory and directories to remove

    generated_dir = ["MIP", "MIP_scaled", "Phase", "Phase_scaled", "Image_binary_mask", "MIP_segmented",
                     "MIP_segmented_scaled", "Tomogram_binary_mask", "Tomogram_segmented"]

    # Remove directories
    remove_directories(dataset_location, generated_dir)
