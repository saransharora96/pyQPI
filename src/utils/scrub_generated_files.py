from utils.dir_utils import remove_directories


if __name__ == "__main__":
    # Define the base directory and directories to remove
    BASE_DIR = r"/data"
    generated_dir = ["MIP", "MIP_scaled", "Phase", "Phase_scaled", "Image_binary_mask", "MIP_segmented",
                     "MIP_segmented_scaled", "Tomogram_binary_mask", "Tomogram_segmented"]

    # Remove directories
    remove_directories(BASE_DIR, generated_dir)
