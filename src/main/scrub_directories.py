from src.utils.dir_utils import remove_directories


if __name__ == "__main__":
    # Define the base directory and directories to remove
    BASE_DIR = "D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\OracleQPI\pyQPI\data"
    DIRECTORIES_TO_REMOVE = ["MIP", "MIP_scaled", "Phase", "Phase_scaled"]

    # Remove directories
    remove_directories(BASE_DIR, DIRECTORIES_TO_REMOVE)
