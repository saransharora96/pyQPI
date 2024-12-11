import time
from src.install_modules import upgrade_pip, install_packages
from src.config.config_extract_parameters import INSTALLATION_NEEDED

# class definitions

# functions

if __name__ == "__main__":

    start_time = time.time()  # Record the start time

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    dataset_location = (
        r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\OracleQPI\code\data"
    )

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")