import time
from src.install_modules import upgrade_pip, install_packages
from src.config.config_radiation_resistance import INSTALLATION_NEEDED
from processing import process_directory
import logging


# Configure logging
logging.basicConfig(
    filename="../../src/logs/processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":

    start_time = time.time()  # Record the start time

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    dataset_location = (
        r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\OracleQPI\pyQPI\data"
    )

    # test_cell =Cell(dataset_location,'resistant',1)
    #
    # test_tomogram = test_cell.read_tomogram()
    # plt.imshow(test_tomogram[110, :, :], cmap='gray')  # Visualize the 11th slice
    # plt.colorbar()
    # plt.show()
    #
    # dry_mass = test_cell.calculate_dry_mass(backgroundRI, alpha, pixel_x, pixel_y, pixel_z)
    # print(f"Dry Mass: {dry_mass} picograms")

    logging.info("Starting processing...")

    try:
        process_directory(
            base_dir=dataset_location,
            output_csv="extracted_parameters.csv",
        )
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error(f"Processing failed with error: {e}")
        raise

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")




