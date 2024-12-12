from tifffile import TiffFile
import numpy as np


class Cell:
    """
    A class to handle data from each cancer cell in the Quantitative Phase Imaging dataset.
    Each data point will likely be a Tomogram, and other generated data from the Tomogram, like a phase image or a maximum intensity projection (MIP)
    In this particular project, we are working with Radiation Resistance in Head & Neck Cancer.
    All methods for single instances of cells and their processing will be included here.
    """

    def __init__(self, tomogram_path: str, radiation_resistance: str, dish_number: int):
        """
        Initialize the data for a cell.

        :param tomogram_path (str): The path to the TIFF file containing the tomogram.
        :param radiation_resistance (str): The radiation resistance of the cell. (sensitive, intermediate, resistant)
        :param dish_number (int): The dish number from which the cell data was collected.
        """
        self.tomogram_path = tomogram_path
        self.tomogram = self.read_tomogram() # only load when needed
        self.radiation_resistance = radiation_resistance
        self.dish_number = dish_number

    def read_tomogram(self):
        """
        Reads the tomogram from the given file path.

        :return: A numpy array representing the tomogram.
        """
        with TiffFile(self.tomogram_path) as tif:
            tomogram = tif.asarray()/10000
        return tomogram

    def calculate_dry_mass(self, background_ri: float, alpha: float, pixel_x: float, pixel_y: float, pixel_z: float) -> float:
        """
        Calculate the dry mass of the cell.

        Args:
            background_ri (float): The background refractive index.
            alpha (float): Calibration factor to convert RI differences to dry mass density.
            pixel_x (float): The size of the pixel in the x-dimension.
            pixel_y (float): The size of the pixel in the y-dimension.
            pixel_z (float): The size of the pixel in the z-dimension.

        Returns:
            float: The dry mass of the cell in picograms.
        """
        data = np.array(self.tomogram)
        data = np.subtract(data, background_ri) # Subtract background RI
        data = np.divide(data, alpha)           # Convert to dry mass
        data = np.multiply(data, pixel_x)       # Scale by pixel size in x
        data = np.multiply(data, pixel_y)       # Scale by pixel size in y
        data = np.multiply(data, pixel_z)       # Scale by pixel size in z
        data[data < 0] = 0                      # Remove negative values
        dry_mass = np.sum(data)                 # Calculate total dry mass
        return dry_mass

    def generate_mip(self):
        """
        Generate a Maximum Intensity Projection (MIP) along the Z-axis from the cell's tomogram.

        Returns:
            numpy.ndarray: A 2D numpy array representing the MIP along the Z-axis.
        """
        if self.tomogram is None:
            raise ValueError("Tomogram data is not loaded.")

        mip = np.max(self.tomogram, axis=0)
        return mip

    def generate_phase_delay_image(self, pixel_size, wavelength, medium_ri):
        """
        Generate a phase delay image from the tomogram for QPI.

        Args:
            pixel_size (float): The size of the pixel in micrometers.
            wavelength (float): The wavelength of light used in micrometers.
            medium_ri (float): The refractive index of the medium.

        Returns:
            numpy.ndarray: The phase delay image.
        """
        if self.tomogram is None:
            raise ValueError("Tomogram data is not loaded.")

        phase_shift = (2 * np.pi / wavelength) * np.sum((self.tomogram - medium_ri) * pixel_size, axis=0)
        return phase_shift
