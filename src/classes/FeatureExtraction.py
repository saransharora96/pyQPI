import numpy as np

class FeatureExtraction:

    @staticmethod
    def calculate_dry_mass(tomogram, background_ri, alpha, pixel_x, pixel_y, pixel_z):
        """
        Calculate the dry mass of the cell using float32 precision and rounds to two decimal places,
        optimized for memory efficiency.

        Args:
            tomogram (numpy.ndarray): 3D tomogram.
            background_ri (float): The background refractive index.
            alpha (float): Calibration factor.
            pixel_x, pixel_y, pixel_z (float): Pixel dimensions in microns.

        Returns:
            float: The dry mass of the cell in picograms, rounded to two decimal places.
        """
        # Convert tomogram to float32, reducing precision for memory savings
        tomogram = tomogram.astype(np.float32, copy=False)
        tomogram -= np.float32(background_ri)
        tomogram /= np.float32(alpha)
        tomogram.clip(min=0, out=tomogram)

        # Calculate the voxel volume in float32 to maintain consistent precision
        voxel_volume = np.float32(pixel_x * pixel_y * pixel_z)

        # Sum the mass, multiply by the voxel volume, round and return the result
        total_mass = np.sum(tomogram) * voxel_volume
        return np.round(total_mass, 2)

    @staticmethod
    def generate_mip(tomogram: np.ndarray):
        """
        Generate a Maximum Intensity Projection (MIP) along the Z-axis using float32.

        Args:
            tomogram (numpy.ndarray): 3D tomogram.

        Returns:
            numpy.ndarray: 2D MIP image.
        """
        return np.max(tomogram.astype(np.float32, copy=False), axis=0)

    @staticmethod
    def generate_phase_delay_image(tomogram: np.ndarray, pixel_size: float, wavelength: float, medium_ri: float):
        """
        Generate a phase delay image from the tomogram for QPI using float32 precision.

        Args:
            tomogram (numpy.ndarray): 3D tomogram.
            pixel_size (float): The size of the pixel in micrometers.
            wavelength (float): The wavelength of light used in micrometers.
            medium_ri (float): The refractive index of the medium.

        Returns:
            numpy.ndarray: The phase delay image.
        """
        tomogram = tomogram.astype(np.float32, copy=False)
        phase_shift = (2 * np.pi / np.float32(wavelength)) * np.sum((tomogram - np.float32(medium_ri)) * np.float32(pixel_size), axis=0)
        return phase_shift
