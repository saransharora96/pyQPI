import cupy as cp
from config.config_radiation_resistance import background_ri, alpha, pixel_x, pixel_y, pixel_z, wavelength

class FeatureExtraction:

    @staticmethod
    def calculate_dry_mass(tomogram, background_ri, alpha, pixel_x, pixel_y, pixel_z):
        """
        Calculate the dry mass of the cell using float32 precision and rounds to two decimal places,
        optimized for memory efficiency.

        Args:
            tomogram (cupy.ndarray): 3D tomogram.
            background_ri (float): The background refractive index.
            alpha (float): Calibration factor.
            pixel_x, pixel_y, pixel_z (float): Pixel dimensions in microns.

        Returns:
            float: The dry mass of the cell in picograms, rounded to two decimal places.
        """
        # Ensure tomogram is a CuPy array
        tomogram = cp.asarray(tomogram, dtype=cp.float32)
        tomogram -= cp.float32(background_ri)
        tomogram /= cp.float32(alpha)
        tomogram = cp.clip(tomogram, 0, None)

        # Calculate the voxel volume in float32
        voxel_volume = cp.float32(pixel_x * pixel_y * pixel_z)
        total_mass = cp.sum(tomogram) * voxel_volume
        del tomogram
        cp.get_default_memory_pool().free_all_blocks()

        return float(cp.asnumpy(cp.round(total_mass, 2)))

    @staticmethod
    def calculate_cell_volume(binary_mask: cp.ndarray, pixel_x: float, pixel_y: float, pixel_z: float) -> float:
        """
        Calculate the total volume of a cell based on a binary tomogram mask.

        Args:
            binary_mask (cupy.ndarray): A 3D binary mask of the tomogram where white pixels (True/1) indicate the cell.
            pixel_x (float): Pixel size in the x-dimension (in micrometers).
            pixel_y (float): Pixel size in the y-dimension (in micrometers).
            pixel_z (float): Pixel size in the z-dimension (in micrometers).

        Returns:
            float: Total volume of the cell in cubic micrometers.
        """
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("The binary mask is empty or None.")

        # Ensure binary mask is a CuPy array and boolean
        binary_mask = cp.asarray(binary_mask, dtype=bool)

        white_pixel_count = cp.sum(binary_mask)  # Count the number of white pixels (True values)         
        voxel_volume = pixel_x * pixel_y * pixel_z  # Calculate the volume of a single voxel        
        total_volume = white_pixel_count * voxel_volume  # Calculate total cell volume

        del binary_mask
        cp.get_default_memory_pool().free_all_blocks()

        return float(cp.asnumpy(total_volume))

    @staticmethod
    def generate_mip(tomogram: cp.ndarray):
        """
        Generate a Maximum Intensity Projection (MIP) along the Z-axis using float32.

        Args:
            tomogram (cupy.ndarray): 3D tomogram.

        Returns:
            cupy.ndarray: 2D MIP image.
        """
        tomogram = cp.asarray(tomogram, dtype=cp.float32)
        mip = cp.max(tomogram, axis=0)
        del tomogram
        cp.get_default_memory_pool().free_all_blocks()
        return mip

    @staticmethod
    def generate_phase_delay_image(tomogram: cp.ndarray, pixel_size: float, wavelength: float, medium_ri: float):
        """
        Generate a phase delay image from the tomogram for QPI using float32 precision.

        Args:
            tomogram (cupy.ndarray): 3D tomogram.
            pixel_size (float): The size of the pixel in micrometers.
            wavelength (float): The wavelength of light used in micrometers.
            medium_ri (float): The refractive index of the medium.

        Returns:
            cupy.ndarray: The phase delay image.
        """
        tomogram = cp.asarray(tomogram, dtype=cp.float32)
        phase_shift = (2 * cp.pi / cp.float32(wavelength)) * cp.sum((tomogram - cp.float32(medium_ri)) * cp.float32(pixel_size), axis=0)
        del tomogram
        cp.get_default_memory_pool().free_all_blocks()
        return phase_shift

    FEATURE_METHODS = {
        "dry_mass": {
            "method": calculate_dry_mass,
            "data_type": "tomogram_segmented",
            "args": [],  # Positional arguments
            "kwargs": {"background_ri": background_ri, "alpha": alpha, "pixel_x": pixel_x, "pixel_y": pixel_y, "pixel_z": pixel_z},
        },
        "cell_volume": {
            "method": calculate_cell_volume,
            "data_type": "tomogram_binary_mask",
            "args": [],
            "kwargs": {"pixel_x": pixel_x, "pixel_y": pixel_y, "pixel_z": pixel_z},
        }
    }
