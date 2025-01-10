import os
import cupy as cp
import h5py
from tifffile import imwrite
from typing import Dict, Any
from classes.Segmentation import Segmentation, process_tomogram
from classes.FeatureExtraction import FeatureExtraction
from memory_profiler import profile


class AuxiliaryDataGeneration:
    def __init__(self, cell: Any, directories: Dict[str, str], pixel_x: float, wavelength: float, background_ri: float):
        """
        Initializes the AuxiliaryDataGenerator with all necessary parameters.

        Args:
            cell (Any): The cell object containing the tomogram and metadata.
            directories (Dict[str, str]): Dictionary specifying where each type of output should be stored.
            pixel_x (float): Pixel size in micrometers.
            wavelength (float): Wavelength of light used.
            background_ri (float): Background refractive index.
        """
        self.cell = cell
        self.directories = directories
        self.pixel_x = pixel_x
        self.wavelength = wavelength
        self.background_ri = background_ri

        # Ensure all directories exist
        for directory in directories.values():
            self.ensure_directory_exists(directory)

    @staticmethod
    def ensure_directory_exists(directory):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def save_image(data, directory, filename, scale=False):
        """Save an image, optionally scaling it to [0, 1]."""
        path = os.path.join(directory, f"{filename}.tiff")
        if scale:
            data = (data - data.min()) / (data.max() - data.min())

        # Convert CuPy array to NumPy array explicitly
        data = data.get()
        imwrite(path, data.astype(cp.float32))
        return path
    
    @staticmethod
    def save_h5(data, directory, filename):
        """Save data to an HDF5 file."""
        path = os.path.join(directory, f"{filename}.h5")
        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("data", data=data.get(), dtype='float32', compression="gzip")
        return path

    def generate_and_save_auxiliary_data(self):
        """Generates and saves all required auxiliary data based on the cell's tomogram."""
        try:
            # Load the tomogram from the cell
            tomogram = self.cell.load_tomogram()
            base_name = os.path.splitext(os.path.basename(self.cell.tomogram_path))[0]

            # Process tomogram to generate binary masks and segmented images
            segmenter = Segmentation(offset=-0.005)
            tomogram_binary_mask = process_tomogram(tomogram, segmenter)
            tomogram_binary_mask_path = self.save_h5(tomogram_binary_mask, self.directories['tomogram_binary_mask'], f"{base_name}_tomogram_binary_mask")
            self.cell.add_auxiliary_data_path('tomogram_binary_mask', tomogram_binary_mask_path)

            # Generate and save 2D binary mask
            binary_mask_image = FeatureExtraction.generate_mip(tomogram_binary_mask)
            binary_mask_image_path = self.save_image(binary_mask_image, self.directories['image_binary_mask'], f"{base_name}_image_binary_mask")
            self.cell.add_auxiliary_data_path('image_binary_mask', binary_mask_image_path)

            # Segment the tomogram
            tomogram_segmented = Segmentation.apply_binary_mask(tomogram, tomogram_binary_mask)
            tomogram_segmented_path = self.save_h5(tomogram_segmented, self.directories['tomogram_segmented'], f"{base_name}_tomogram_segmented")
            self.cell.add_auxiliary_data_path('tomogram_segmented', tomogram_segmented_path)

            # Generate and save MIP
            mip = FeatureExtraction.generate_mip(tomogram)
            mip_path = self.save_h5(mip, self.directories['mip'], f"{base_name}_mip")
            self.cell.add_auxiliary_data_path('mip', mip_path)

            # #Generate and save scaled MIP
            # mip_scaled_path = self.save_image(mip, self.directories['mip_scaled'], f"{base_name}_mip_scaled", scale=True)
            # self.cell.add_auxiliary_data_path('mip_scaled', mip_scaled_path)

            # Generate and save segmented MIP
            segmented_mip = FeatureExtraction.generate_mip(tomogram_segmented)
            segmented_mip_path = self.save_h5(segmented_mip, self.directories['mip_segmented'], f"{base_name}_mip_segmented")
            self.cell.add_auxiliary_data_path('mip_segmented', segmented_mip_path)
            del segmented_mip  # Free memory

            # Generate and save scaled segmented MIP
            mip_segmented_scaled = Segmentation.apply_binary_mask((mip-mip.min())/(mip.max()-mip.min()), binary_mask_image)
            mip_segmented_scaled_path = self.save_image(mip_segmented_scaled, self.directories["mip_segmented_scaled"], f"{base_name}_segmented_MIP_scaled")
            self.cell.add_auxiliary_data_path('mip_segmented_scaled', mip_segmented_scaled_path)
            del mip_segmented_scaled  # Free memory

            # # Generate and save phase shift
            # phase_shift = FeatureExtraction.generate_phase_delay_image(tomogram, self.pixel_x, self.wavelength, self.background_ri)
            # phase_shift_path = self.save_image(phase_shift, self.directories['phase'], f"{base_name}_phase_shift")
            # self.cell.add_auxiliary_data_path('phase_shift', phase_shift_path)
            # phase_shift_scaled_path = self.save_image(phase_shift, self.directories['phase_scaled'], f"{base_name}_phase_shift_scaled", scale=True)
            # self.cell.add_auxiliary_data_path('phase_shift_scaled', phase_shift_scaled_path)

            del tomogram
            del tomogram_segmented
            del tomogram_binary_mask
            del mip
            del binary_mask_image  # Free memory
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            print(f"Error during auxiliary data generation: {e}")
        
        finally:
            # Clean up memory by unloading loaded data
            self.cell.unload_all_data()