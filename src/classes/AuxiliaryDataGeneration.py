import os
import cupy as cp
import h5py
from tifffile import imwrite, TiffWriter
from typing import Dict, Any
from classes.Segmentation import Segmentation, process_tomogram
from classes.FeatureExtraction import FeatureExtraction
from memory_profiler import profile
from utils.dir_utils import ensure_cupy_array
import asyncio
import aiofiles
import logging


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
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(self.ensure_all_directories_exist())
        else:
            asyncio.run(self.ensure_all_directories_exist())

    async def ensure_all_directories_exist(self):
        tasks = [self.ensure_directory_exists(directory) for directory in self.directories.values()]
        await asyncio.gather(*tasks)

    async def ensure_directory_exists(self, directory):
        """Asynchronously create a directory if it doesn't exist."""
        loop = asyncio.get_event_loop()
        if not os.path.exists(directory):
            await loop.run_in_executor(None, os.makedirs, directory)

    async def save_image(self, data, directory, filename, scale=False):
        """Save an image asynchronously, optionally scaling it to [0, 1]."""
        path = os.path.join(directory, f"{filename}.tiff")
        try:
            if scale:
                data = (data - data.min()) / (data.max() - data.min())

            data = data.get()  # Convert CuPy array to NumPy
            await asyncio.to_thread(self._sync_save_image, data, path)
            return path
        except Exception as e:
            logging.error(f"Failed to save image at {path}: {e}")
            return None

    def _sync_save_image(self, data, path):
        with TiffWriter(path) as tiff_writer:
            tiff_writer.write(data.astype('float32'))

    async def save_h5(self, data, directory, filename):
        """Save data to an HDF5 file asynchronously."""
        path = os.path.join(directory, f"{filename}.h5")
        try:
            await asyncio.to_thread(self._sync_save_h5, data, path)
            return path
        except Exception as e:
            logging.error(f"Failed to save HDF5 file at {path}: {e}")
            return None

    def _sync_save_h5(self, data, path):
        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("data", data=data.get(), dtype='float32', compression="gzip")


    async def generate_and_save_auxiliary_data(self):
        """Generates and saves all required auxiliary data based on the cell's tomogram."""
        try:
            
            # Load the tomogram from the cell
            tomogram = await self.cell.load_tomogram()
            if tomogram is None or tomogram.size == 0:
                raise ValueError(f"Invalid tomogram: {type(tomogram)}, size={getattr(tomogram, 'size', 'N/A')}")

            base_name = os.path.splitext(os.path.basename(self.cell.tomogram_path))[0]

            # Process tomogram to generate binary masks and segmented images
            try:
                segmenter = Segmentation(offset=-0.005)
                tomogram_binary_mask = process_tomogram(tomogram, segmenter)
                if tomogram_binary_mask is None:
                    raise ValueError(f"process_tomogram returned None for {self.cell.tomogram_path}")
                tomogram_binary_mask_path = await self.save_h5(tomogram_binary_mask, self.directories['tomogram_binary_mask'], f"{base_name}_tomogram_binary_mask")
                if tomogram_binary_mask_path:
                    self.cell.add_auxiliary_data_path('tomogram_binary_mask', tomogram_binary_mask_path)
                else:
                    logging.error(f"Failed to save tomogram_binary_mask for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating tomogram_binary_mask for {self.cell.tomogram_path}: {e}", exc_info=True)
                tomogram_binary_mask_path = await self.save_h5(tomogram_binary_mask, self.directories['tomogram_binary_mask'], f"{base_name}_tomogram_binary_mask")
                if tomogram_binary_mask_path:
                    self.cell.add_auxiliary_data_path('tomogram_binary_mask', tomogram_binary_mask_path)
                else:
                    logging.error(f"Failed to save tomogram_binary_mask for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating tomogram_binary_mask for {self.cell.tomogram_path}: {e}", exc_info=True)

            # Generate and save 2D binary mask
            try:
                binary_mask_image = FeatureExtraction.generate_mip(tomogram_binary_mask)
                binary_mask_image_path = await self.save_image(binary_mask_image, self.directories['image_binary_mask'], f"{base_name}_image_binary_mask")
                if binary_mask_image_path:
                    self.cell.add_auxiliary_data_path('image_binary_mask', binary_mask_image_path)
                else:
                    logging.error(f"Failed to save binary_mask_image for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating binary_mask_image for {self.cell.tomogram_path}: {e}", exc_info=True)

            # Segment the tomogram
            try:
                tomogram_segmented = Segmentation.apply_binary_mask(tomogram, tomogram_binary_mask)
                tomogram_segmented_path = await self.save_h5(tomogram_segmented, self.directories['tomogram_segmented'], f"{base_name}_tomogram_segmented")
                if tomogram_segmented_path:
                    self.cell.add_auxiliary_data_path('tomogram_segmented', tomogram_segmented_path)
                else:
                    logging.error(f"Failed to save tomogram_segmented for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating tomogram_segmented for {self.cell.tomogram_path}: {e}", exc_info=True)

            # Generate and save MIP
            try:
                mip = FeatureExtraction.generate_mip(tomogram)
                mip_path = await self.save_h5(mip, self.directories['mip'], f"{base_name}_mip")
                if mip_path:
                    self.cell.add_auxiliary_data_path('mip', mip_path)
                else:
                    logging.error(f"Failed to save MIP for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating MIP for {self.cell.tomogram_path}: {e}", exc_info=True)

            # #Generate and save scaled MIP
            # mip_scaled_path = self.save_image(mip, self.directories['mip_scaled'], f"{base_name}_mip_scaled", scale=True)
            # self.cell.add_auxiliary_data_path('mip_scaled', mip_scaled_path)

            # Generate and save segmented MIP
            try:
                segmented_mip = FeatureExtraction.generate_mip(tomogram_segmented)
                segmented_mip_path = await self.save_h5(segmented_mip, self.directories['mip_segmented'], f"{base_name}_mip_segmented")
                if segmented_mip_path:
                    self.cell.add_auxiliary_data_path('mip_segmented', segmented_mip_path)
                else:
                    logging.error(f"Failed to save segmented MIP for {self.cell.tomogram_path}")
                del segmented_mip
            except Exception as e:
                logging.error(f"Error generating segmented MIP for {self.cell.tomogram_path}: {e}", exc_info=True)

            # Generate and save scaled segmented MIP
            try:
                mip_segmented_scaled = Segmentation.apply_binary_mask((mip-mip.min())/(mip.max()-mip.min()), binary_mask_image)
                mip_segmented_scaled_path = await self.save_image(mip_segmented_scaled, self.directories["mip_segmented_scaled"], f"{base_name}_segmented_MIP_scaled")
                if mip_segmented_scaled_path:
                    self.cell.add_auxiliary_data_path('mip_segmented_scaled', mip_segmented_scaled_path)
                else:
                    logging.error(f"Failed to save scaled segmented MIP for {self.cell.tomogram_path}")
                del mip_segmented_scaled
            except Exception as e:
                logging.error(f"Error generating scaled segmented MIP for {self.cell.tomogram_path}: {e}", exc_info=True)

            # # Generate and save phase shift
            # phase_shift = FeatureExtraction.generate_phase_delay_image(tomogram, self.pixel_x, self.wavelength, self.background_ri)
            # phase_shift_path = self.save_image(phase_shift, self.directories['phase'], f"{base_name}_phase_shift")
            # self.cell.add_auxiliary_data_path('phase_shift', phase_shift_path)
            # phase_shift_scaled_path = self.save_image(phase_shift, self.directories['phase_scaled'], f"{base_name}_phase_shift_scaled", scale=True)
            # self.cell.add_auxiliary_data_path('phase_shift_scaled', phase_shift_scaled_path)

        except Exception as e:
            logging.error(f"Error during auxiliary data generation for cell {self.cell.tomogram_path}: {e}", exc_info=True)
        finally:
            # Clean up memory by unloading loaded data
            self.cell.unload_all_data()
            self.cell.unload_all_data()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()