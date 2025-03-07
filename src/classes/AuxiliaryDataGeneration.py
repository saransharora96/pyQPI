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
from config.config_radiation_resistance import chunked_save_threshold, chunk_size


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

    async def retry_cupy_allocation(self, func, *args, retries=10, delay=3, **kwargs):
        """
        Retry CuPy memory allocation on failure, supporting both coroutines and regular functions.
        """
        for attempt in range(retries):
            try:
                # Check if the function is a coroutine
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except cp.cuda.memory.OutOfMemoryError:
                if attempt < retries - 1:
                    logging.warning(f"OutOfMemoryError: Retrying CuPy allocation in {delay}s (Attempt {attempt + 1}/{retries})...")
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    await asyncio.sleep(delay)
                else:
                    logging.error("OutOfMemoryError: All retries failed.")
                    raise

    async def save_image(self, data, directory, filename, scale=False):
        """Save an image asynchronously, optionally scaling it to [0, 1]."""
        path = os.path.join(directory, f"{filename}.tiff")
        try:
            if scale:
                data = (data - data.min()) / (data.max() - data.min())
            data = data.get()  # Convert CuPy array to NumPy

            # Use incremental write for large files
            if data.nbytes > chunked_save_threshold:
                await asyncio.to_thread(self._incremental_write_tiff, data, path)
            else:
                await asyncio.to_thread(self._sync_save_image, data, path)
                
            return path
        except Exception as e:
            logging.error(
                f"Failed to save image at {path}. Error: {str(e)} | "
                f"Data size: {data.nbytes / (1024**3):.2f} GB | "
                f"Data shape: {getattr(data, 'shape', 'Unknown')} | "
                f"Data type: {getattr(data, 'dtype', 'Unknown')}",
                exc_info=True,
                )
            return None
        finally:
            del data    

    def _sync_save_image(self, data, path):
        with TiffWriter(path) as tiff_writer:
            tiff_writer.write(data.astype('float32'))

    def _incremental_write_tiff(self, data, path, chunk_size=chunk_size):
        """Synchronous incremental write for large 3D arrays."""
        try:
            with TiffWriter(path, bigtiff=True) as tiff_writer:
                for i in range(0, data.shape[0], chunk_size):
                    chunk = data[i:i + chunk_size]
                    tiff_writer.write(chunk.astype('float32'), contiguous=False)
                logging.info(f"Data size: {data.nbytes / (1024**3):.2f} GB, Saved incrementally: {path}")
        except Exception as e:
            logging.error(f"Failed to write BigTIFF file at {path}: {e}", exc_info=True)
            raise

    async def save_h5(self, data, directory, filename):
        """Save data to an HDF5 file asynchronously."""
        path = os.path.join(directory, f"{filename}.h5")
        try:
            data = data.get()
            await asyncio.to_thread(self._sync_save_h5, data, path)
            return path
        except Exception as e:
            logging.error(f"Failed to save HDF5 file at {path}: {e}")
            return None
        finally:
            del data

    def _sync_save_h5(self, data, path):
        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("data", data=data, dtype='float32', compression="gzip")

    async def generate_and_save_auxiliary_data(self):
        """Generates and saves all required auxiliary data based on the cell's tomogram."""
        try:
            
            # Load the tomogram from the cell
            tomogram = await self.retry_cupy_allocation(self.cell.load_tomogram)
            if tomogram is None or tomogram.size == 0:
                raise ValueError(f"Invalid tomogram: {type(tomogram)}, size={getattr(tomogram, 'size', 'N/A')}")

            base_name = os.path.splitext(os.path.basename(self.cell.tomogram_path))[0]

            # Process tomogram to generate binary masks and segmented images
            try:
                segmenter = Segmentation(method="min_cross_entropy")
                tomogram_without_noisy_bottom_planes, tomogram_binary_mask, bottom_plane, max_plane = await self.retry_cupy_allocation(process_tomogram, tomogram, segmenter)
                if tomogram_binary_mask is None:
                    raise ValueError(f"process_tomogram returned None for {self.cell.tomogram_path}")
                tomogram_binary_mask_path = await self.save_image(tomogram_binary_mask, self.directories['tomogram_binary_mask'], f"{base_name}_tomogram_binary_mask")
                tomogram_without_noisy_bottom_planes_path = await self.save_image(tomogram_without_noisy_bottom_planes, self.directories['tomogram_without_noisy_bottom_planes'], f"{base_name}_tomogram_without_noisy_bottom_planes" )
                if tomogram_binary_mask_path:
                    self.cell.add_auxiliary_data_path('tomogram_binary_mask', tomogram_binary_mask_path)
                else:
                    logging.error(f"Failed to save tomogram_binary_mask for {self.cell.tomogram_path}")
                if tomogram_without_noisy_bottom_planes_path:
                    self.cell.add_auxiliary_data_path('tomogram_without_noisy_bottom_planes', tomogram_without_noisy_bottom_planes_path)
                else:
                    logging.error(f"Failed to save tomogram_without_noisy_bottom_planes for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating tomogram_binary_mask for {self.cell.tomogram_path}: {e}", exc_info=True)

            # # Generate and save 2D binary mask
            # try:
            #     binary_mask_image = FeatureExtraction.generate_mip(tomogram_binary_mask)
            #     binary_mask_image_path = await self.save_image(binary_mask_image, self.directories['image_binary_mask'], f"{base_name}_image_binary_mask")
            #     if binary_mask_image_path:
            #         self.cell.add_auxiliary_data_path('image_binary_mask', binary_mask_image_path)
            #     else:
            #         logging.error(f"Failed to save binary_mask_image for {self.cell.tomogram_path}")
            # except Exception as e:
            #     logging.error(f"Error generating binary_mask_image for {self.cell.tomogram_path}: {e}", exc_info=True)

            # Segment the tomogram
            try:
                tomogram_segmented = await self.retry_cupy_allocation(Segmentation.apply_binary_mask, tomogram_without_noisy_bottom_planes, tomogram_binary_mask)
                tomogram_segmented_path = await self.save_h5(tomogram_segmented, self.directories['tomogram_segmented'], f"{base_name}_tomogram_segmented")
                if tomogram_segmented_path:
                    self.cell.add_auxiliary_data_path('tomogram_segmented', tomogram_segmented_path)
                else:
                    logging.error(f"Failed to save tomogram_segmented for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating tomogram_segmented for {self.cell.tomogram_path}: {e}", exc_info=True)

            del tomogram_binary_mask, tomogram_segmented
            cp.get_default_memory_pool().free_all_blocks()

            # Generate and save MIP
            try:
                mip = await self.retry_cupy_allocation(FeatureExtraction.generate_mip, tomogram)
                mip_path = await self.save_image(mip, self.directories['mip'], f"{base_name}_mip")
                if mip_path:
                    self.cell.add_auxiliary_data_path('mip', mip_path)
                else:
                    logging.error(f"Failed to save MIP for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating MIP for {self.cell.tomogram_path}: {e}", exc_info=True)

                        # Generate and save MIP
            del tomogram
            cp.get_default_memory_pool().free_all_blocks()
            try:
                mip_without_noisy_bottom_planes = await self.retry_cupy_allocation(FeatureExtraction.generate_mip, tomogram_without_noisy_bottom_planes)
                mip_without_noisy_bottom_planes_path = await self.save_image(mip_without_noisy_bottom_planes, self.directories['mip_without_noisy_bottom_planes'], f"{base_name}_mip_without_noisy_bottom_planes")
                if mip_without_noisy_bottom_planes_path:
                    self.cell.add_auxiliary_data_path('mip_without_noisy_bottom_planes', mip_without_noisy_bottom_planes_path)
                else:
                    logging.error(f"Failed to save MIP_without_noisy_bottom_planes for {self.cell.tomogram_path}")
            except Exception as e:
                logging.error(f"Error generating MIP_without_noisy_bottom_planes for {self.cell.tomogram_path}: {e}", exc_info=True)

            # #Generate and save scaled MIP
            # mip_scaled_path = self.save_image(mip, self.directories['mip_scaled'], f"{base_name}_mip_scaled", scale=True)
            # self.cell.add_auxiliary_data_path('mip_scaled', mip_scaled_path)

            del tomogram_without_noisy_bottom_planes
            cp.get_default_memory_pool().free_all_blocks()

            # # Generate and save segmented MIP
            # try:
            #     segmented_mip = FeatureExtraction.generate_mip(tomogram_segmented)
            #     segmented_mip_path = await self.save_h5(segmented_mip, self.directories['mip_segmented'], f"{base_name}_mip_segmented")
            #     if segmented_mip_path:
            #         self.cell.add_auxiliary_data_path('mip_segmented', segmented_mip_path)
            #     else:
            #         logging.error(f"Failed to save segmented MIP for {self.cell.tomogram_path}")
            #     del segmented_mip, tomogram_segmented
            #     cp.get_default_memory_pool().free_all_blocks()
            # except Exception as e:
            #     logging.error(f"Error generating segmented MIP for {self.cell.tomogram_path}: {e}", exc_info=True)

            # Generate and save scaled segmented MIP
            # try:
            #     mip_segmented_scaled = Segmentation.apply_binary_mask((mip-mip.min())/(mip.max()-mip.min()), binary_mask_image)
            #     mip_segmented_scaled_path = await self.save_image(mip_segmented_scaled, self.directories["mip_segmented_scaled"], f"{base_name}_segmented_MIP_scaled")
            #     if mip_segmented_scaled_path:
            #         self.cell.add_auxiliary_data_path('mip_segmented_scaled', mip_segmented_scaled_path)
            #     else:
            #         logging.error(f"Failed to save scaled segmented MIP for {self.cell.tomogram_path}")
            #     del mip_segmented_scaled, binary_mask_image, mip
            #     cp.get_default_memory_pool().free_all_blocks()
            # except Exception as e:
            #     logging.error(f"Error generating scaled segmented MIP for {self.cell.tomogram_path}: {e}", exc_info=True)

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
            return bottom_plane, max_plane