import cupy as cp
from cucim.skimage.morphology import binary_dilation, binary_erosion, remove_small_objects, ball, disk
from cucim.skimage.measure import label
from cucim.skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage import binary_fill_holes
from utils.dir_utils import ensure_cupy_array
import logging
from config.config_radiation_resistance import bottom_threshold_factor
import numpy as np
from skimage.filters import threshold_li

class Segmentation:
    def __init__(
        self,
        block_size=751,
        offset=-0.05,
        threshold_correction_factor=1,
        threshold_min=0,
        threshold_max=1,
        method="otsu",
        use_gpu=True,
    ):
        self.block_size = block_size
        self.offset = offset
        self.threshold_correction_factor = threshold_correction_factor
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.method = method
        self.use_gpu = use_gpu

    def threshold_data(self, image, method=None, manual_threshold=None):
        if image is None or image.size == 0 or cp.all(image == 0):
            raise ValueError("Image is empty or invalid.")

        method = method or self.method
        image = ensure_cupy_array(image)

        if method == "manual":
            if manual_threshold is None:
                raise ValueError("Manual threshold must be provided for the 'manual' method.")
            binary_mask = image > manual_threshold
        elif method == "otsu":
            threshold = threshold_otsu(image)  # cuCIM's threshold_otsu works with CuPy arrays
            binary_mask = image > (threshold + self.offset)
        elif method == "adaptive":
            if self.block_size % 2 == 0:
                raise ValueError("Block size for adaptive thresholding must be odd.")
            threshold = threshold_local(image, self.block_size, offset=self.offset)  # cuCIM's implementation
            binary_mask = image > threshold
        elif method == "min_cross_entropy":
            raw_threshold = threshold_li(cp.asnumpy(image))
            #adjusted_threshold = raw_threshold * self.threshold_correction_factor
            #final_threshold = cp.clip(adjusted_threshold, self.threshold_min, self.threshold_max).item()
            binary_mask = cp.asarray(image > raw_threshold)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'manual', 'otsu', 'adaptive', 'min_cross_entropy'.")

        del image  # Free memory of the input image
        cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory
        return binary_mask

    @staticmethod
    def apply_binary_mask(image, binary_mask):
        if image is None or image.size == 0 or cp.all(image == 0):
            raise ValueError("Image is empty or invalid.")
        if binary_mask is None or binary_mask.size == 0 or binary_mask.shape != image.shape:
            raise ValueError("Binary mask is invalid or does not match the image dimensions.")

        return cp.where(binary_mask, image, 0)

    @staticmethod
    def remove_small_objects_from_mask(binary_mask, min_size=250):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        return remove_small_objects(binary_mask, min_size=min_size)

    @staticmethod
    def dilate_erode_mask(binary_mask, operation="dilate", radius=1, iterations=1):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        structuring_element = ball(radius) if binary_mask.ndim == 3 else disk(radius)
        processed_mask = binary_mask
        for _ in range(iterations):
            if operation == "dilate":
                binary_dilation(processed_mask, structuring_element, out=processed_mask)
            elif operation == "erode":
                binary_erosion(processed_mask, structuring_element, out=processed_mask)
            else:
                raise ValueError(f"Unknown operation: {operation}. Supported operations: 'dilate', 'erode'.")

        del binary_mask  # Free memory of the input mask
        cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory
        return processed_mask

    @staticmethod
    def fill_holes_in_mask(binary_mask):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")
        return ensure_cupy_array(binary_fill_holes(cp.asnumpy(binary_mask)), dtype=bool)


    @staticmethod
    def fill_holes_slice_by_slice(binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        filled_mask = cp.zeros_like(binary_mask, dtype=bool)
        for i in range(binary_mask.shape[0]):
            filled_mask[i] = ensure_cupy_array(binary_fill_holes(cp.asnumpy(binary_mask[i])), dtype=bool)
            del slice_cpu  # Free CPU memory
            cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory

        return filled_mask

    @staticmethod
    def keep_largest_connected_component(binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        labeled_mask, num_features = label(binary_mask, connectivity=3, return_num=True)
        if num_features == 0:
            return binary_mask  # Return the mask unchanged if no features are present

        largest_component_label = 0
        max_size = 0
        for i in range(1, num_features + 1):
            component_size = cp.sum(labeled_mask == i)
            if component_size > max_size:
                max_size = component_size
                largest_component_label = i

        largest_component = labeled_mask == largest_component_label
        del labeled_mask  # Free memory for the labeled mask
        cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory
        return largest_component
    
    @staticmethod
    def remove_noisy_bottom_planes(binary_mask):
        """
        Finds the plane with the maximum number of 1s in a 3D binary mask, the nearest plane below it 
        with a user-defined multiplication_factor of the number of 1s, and logs the bottom plane index.

        Args:
            binary_mask (cupy.ndarray): A 3D binary mask.
            log_dir (str): Directory to save the log file.
            log_filename (str): Name of the log file.

        Returns:
            tuple: Indices of the plane with the maximum number of 1s and the nearest plane below it with 
                a user-defined multiplication_factor of the number of 1s. If no such plane is found, returns None for the second index.
        """
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")
        
        binary_mask = ensure_cupy_array(binary_mask)
        plane_sums = cp.sum(binary_mask, axis=(1, 2))       # Calculate the sum of ones for each plane along the Z-axis
        max_plane_index = int(cp.argmax(plane_sums))
        max_plane_count = plane_sums[max_plane_index]
        desired_count = bottom_threshold_factor*max_plane_count 
               
        # Find the nearest plane below the max_plane_index with the desired count
        below_planes = plane_sums[:max_plane_index]
        if below_planes.size > 0:
            below_desired_count_indices = cp.where(below_planes <= desired_count)[0]
            if below_desired_count_indices.size > 0:
                nearest_below_index = int(below_desired_count_indices[-1])  # Last plane satisfying the condition
            else:
                nearest_below_index = max_plane_index - 1  # Default to the plane immediately below the max plane
        else:
            nearest_below_index = 0  # Default to the first plane if no planes exist below the max

        return nearest_below_index, max_plane_index

    @staticmethod
    def remove_upto_plane(tomogram, bottom_plane):
        """
        Removes all planes up to and including `nearest_below_index` from the tomogram.

        Args:
            tomogram (cupy.ndarray): A 3D tomogram.
            nearest_below_index (int): The index up to which planes should be removed (inclusive).

        Returns:
            cupy.ndarray: The updated tomogram with planes removed.
        """
        if tomogram is None or tomogram.size == 0 or tomogram.ndim != 3:
            raise ValueError("Input must be a non-empty 3D tomogram.")
        if bottom_plane is None or bottom_plane < 0 or bottom_plane >= tomogram.shape[0]:
            raise ValueError("Invalid `bottom_plane`. It must be within the range of tomogram planes.")

        # Ensure the tomogram is a CuPy array
        tomogram = ensure_cupy_array(tomogram)

        return tomogram[bottom_plane + 1 :, :, :]


def process_tomogram(tomogram, processor):
    try:
        if not isinstance(tomogram, cp.ndarray):
            tomogram = cp.asarray(tomogram)

        binary_mask = processor.threshold_data(tomogram)
        bottom_plane, max_plane = processor.remove_noisy_bottom_planes(binary_mask)
        tomogram_with_noisy_parts_removed = processor.remove_upto_plane(tomogram, bottom_plane)
        binary_mask_with_noisy_parts_removed = processor.remove_upto_plane(binary_mask, bottom_plane)

        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Generated binary mask is invalid (None or empty).")

        return tomogram_with_noisy_parts_removed, binary_mask_with_noisy_parts_removed, bottom_plane, max_plane
    
    except Exception as e:
        logging.error(f"Error in process_tomogram: {e}", exc_info=True)
        return None
