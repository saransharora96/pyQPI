import numpy as np
import cupy as cp
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, binary_dilation, binary_erosion, ball, disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

class Segmentation:

    def __init__(self, block_size=751, offset=-0.05, use_gpu=True):
        self.block_size = block_size
        self.offset = offset
        self.use_gpu = use_gpu

    def threshold_data(self, image, method="otsu", manual_threshold=None):
        if image is None or image.size == 0 or cp.all(image == 0):
            raise ValueError("Image is empty or invalid.")

        image = cp.asarray(image, dtype=cp.float32)

        if method == "manual":
            if manual_threshold is None:
                raise ValueError("Manual threshold must be provided for the 'manual' method.")
            binary_mask = image > manual_threshold
        elif method == "otsu":
            threshold = threshold_otsu(cp.asnumpy(image))  # Need to use numpy here as threshold_otsu does not support CuPy
            binary_mask = image > (threshold + self.offset)
        elif method == "adaptive":
            if self.block_size % 2 == 0:
                raise ValueError("Block size for adaptive thresholding must be odd.")
            threshold = threshold_local(cp.asnumpy(image), self.block_size, offset=self.offset)  # Need to use numpy here
            binary_mask = image > cp.asarray(threshold)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'manual', 'otsu', 'adaptive'.")

        return binary_mask

    @staticmethod
    def apply_binary_mask(image, binary_mask):
        if image is None or image.size == 0 or np.all(image == 0):
            raise ValueError("Image is empty or invalid.")
        if binary_mask is None or binary_mask.size == 0 or binary_mask.shape != image.shape:
            raise ValueError("Binary mask is invalid or does not match the image dimensions.")

        return cp.where(binary_mask, image, 0)

    @staticmethod
    def remove_small_objects_from_mask(binary_mask, min_size=250):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        return cp.asarray(remove_small_objects(cp.asnumpy(binary_mask), min_size=min_size))

    @staticmethod
    def dilate_erode_mask(binary_mask, operation="dilate", radius=1, iterations=1):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        structuring_element = cp.asarray(ball(radius)) if binary_mask.ndim == 3 else disk(radius)
        processed_mask = binary_mask
        for _ in range(iterations):
            if operation == "dilate":
                processed_mask = cp.asarray(binary_dilation(cp.asnumpy(processed_mask), structuring_element))
            elif operation == "erode":
                processed_mask = cp.asarray(binary_erosion(cp.asnumpy(processed_mask), structuring_element))
            else:
                raise ValueError(f"Unknown operation: {operation}. Supported operations: 'dilate', 'erode'.")

        return processed_mask

    @staticmethod
    def fill_holes_in_mask(binary_mask):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        return cp.asarray(binary_fill_holes(cp.asnumpy(binary_mask)))

    @staticmethod
    def fill_holes_slice_by_slice(self, binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        filled_mask = cp.zeros_like(binary_mask, dtype=bool)
        for i in range(binary_mask.shape[0]):
            filled_mask[i] = cp.asarray(binary_fill_holes(cp.asnumpy(binary_mask[i])))

        return filled_mask

    @staticmethod
    def keep_largest_connected_component(self, binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        labeled_mask, num_features = label(cp.asnumpy(binary_mask), connectivity=3, return_num=True)
        if num_features == 0:
            return binary_mask  # Return the mask unchanged if no features are present

        component_sizes = [cp.sum(labeled_mask == i) for i in range(1, num_features + 1)]
        largest_component_label = component_sizes.index(max(component_sizes)) + 1

        return labeled_mask == largest_component_label


def process_tomogram(tomogram, processor):
    # Ensure the tomogram is a CuPy array
    if not isinstance(tomogram, cp.ndarray):
        tomogram = cp.asarray(tomogram)

    binary_mask = processor.threshold_data(tomogram)
    return binary_mask
