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
        if image is None or image.size == 0 or np.all(image == 0):
            raise ValueError("Image is empty or invalid.")
        xp = cp if self.use_gpu else np
        image = xp.asarray(image, dtype=xp.float32)

        if method == "manual":
            if manual_threshold is None:
                raise ValueError("Manual threshold must be provided for the 'manual' method.")
            binary_mask = image > manual_threshold
        elif method == "otsu":
            threshold = xp.asnumpy(threshold_otsu(xp.asnumpy(image))) if self.use_gpu else threshold_otsu(image)
            binary_mask = image > (threshold + self.offset)
        elif method == "adaptive":
            if self.block_size % 2 == 0:
                raise ValueError("Block size for adaptive thresholding must be odd.")
            threshold = xp.asarray(threshold_local(xp.asnumpy(image), self.block_size, offset=self.offset)) if self.use_gpu else threshold_local(image, self.block_size, offset=self.offset)
            binary_mask = image > threshold
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'manual', 'otsu', 'adaptive'.")

        return binary_mask

    def apply_binary_mask(self, image, binary_mask):
        if image is None or image.size == 0 or np.all(image == 0):
            raise ValueError("Image is empty or invalid.")
        if binary_mask is None or binary_mask.size == 0 or binary_mask.shape != image.shape:
            raise ValueError("Binary mask is invalid or does not match the image dimensions.")

        xp = cp if self.use_gpu else np
        return xp.where(binary_mask, image, 0)

    def remove_small_objects_from_mask(self, binary_mask, min_size=250):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        xp = cp if self.use_gpu else np
        return xp.asarray(remove_small_objects(xp.asnumpy(binary_mask), min_size=min_size))

    def dilate_erode_mask(self, binary_mask, operation="dilate", radius=1, iterations=1):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        xp = cp if self.use_gpu else np
        structuring_element = xp.asarray(ball(radius)) if binary_mask.ndim == 3 else disk(radius)
        processed_mask = binary_mask
        for _ in range(iterations):
            if operation == "dilate":
                processed_mask = xp.asarray(binary_dilation(xp.asnumpy(processed_mask), structuring_element))
            elif operation == "erode":
                processed_mask = xp.asarray(binary_erosion(xp.asnumpy(processed_mask), structuring_element))
            else:
                raise ValueError(f"Unknown operation: {operation}. Supported operations: 'dilate', 'erode'.")

        return processed_mask

    def fill_holes_in_mask(self, binary_mask):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        xp = cp if self.use_gpu else np
        return xp.asarray(binary_fill_holes(xp.asnumpy(binary_mask)))

    def fill_holes_slice_by_slice(self, binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        xp = cp if self.use_gpu else np
        filled_mask = xp.zeros_like(binary_mask, dtype=bool)
        for i in range(binary_mask.shape[0]):
            filled_mask[i] = xp.asarray(binary_fill_holes(xp.asnumpy(binary_mask[i])))

        return filled_mask

    def keep_largest_connected_component(self, binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        xp = cp if self.use_gpu else np
        labeled_mask, num_features = label(xp.asnumpy(binary_mask), connectivity=3, return_num=True)
        if num_features == 0:
            return binary_mask  # Return the mask unchanged if no features are present

        component_sizes = [xp.sum(labeled_mask == i) for i in range(1, num_features + 1)]
        largest_component_label = component_sizes.index(max(component_sizes)) + 1

        return labeled_mask == largest_component_label


def process_tomogram(tomogram, processor):

    binary_mask = processor.threshold_data(tomogram)
    # binary_mask = remove_small_objects_from_mask(binary_mask, min_size=500)
    # binary_mask = fill_holes_slice_by_slice(binary_mask)
    # binary_mask = dilate_erode_mask(binary_mask, radius=2, iterations=3)

    return binary_mask
