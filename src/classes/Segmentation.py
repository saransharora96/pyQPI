import cupy as cp
from cupyx.scipy.ndimage import binary_fill_holes, label, binary_dilation, binary_erosion

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
            binary_mask = self.otsu_threshold_gpu(image)
        elif method == "adaptive":
            if self.block_size % 2 == 0:
                raise ValueError("Block size for adaptive thresholding must be odd.")
            binary_mask = self.adaptive_threshold_gpu(image)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'manual', 'otsu', 'adaptive'.")

        return binary_mask

    @staticmethod
    def otsu_threshold_gpu(image):
        hist, bin_edges = cp.histogram(image.ravel(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight1 = cp.cumsum(hist)
        weight2 = cp.cumsum(hist[::-1])[::-1]
        mean1 = cp.cumsum(hist * bin_centers) / weight1
        mean2 = (cp.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
        variance_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        idx = cp.argmax(variance_between)
        return image > bin_centers[idx]

    @staticmethod
    def adaptive_threshold_gpu(image, block_size, offset):
        pad_size = block_size // 2
        padded_image = cp.pad(image, pad_width=pad_size, mode='reflect')
        threshold_image = cp.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                local_region = padded_image[i:i+block_size, j:j+block_size]
                threshold_image[i, j] = local_region.mean() - offset

        return image > threshold_image

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

        labels, num_features = label(binary_mask)
        component_sizes = cp.bincount(labels.ravel())
        mask = component_sizes >= min_size
        return mask[labels]

    @staticmethod
    def dilate_erode_mask(binary_mask, operation="dilate", radius=1, iterations=1):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        structuring_element = cp.zeros((2 * radius + 1,) * binary_mask.ndim)
        coords = cp.indices((2 * radius + 1,) * binary_mask.ndim) - radius
        structuring_element[cp.sqrt((coords ** 2).sum(axis=0)) <= radius] = 1

        for _ in range(iterations):
            if operation == "dilate":
                binary_mask = binary_dilation(binary_mask, structure=structuring_element)
            elif operation == "erode":
                binary_mask = binary_erosion(binary_mask, structure=structuring_element)
            else:
                raise ValueError(f"Unknown operation: {operation}. Supported operations: 'dilate', 'erode'.")

        return binary_mask

    @staticmethod
    def fill_holes_in_mask(binary_mask):
        if binary_mask is None or binary_mask.size == 0:
            raise ValueError("Binary mask is empty or invalid.")

        return binary_fill_holes(binary_mask)

    @staticmethod
    def fill_holes_slice_by_slice(binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        filled_mask = cp.zeros_like(binary_mask, dtype=bool)
        for i in range(binary_mask.shape[0]):
            filled_mask[i] = binary_fill_holes(binary_mask[i])

        return filled_mask

    @staticmethod
    def keep_largest_connected_component(binary_mask):
        if binary_mask is None or binary_mask.size == 0 or binary_mask.ndim != 3:
            raise ValueError("Input must be a non-empty 3D binary mask.")

        labels, num_features = label(binary_mask)
        if num_features == 0:
            return binary_mask  # Return the mask unchanged if no features are present

        component_sizes = cp.bincount(labels.ravel())
        largest_component = cp.argmax(component_sizes[1:]) + 1
        return labels == largest_component


def process_tomogram(tomogram, processor):
    if not isinstance(tomogram, cp.ndarray):
        tomogram = cp.asarray(tomogram)
    return processor.threshold_data(tomogram)
