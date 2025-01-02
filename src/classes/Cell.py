import os
import gc
from tifffile import TiffFile
import cupy as cp

class Cell:
    def __init__(self, tomogram_path: str, radiation_resistance: str, dish_number: int):
        self.tomogram_path = tomogram_path
        self.auxiliary_data_paths = {}
        self.loaded_data = {}
        self.radiation_resistance = radiation_resistance
        self.dish_number = dish_number

    def load_data(self, data_type: str):
        """
        Load specified data type if it exists and is not already loaded, converting it to a CuPy array.
        """
        if data_type == 'tomogram' and 'tomogram' not in self.loaded_data:
            if os.path.exists(self.tomogram_path):
                with TiffFile(self.tomogram_path) as tif:
                    # Load data as CuPy array directly
                    self.loaded_data['tomogram'] = cp.asarray(tif.asarray().astype('float32'))/10000
            else:
                print(f"Error: Tomogram file not found at {self.tomogram_path}")
                return None
        elif data_type in self.auxiliary_data_paths and data_type not in self.loaded_data:
            with TiffFile(self.auxiliary_data_paths[data_type]) as tif:
                self.loaded_data[data_type] = cp.asarray(tif.asarray().astype('float32'))

        return self.loaded_data.get(data_type)

    def unload_data(self, data_type: str):
        """
        Unload specified data type from memory, explicitly clearing CuPy memory.
        """
        if data_type in self.loaded_data:
            del self.loaded_data[data_type]
            gc.collect()  # Trigger Python garbage collection, may not free GPU memory immediately
            cp.get_default_memory_pool().free_all_blocks()  # Explicitly free CuPy memory blocks

    def add_auxiliary_data_path(self, data_type: str, file_path: str):
        self.auxiliary_data_paths[data_type] = file_path

    def load_tomogram(self):
        return self.load_data('tomogram')

    def unload_tomogram(self):
        self.unload_data('tomogram')

    def unload_all_data(self):
        for data_type in list(self.loaded_data.keys()):
            self.unload_data(data_type)