from tifffile import TiffFile
import gc
import os


class Cell:
    """
    A class to handle data for a single cancer cell, including tomogram and auxiliary data.
    """
    def __init__(self, tomogram_path: str, radiation_resistance: str, dish_number: int):
        self.tomogram_path = tomogram_path  # Path to the TIFF file containing the original tomogram.
        self.auxiliary_data_paths = {}      # Stores paths for auxiliary data files.
        self.loaded_data = {}               # Stores loaded data arrays.
        self.radiation_resistance = radiation_resistance
        self.dish_number = dish_number

    def load_data(self, data_type: str):
        """
        Load specified data type if it exists and is not already loaded.
        """
        # Check if the data type refers to the original tomogram.
        if data_type == 'tomogram' and 'tomogram' not in self.loaded_data:
            if os.path.exists(self.tomogram_path):
                with TiffFile(self.tomogram_path) as tif:
                    # Changed to float32 to reduce memory usage
                    self.loaded_data['tomogram'] = tif.asarray().astype('float32')
            else:
                print(f"Error: Tomogram file not found at {self.tomogram_path}")
                return None
        elif data_type in self.auxiliary_data_paths and data_type not in self.loaded_data:
            with TiffFile(self.auxiliary_data_paths[data_type]) as tif:
                self.loaded_data[data_type] = tif.asarray().astype('float32')

        return self.loaded_data.get(data_type)

    def unload_data(self, data_type: str):
        """
        Unload specified data type from memory.
        """
        if data_type in self.loaded_data:
            del self.loaded_data[data_type]
            gc.collect()
        else:
            print(f"No data loaded for {data_type}")

    def add_auxiliary_data_path(self, data_type: str, file_path: str):
        """
        Add or update the file path for a specific type of auxiliary data.
        """
        self.auxiliary_data_paths[data_type] = file_path

    def load_tomogram(self):
        """
        Load the original tomogram and convert to float32 to reduce memory usage.
        This ensures the data is managed specifically as 'tomogram' type.
        """
        return self.load_data('tomogram')

    def unload_tomogram(self):
        """
        Unload the tomogram from instance memory.
        """
        self.unload_data('tomogram')

    def unload_all_data(self):
        """
        Unload all data from instance.
        """
        for data_type in list(self.loaded_data.keys()):
            self.unload_data(data_type)

