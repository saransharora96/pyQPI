# pyQPI: An Explainable QPI Analysis Toolkit (work in progress)
Author: Saransh Arora

## Overview

This project provides a comprehensive toolkit for processing and analyzing tomographic data of cellular samples with a focus on integrating explainable AI (XAI) techniques. The overarching goal is to demonstrate the utility of XAI models in disease stratification, leveraging quantitative phase imaging (QPI) data, particularly using a dataset for understanding radiation resistance in head and neck cancer.

The toolkit includes modules for data segmentation, feature extraction, and radiation resistance classification. Built with efficiency and scalability in mind, the project leverages GPU acceleration and includes utilities for managing and processing large datasets.

## Features

- **Segmentation**: Threshold-based segmentation and morphological operations.
- **Feature Extraction**: Calculations like dry mass and cell volume from segmented tomograms. (more to be added)
- **Auxiliary Data Generation**: Maximum Intensity Projection (MIP) images, phase shift images, and binary masks. (used for further analysis)
- **Radiation Resistance Analysis**: Categorization and feature extraction for datasets labeled by radiation resistance.
- **Explainable AI Integration**: Tools for applying interpretable models and feature importance analysis.
- **Scalable Processing**: Designed to handle large datasets with resumable and efficient processing.

## Current Progress

### Completed Work

1. **Dataset Preparation**:
   - Established a pipeline to create isogenic cell lines exposed to varying radiation doses.
   - Used optical diffraction tomography to build a comprehensive 3-D dataset of refractive index tomograms.
   - Preprocessed tomograms for standardization and normalization.

2. **Data Processing Modules**:
   - Developed modules for segmentation, auxiliary data generation, and feature extraction.
   - Designed tools to calculate key morphological and biochemical metrics.

### Future Work

1. **Initial Analysis**:
   - Implemented classical interpretable models and initial feature importance studies.
   - Identified potential phenotypic markers for radiation resistance, such as lamellipodia area and lipid mass.

2. **Explainable Deep Learning**:
   - Train multi-input CNNs for phenotype classification with SHapley Additive exPlanations (SHAP) for interpretability.
   - Augment data through axis-aligned transformations to improve model performance.

3. **Advanced Feature Analysis**:
   - Aggregate feature histograms using numerical metrics extracted from tomograms.
   - Develop and test XGBoost models with feature selection and partial dependence plots.

4. **Biological Validation**:
   - Correlate identified biomarkers with known biological pathways.
   - Quantify the number of cells needed for reliable diagnostics.

5. **Method Comparison**:
   - Systematically compare the trade-offs between classical rule-based methods and modern explainable AI techniques.

### Python Dependencies
- Python 3.8+
- CuPy
- NumPy
- pandas
- tifffile
- scikit-image
- scipy
- tqdm
- memory-profiler

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saransharora96/pyQPI.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure GPU support for CuPy (CUDA installation required).

## File Structure

```
.
├── classes
│   ├── Cell.py                # Class for managing tomographic data
│   ├── AuxiliaryDataGeneration.py # Class for generating auxiliary data
│   ├── Segmentation.py        # Segmentation utilities
│   ├── FeatureExtraction.py   # Feature extraction utilities
├── utils
│   ├── dir_utils.py           # Utilities for file and directory operations
├── config
│   ├── config_radiation_resistance.py # Configuration for analysis
├── main
│   ├── process_dataset.py     # Main script for dataset processing
│   ├── radiation_resistance_analysis.py # Analysis script
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
```

## Usage

### Dataset Processing: Main file execution

1. Configure paths and parameters in `config/config_radiation_resistance.py`.
2. Run the processing script:
   ```bash
   python main/radiation_resistance_analysis.py
   ```

### Auxiliary Data Generation

The `AuxiliaryDataGeneration` class generates auxiliary data such as binary masks, segmented tomograms, and phase shift images:

```python
from classes.AuxiliaryDataGeneration import AuxiliaryDataGeneration
from classes.Cell import Cell

directories = {
    'mip': './output/MIP',
    'phase': './output/Phase',
    # Add other directories as needed
}
cell = Cell(tomogram_path="./data/sample.tiff", radiation_resistance="sensitive", dish_number=1)
generator = AuxiliaryDataGeneration(cell, directories, pixel_x=0.095, wavelength=532e-9, background_ri=1.337)
generator.generate_and_save_auxiliary_data()
```

### Segmentation

Use `Segmentation` to process and refine tomograms:

```python
from classes.Segmentation import Segmentation, process_tomogram
import cupy as cp

tomogram = cp.random.random((256, 256, 256))  # Example tomogram
gpu_segmenter = Segmentation(offset=-0.005)
binary_mask = process_tomogram(tomogram, gpu_segmenter)
```

### Feature Extraction

Compute dry mass and cell volume:

```python
from classes.FeatureExtraction import FeatureExtraction
import cupy as cp

tomogram = cp.random.random((256, 256, 256))  # Example tomogram
binary_mask = tomogram > 0.5

mass = FeatureExtraction.calculate_dry_mass(tomogram, background_ri=1.337, alpha=0.2, pixel_x=0.095, pixel_y=0.095, pixel_z=0.19)
volume = FeatureExtraction.calculate_cell_volume(binary_mask, pixel_x=0.095, pixel_y=0.095, pixel_z=0.19)
print(f"Dry Mass: {mass} pg, Volume: {volume} µm^3")
```

## Configuration

Modify `config/config_radiation_resistance.py` to set up:

- Dataset location (`dataset_location`)
- Logging paths (`processing_log_path`, `output_csv_path`)
- Resistance mapping (`resistance_mapping`)

## Logging and Error Tracking

Logs are stored in `../pyQPI/src/logs/` and provide detailed information about errors and processing status.
