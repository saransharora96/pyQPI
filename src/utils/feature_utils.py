from tifffile import imwrite
import numpy as np
from typing import Any
import os

# utilities for generating auxiliary images
def generate_and_save_mip(cell: Any, file_path: str, mip_dir: str, mip_scaled_dir: str) -> None:
    """
    Generate and save Maximum Intensity Projection (MIP) images.
    """
    mip_file_name = os.path.splitext(os.path.basename(file_path))[0] + "_MIP.tiff"
    mip_file_path = os.path.join(mip_dir, mip_file_name)
    mip_scaled_file_name = os.path.splitext(os.path.basename(file_path))[0] + "_MIP_scaled.tiff"
    mip_scaled_file_path = os.path.join(mip_scaled_dir, mip_scaled_file_name)

    if not os.path.exists(mip_file_path):
        mip = cell.generate_mip()
        mip_scaled = (mip - mip.min()) / (mip.max() - mip.min())
        imwrite(str(mip_file_path), mip.astype(np.float32))
        imwrite(str(mip_scaled_file_path), mip_scaled.astype(np.float32))

def generate_and_save_phase(cell: Any, file_path: str, phase_dir: str, phase_scaled_dir: str, pixel_x: float, wavelength: float, background_ri: float) -> None:
    """
    Generate and save phase shift images.
    """
    phase_file_name = os.path.splitext(os.path.basename(file_path))[0] + "_phase_shift.tiff"
    phase_file_path = os.path.join(phase_dir, phase_file_name)
    phase_scaled_file_name = os.path.splitext(os.path.basename(file_path))[0] + "_phase_shift_scaled.tiff"
    phase_scaled_file_path = os.path.join(phase_scaled_dir, phase_scaled_file_name)

    if not os.path.exists(phase_file_path):
        phase_shift = cell.generate_phase_delay_image(pixel_x, wavelength, background_ri)
        phase_shift_scaled = (phase_shift - phase_shift.min()) / (phase_shift.max() - phase_shift.min())
        imwrite(str(phase_file_path), phase_shift.astype(np.float32))
        imwrite(str(phase_scaled_file_path), phase_shift_scaled.astype(np.float32))