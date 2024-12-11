class Cell:
    """
    A class to handle data from each cancer cell in the Quantitative Phase Imaging dataset.

    Each data point will likely be a Tomogram, and other generated data from the Tomogram, like a phase image or a maximum intensity projection (MIP)

    In this particular project, we are working with Radiation Resistance in Head & Neck Cancer

    All methods for single instances of cells and their processing will be included here.
    """

    def __init__(self):