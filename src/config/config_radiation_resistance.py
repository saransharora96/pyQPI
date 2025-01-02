INSTALLATION_NEEDED = False
RESUME_PROCESSING = False

# Tomocube Instrument Parameters
wavelength = 532e-9       # wavelength (m)
alpha = 0.2               # refractive index increment (um^3/pg)
pixel_x = 0.095           # pixel size (um)
pixel_y = 0.095
pixel_z = 0.19
background_ri = 1.337

resistance_mapping = {
    "cell_line_1": "sensitive",
    "cell_line_2": "resistant",
    "cell_line_3": "intermediate"
}