INSTALLATION_NEEDED = False

# Tomocube Instrument Parameters
wavelength = 532e-9       # wavelength (m)
alpha = 0.2               # refractive index increment (um^3/pg)
pixel_x = 0.095           # pixel size (um)
pixel_y = 0.095
pixel_z = 0.19
background_ri = 1.337

resistance_mapping = {
    "CL1": "sensitive",
    "CL2": "resistant",
    "CL3": "intermediate"
}