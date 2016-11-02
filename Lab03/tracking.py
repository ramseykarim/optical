import numpy as np
import matplotlib.pyplot as plt
import filefunctions as gsf


HAT_PATH = "/home/data/Planet_Transit/HAT-P-56/"
HAT_SCI_PATH = HAT_PATH + "SI/"
HAT_DRK_PATH = HAT_PATH + "darks120s/"

file_names = gsf.generate_names(HAT_SCI_PATH)
data = gsf.fits_open(file_names[0])


class Tracking:
    def __init__(self, sci_path=HAT_SCI_PATH, dark_path=HAT_DRK_PATH, band='v'):
        self.sci_path = sci_path
        self.dark_frame = gsf.process_dark(dark_path)
        self.response_map = gsf.fits_open(band.lower() + "_band_response_map.fts")
        self.data = False

    def dark_subtraction(self, science_frame):
        data_dark_sub = science_frame - self.dark_frame
        data_dark_sub -= np.median(data_dark_sub)
        return data_dark_sub

    # add in flattening here
