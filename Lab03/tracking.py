import numpy as np
import matplotlib.pyplot as plt
import filefunctions as gsf


HAT_PATH = "/home/data/Planet_Transit/HAT-P-56/"
HAT_SCI_PATH = HAT_PATH + "SI/"
HAT_DRK_PATH = HAT_PATH + "darks120s/"
RESPONSE_PATH = "ResponseMaps/"

file_names = gsf.generate_names(HAT_SCI_PATH)
data = gsf.fits_open(file_names[0])


def median_subtract(science_frame):
    median = np.median(science_frame)
    return science_frame - median


class Tracking:
    def __init__(self, sci_path=HAT_SCI_PATH, dark_path=HAT_DRK_PATH, band='v'):
        self.sci_path = sci_path
        self.dark_frame = gsf.process_dark(dark_path)
        self.response_map = gsf.fits_open(RESPONSE_PATH + band.lower() + "_band_response_map.fts")
        self.data = False
        self.file_names = gsf.generate_names(HAT_SCI_PATH)

    def file_grab(self, name):
        return gsf.fits_open(name)

    def dark_subtract(self, science_frame):
        return science_frame - self.dark_frame

    def flatten(self, science_frame):
        return science_frame / self.response_map

    def make_adjustments(self, science_frame):
        return median_subtract(self.flatten(self.dark_subtract(science_frame)))

    def find_science_star(self, science_frame):
        plt.imshow(science_frame)
        plt.show()


t = Tracking()
t.data = t.make_adjustments(t.file_grab(t.file_names[0]))
t.find_science_star(t.data)
