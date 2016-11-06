import numpy as np
import matplotlib.pyplot as plt
import filefunctions as gsf
import star


HAT_PATH = "/home/data/Planet_Transit/HAT-P-56/"
HAT_SCI_PATH = HAT_PATH + "SI/"
HAT_DRK_PATH = HAT_PATH + "darks120s/"
RESPONSE_PATH = "ResponseMaps/"

file_names = gsf.generate_names(HAT_SCI_PATH)
data = gsf.fits_open(file_names[0])


def median_subtract(science_frame):
    median = np.median(science_frame)
    return science_frame - median


def find_star(science_frame,
              (initial_x, initial_y),
              search_radius=50, coarse_radius=20, fine_radius=15):
    x_t, y_t, x, y, star_box = star.find_centroid_in_range(science_frame,
                                                           (initial_x, initial_y),
                                                           search_radius, coarse_radius,
                                                           fine_radius)
    return x_t, y_t, x, y, star_box


class Tracking:
    def __init__(self, sci_path=HAT_SCI_PATH, dark_path=HAT_DRK_PATH, band='v'):
        self.sci_path = sci_path
        self.dark_frame = gsf.process_dark(dark_path)
        self.response_map = gsf.fits_open(RESPONSE_PATH + band.lower() + "_band_response_map.fts")
        self.stars = {}
        self.file_names = gsf.generate_names(HAT_SCI_PATH)

    def file_grab(self, index):
        return gsf.fits_open(self.file_names[index])

    def dark_subtract(self, science_frame):
        return science_frame - self.dark_frame

    def flatten(self, science_frame):
        return science_frame / self.response_map

    def make_adjustments(self, science_frame):
        return median_subtract(self.flatten(self.dark_subtract(science_frame)))

    def find_initial(self, identifier, science_frame,
                     initial_x, initial_y):
        star_instance = star.Star(identifier)
        star_instance.add_frame(find_star(science_frame, (initial_x, initial_y)))
        self.stars[identifier] = star_instance

    def star_search_loop(self, index, initial_x, initial_y):
        science_frame = self.file_grab(index)
        science_frame = self.make_adjustments(science_frame)
        print "SCIFRM MED", np.median(science_frame)
        self.find_initial('science', science_frame, initial_x, initial_y)
        self.stars['science'].get_aperture(0)
        # TODO write this function!


t = Tracking()
t.star_search_loop(0, 842, 857)
