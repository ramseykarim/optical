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

    def file_grab(self, index):
        return gsf.fits_open(self.file_names[index])

    def dark_subtract(self, science_frame):
        return science_frame - self.dark_frame

    def flatten(self, science_frame):
        return science_frame / self.response_map

    def make_adjustments(self, science_frame):
        return median_subtract(self.flatten(self.dark_subtract(science_frame)))

    def find_science_star(self, index, initial_x, initial_y,
                          search_radius=50, coarse_radius=20, fine_radius=10):
        science_frame = self.file_grab(index)
        science_frame = self.make_adjustments(science_frame)
        find_centroid_in_range(science_frame, initial_x, initial_y, search_radius,
                               coarse_radius, fine_radius)
        row_sum = np.sum(science_frame, axis=1)
        print science_frame.shape
        col_sum = np.sum(science_frame, axis=0)
        print "ROW", row_sum.shape
        print "COL", col_sum.shape
        plt.figure()
        plt.imshow(science_frame)
        plt.show()


def find_centroid_in_range(science_frame, initial_x, initial_y,
                           search_radius, coarse_radius, fine_radius):
    lo_x, hi_x = initial_x - search_radius, initial_x + search_radius
    lo_y, hi_y = initial_y - search_radius, initial_y + search_radius
    search_box = science_frame[lo_x:hi_x, lo_y:hi_y]
    x_c, y_c = centroid_2d(search_box)
    x_c, y_c = x_c + lo_x, y_c + lo_y
    x_max, y_max = np.where(search_box == np.max(search_box))
    lo_x_f, hi_x_f = x_max - coarse_radius, x_max + coarse_radius
    lo_y_f, hi_y_f = y_max - coarse_radius, y_max + coarse_radius
    star_box = search_box[lo_x_f:hi_x_f, lo_y_f:hi_y_f]
    x_c_f, y_c_f = centroid_2d(star_box)
    lo_x_f, hi_x_f = x_c_f - fine_radius, x_c_f + fine_radius
    lo_y_f, hi_y_f = y_c_f - fine_radius, y_c_f + fine_radius
    star_box = star_box[lo_x_f:hi_x_f, lo_y_f:hi_y_f]
    x_c_f, y_c_f = x_c_f + lo_x + lo_x_f, y_c_f + lo_y + lo_y_f
    plt.figure()
    plt.imshow(star_box)
    print x_c, y_c
    print x_c_f, y_c_f
#    print "ROUGH CENTROID: ({0:f}, {1:f})".format(x_c, y_c)
#    print "FINER CENTROID: ({0:f}, {1:f})".format(x_c_f, y_c_f)


def centroid(y):
    x = np.arange(len(y))
    return np.sum(x * y) / np.sum(y)


def centroid_2d(box):
    x_centroid = centroid(np.sum(box, axis=1))
    y_centroid = centroid(np.sum(box, axis=0))
    print x_centroid
    print y_centroid
    return x_centroid, y_centroid


t = Tracking()
t.find_science_star(0, 842, 857)
