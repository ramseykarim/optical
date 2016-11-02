import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import filefunctions as gsf

MAIN_PATH = "/home/rkarim/flatfields/"
FLAT_PATH = "_band/"
DARK_PATH = "_band_darks/"
MAP_PATH = "./ResponseMaps/"


def process_flat(path):
    """
    Coordinates the data cube construction for FLAT frames.
    :param path: String path to flats.
    :return: Data cube (n x DIM_1 x DIM_2) and mean list (n)
    """
    file_name_list = gsf.generate_names(path)
    data_stack = np.array([]).reshape(0, gsf.DIM_1, gsf.DIM_2)
    mean_list = np.array([])
    nonlocal_variables = {'mean_list': mean_list,
                          'data_stack': data_stack}

    def quick_fits_append(file_name):
        """
        Quickly builds up a cube of FITS file data.
        For use with FLAT frames.
        THIS FUNCTION MODIFIES INPUTS!
        :param file_name: String name of FITS file.
        :return: Function does not return. It modifies existing arrays.
        """
        print '.',
        data = gsf.fits_open(file_name)
        nonlocal_variables['data_stack'] = np.concatenate([nonlocal_variables['data_stack'], [data]])
        nonlocal_variables['mean_list'] = np.append(nonlocal_variables['mean_list'], np.mean(data))
        return

    [quick_fits_append(f) for f in file_name_list]
    data_stack = nonlocal_variables['data_stack']
    mean_list = nonlocal_variables['mean_list']
    print "!\nDone opening/processing FLAT directory."
    return data_stack, mean_list


def generate_response_map(cube, mean_list):
    """
    Generates the response map.
    0th axis is each flat.
    1st and 2nd axes are the image dimensions.
    :param cube: The cube, (n x DIM_1 x DIM_2)
    :param mean_list: The list of means for each of n flats
    :return: Array (DIM_1 x DIM_2) of fitted slopes
    """
    get_slope = make_slope_function(mean_list)
    responses = np.zeros([gsf.DIM_1, gsf.DIM_2])
    print "Calculating response map..."
    for i in range(gsf.DIM_1):
        print "{0:.2f}%\r".format((float(i) / gsf.DIM_1) * 100.),
        for j in range(gsf.DIM_2):
            response = get_slope(cube[:, i, j])
            responses[i, j] = response
    print "\nDone"
    return responses


def make_slope_function(x_in):
    def get_slope(y_in):
        a = np.polyfit(x_in, y_in, deg=1)
        return a[0]

    return get_slope


class ResponseMap:
    """
    RESPONSE MAP CLASS
    This class will create a response map
    for a given band.
    """

    def __init__(self, band):
        """
        Creates ResponseMap object for a given band.
        :param band: 'v' or 'r' or other single character
        descriptor of a band, assuming that band has
        flats and darks stored under the directory
        MAIN_PATH
        """
        self.band = band.upper()
        self.response_map = False
        self.unpack()
        self.write_response_map()
        self.plot_response_map()

    def unpack(self):
        cube, means = process_flat(MAIN_PATH + self.band + FLAT_PATH)
        dark_average = gsf.process_dark(MAIN_PATH + self.band + DARK_PATH)
        cube -= dark_average
        self.response_map = generate_response_map(cube, means)

    def write_response_map(self):
        hdu = pf.PrimaryHDU(self.response_map)
        hdu.writeto(MAP_PATH + self.band.lower() + "_band_response_map.fts")

    def plot_response_map(self):
        plt.imshow(self.response_map)
        plt.title(self.band + " Band Response Map")
        plt.colorbar()
        plt.show()
