import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import getsortedfields as gsf

MAIN_PATH = "/home/rkarim/flatfields/"
FLAT_PATH = "_band/"
DARK_PATH = "_band_darks/"
DIM_1 = 1336
DIM_2 = 2004


def fits_open(file_name):
    """
    Opens a FITS file.
    :param file_name: String file name
    :return: Array contained in the first
    HDU object in the HDUList.
    """
    hdu = pf.open(file_name)
    data = hdu[0].data
    hdu.close()
    return data


def quick_fits_collapse(data_pile, file_name):
    """
    Quickly builds a sum array from FITS file data.
    For use with DARK frames.
    THIS FUNCTION MODIFIES INPUTS!
    :param data_pile: Already-initialized array of the same dimensions
    as a single FITS array.
    Is modified.
    :param file_name: String name of FITS file.
    :return: Function does not return. It modifies the first input.
    """
    print '.',
    data = fits_open(file_name)
    data_pile += data
    return


def process_flat(file_name_list):
    """
    Coordinates the data cube construction for FLAT frames.
    :param file_name_list: List of string file names.
    :return: Data cube (n x DIM_1 x DIM_2) and mean list (n)
    """
    data_stack = np.array([]).reshape(0, DIM_1, DIM_2)
    mean_list = np.array([])
    nonlocal_variables = {'mean_list': mean_list,
                          'data_stack': data_stack}

    # noinspection PyUnresolvedReferences
    def quick_fits_append(file_name):
        """
        Quickly builds up a cube of FITS file data.
        For use with FLAT frames.
        THIS FUNCTION MODIFIES INPUTS!
        :param data_stack: Already-initialized array of correct 3D dimension.
        Is modified.
        :param file_name: String name of FITS file.
        :return: Function does not return. It modifies the first two inputs.
        """
        print '.',
        data = fits_open(file_name)
        nonlocal_variables['data_stack'] = np.concatenate([nonlocal_variables['data_stack'], [data]])
        nonlocal_variables['mean_list'] = np.append(nonlocal_variables['mean_list'], np.mean(data))
        return

    [quick_fits_append(f) for f in file_name_list]
    print "!\nDone opening/processing FLAT directory."
    print "DATA CUBE SHAPE: ", data_stack.shape
    print data_stack
    print "MEAN LIST SHAPE: ", mean_list.shape
    return data_stack, mean_list


def process_dark(file_name_list):
    """
    Coordinates the DARK frame averaging.
    :param file_name_list: List of string file names.
    :return: Averaged DARK frame (DIM_1 x DIM_2)
    """
    data_pile = np.zeros([DIM_1, DIM_2])
    [quick_fits_collapse(data_pile, f) for f in file_name_list]
    print "!\nDone opening/processing DARK directory."
    return data_pile / float(len(file_name_list))


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
    responses = np.zeros([DIM_1, DIM_2])
    print "Calculating response map..."
    for i in range(DIM_1):
        print "\t{0:.2f}%\r".format((float(i)/DIM_1) * 100.),
        for j in range(DIM_2):
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
        # self.plot_response_map()

    def unpack(self):
        flat_file_names = gsf.generate_names(MAIN_PATH + self.band + FLAT_PATH)
        cube, means = process_flat(flat_file_names)
        # dark_file_names = gsf.generate_names(MAIN_PATH + self.band + DARK_PATH)
        # dark_average = process_dark(dark_file_names)
        # cube -= dark_average
        # self.response_map = generate_response_map(cube, means)

    def write_response_map(self):
        hdu = pf.PrimaryHDU(self.response_map)
        hdu.writeto(self.band.lower() + "_band_response_map.fts")

    def plot_response_map(self):
        plt.imshow(self.response_map)
        plt.title(self.band + " Band Response Map")
        plt.colorbar()
        plt.show()


# def unpack(band_name):
#     return_list = np.zeros([1336, 2004])
#     counter = 0
#     for f in glob(MAIN_PATH + band_name):
#         print '.',
#         hdu_list = pf.open(f)
#         return_list += hdu_list[0].data
#         hdu_list.close()
#         counter += 1
#     print '!'
#     return return_list / counter
#
#
# def produce_response(band):
#     band_list = unpack(band + FLAT_PATH)
#     band_dark_list = unpack(band + DARK_PATH)
#     band_field = band_list - band_dark_list
#     band_hdu = pf.PrimaryHDU(band_field)
#     band_hdu.writeto(band + "_band_response_map.fts")
#     return band_field
#
#
# def recover_response(band):
#     hdu = pf.open(band + "_band_response_map.fts")[0]
#     return hdu.data
#
#
# r_band_field = recover_response('r')
# v_band_field = recover_response('v')
#
# color_map = "gist_earth"
#
# plt.figure(1)
# plt.imshow(r_band_field, cmap=color_map)
# plt.title("R Band")
# plt.colorbar()
# plt.figure(2)
# plt.imshow(v_band_field, cmap=color_map)
# plt.title("V Band")
# plt.colorbar()
# plt.show()

