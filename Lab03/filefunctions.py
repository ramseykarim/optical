import subprocess
import sys
import numpy as np
import pyfits as pf


DIM_1 = 1336
DIM_2 = 2004


def generate_names(path):
    bash_cmd = "ls " + path + " | grep \.fts$"
    process = subprocess.Popen(["bash", "-c", bash_cmd],
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    file_name_list = output.split("\n")
    file_name_list.pop()
    return file_name_prefix_list(file_name_list, path)


def file_name_prefix_list(old_list, prefix):
    new_list = []
    for name in old_list:
        new_list.append(prefix + name)
    return new_list


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


def process_dark(path):
    """
    Coordinates the DARK frame averaging.
    :param path: String path to dark files.
    :return: Averaged DARK frame (DIM_1 x DIM_2)
    """
    file_name_list = generate_names(path)
    data_pile = np.zeros([DIM_1, DIM_2])
    nonlocal_variables = {'data_pile': data_pile}

    def quick_fits_collapse(file_name):
        """
        Quickly builds a sum array from FITS file data.
        For use with DARK frames.
        THIS FUNCTION MODIFIES INPUTS!
        :param file_name: String name of FITS file.
        :return: Function does not return. It modifies an existing array.
        """
        sys.stdout.write('.')
        sys.stdout.flush()
        data = fits_open(file_name)
        nonlocal_variables['data_pile'] += data
        return

    [quick_fits_collapse(f) for f in file_name_list]
    print "!\nDone opening/processing DARK directory."
    return data_pile / float(len(file_name_list))




