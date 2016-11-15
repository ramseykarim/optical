import subprocess
import sys
import numpy as np
import pyfits as pf
import re


DIM_1 = 1024
DIM_2 = 1048


def get_paths():
    f = open('path.txt')
    return list(f)


def generate_names(path):
    bash_cmd = "ls " + path + " | grep \.fit$"
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
    !!NOTE!!: This also works fine whenever you want to simply
    average across a bunch of FITS files.
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
    print "!\nDone opening/processing directory: " + path
    return data_pile / float(len(file_name_list))


class Unpack:
    def __init__(self):
        self.prefix = ""
        self.laser = ""
        self.load_paths()

    def load_paths(self):
        path_list = get_paths()
        for p in path_list:
            if re.match(r'^/.+', p):
                self.prefix = p.rstrip() + '/'
            elif re.match(r'.+\.fit', p):
                self.laser = p.rstrip()

    def get_dark075(self):
        """
        Dark frame for 0.75s
        :return: Single averaged frame
        """
        return process_dark(self.prefix + 'dark_0.75s/')

    def get_dark100(self):
        """
        Dark frame for 1.00s
        :return: Single averaged frame
        """
        return process_dark(self.prefix + 'dark_1s/')

    def get_neon(self):
        dark_sub = process_dark(self.prefix + 'neon/') - self.get_dark100()
        return dark_sub - np.median(dark_sub)

    def get_halogen(self):
        dark_sub = process_dark(self.prefix + 'halogen/') - self.get_dark075()
        return dark_sub - np.median(dark_sub)

    def get_laser(self):
        return self.prefix + self.laser

    def get_sun(self):
        return generate_names(self.prefix + 'sun/')
