import nameroots as names
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


SPECTRA_PATH = os.getcwd() + "/SpectralData/"


def average_run(run_list):
    data_cube = np.array(run_list)
    data_cube = np.mean(data_cube, axis=0)
    return data_cube


def plot_single(run):
    r = average_run(run)
    plt.plot(r)
    plt.show()


def get_integration_time(root_name):
    first_file = open(SPECTRA_PATH + root_name + "_00001.txt")
    lines = first_file.readlines()
    return lines[8]


class Unpack:
    def __init__(self):
        self.file_names = names.generate_roots()
        self.spectra = {}

    def __repr__(self):
        self.print_roots()
        return "Unpack object. Use REQUEST_LOOP to select a spectrum."

    def unload_single_root(self, root_name):
        if root_name not in self.file_names:
            raise KeyError("This is not one of the available roots. "
                           "Please use the HELP function to find a better one.")
        all_files = glob.glob(SPECTRA_PATH + root_name + "*")
        all_data = []
        for f in all_files:
            all_data.append(np.loadtxt(f, comments='>>', skiprows=16, usecols=[1]))
        return all_data

    def get_integration_times(self):
        for root_name in self.file_names:
            if root_name == "hope" or root_name == "ramsey" or root_name == "":
                continue
            int_time = get_integration_time(root_name)
            print root_name + " : " + int_time

    def obtain(self, root_name):
        if root_name in self.spectra:
            return self.spectra[root_name]
        else:
            current_spectra = self.unload_single_root(root_name)
            self.spectra[root_name] = current_spectra
            return current_spectra

    def print_roots(self):
        for f in self.file_names:
            print f

    def request_loop(self):
        while True:
            request = raw_input("Enter root name:\n")
            if request.lower() == "stop" or request.lower() == "exit":
                break
            elif request.lower() == "print":
                self.print_roots()
            elif request.lower() == "show":
                plt.show()
            else:
                try:
                    result = self.obtain(request)
                    plot_single(result)
                except KeyError:
                    print "That's not valid. Try again."
