import nameroots as names
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


def average_run(run_list):
    data_cube = np.array(run_list)
    data_cube = np.mean(data_cube, axis=0)
    return data_cube


class Unpack:
    def __init__(self):
        self.file_names = names.generate_roots()
        self.run = self.unload_single_root(self.file_names[1])
        average_run(self.run)
        self.plot_single()

    def unload_single_root(self, root_name):
        if root_name not in self.file_names:
            raise KeyError("This is not one of the available roots."
                           "Please use the HELP function to find a better one.")
        all_files = glob.glob(os.getcwd() + "/SpectralData/" + root_name + "*")
        all_data = []
        for f in all_files:
            all_data.append(np.loadtxt(f, comments='>>', skiprows=16))
        return all_data

    def plot_single(self):
        run = self.run
        plt.plot(run[0][:, 1])
        plt.show()


u = Unpack()
