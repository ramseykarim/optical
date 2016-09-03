import numpy as np
import matplotlib.pyplot as plt
import os


class Unpack:
    def __init__(self, directory):
        self.intervals = []
        self.data = []
        self.file_names = []
        for filename in os.listdir(directory):
            data_raw = np.loadtxt(directory + filename,
                                  delimiter=',', dtype='int32')
            tick_times = data_raw[:, 1]
            self.data.append(tick_times)
            self.file_names.append(filename)
        self.generate_intervals()

    def generate_intervals(self):
        mod_sum = 2147483648 + 2147483647
        for array in self.data:
            last = array[0]
            current_intervals = []
            for time_stamp in array[1:]:
                with np.errstate(over='ignore'):
                    current = time_stamp - last
                if current < 0:
                    current += mod_sum
                current_intervals.append(current)
                last = time_stamp
            self.intervals.append(np.array(current_intervals))

    def plot_data(self):
        for i in range(len(self.data)):
            plt.plot(range(self.data[i].size), self.data[i])
        plt.xlabel("Event")
        plt.ylabel("Time Stamp")
        plt.title("All 8 trials of photon count")
        plt.legend([x[5] for x in self.file_names], loc='upper right')
        plt.show()

    def plot_histogram(self, trial):
        plt.hist(self.intervals[trial])
        plt.show()
