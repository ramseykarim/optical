import numpy as np
import matplotlib.pyplot as plt
import os


class Unpack:
    def __init__(self, directory):
        """
        Initialize Unpack object
        :param directory: string path to the data directory
        """
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
        """
        Don't use this function, it is used in __init__ already
        """
        for array in self.data:
            last = array[0]
            current_intervals = []
            for time_stamp in array[1:]:
                with np.errstate(over='ignore'):
                    current = time_stamp - last
                current_intervals.append(current)
                last = time_stamp
            self.intervals.append(np.array(current_intervals))

    def plot_data(self, trial):
        """
        Plots data from one trial as TimeStamp vs EventNumber
        :param trial: number (0-9) of trial that should be plotted
        """
        plt.plot(self.data[trial], color='k')
        plt.xlabel("Event Number")
        plt.ylabel("Time Stamp (clock ticks)")
        plt.title("Data from: "+self.file_names[trial])
        plt.show()

    def plot_all_data(self):
        """"
        Plots all the data as TimeStamp vs EventNumber
        """
        for i in range(len(self.data)):
            plt.plot(range(self.data[i].size), self.data[i])
        plt.xlabel("Event Number")
        plt.ylabel("Time Stamp (clock ticks)")
        plt.title("All 8 trials of photon count")
        plt.legend([x[5] for x in self.file_names], loc='upper right')
        plt.show()

    def plot_intervals(self, trial):
        """
        Plots intervals for a single trial
        :param trial: number (0-9) of trial whose intervals should be plotted
        """
        plt.plot(self.intervals[trial], '.', color='k')
        plt.xlabel("Event Number")
        plt.ylabel("$dt$ (clock ticks)")
        plt.title("Intervals from: "+self.file_names[trial])
        plt.show()

    def plot_all_intervals(self):
        """
        Plots all intervals
        """
        for i in range(len(self.data)):
            plt.plot(self.intervals[i], ',')
        plt.xlabel("Event Number")
        plt.ylabel("$dt$ (clock ticks)")
        plt.title("All Intervals")
        plt.show()

    def plot_histogram(self, trial):
        """
        Plots a histogram of a single trial
        :param trial: number (0-9) of trial whose histogram should be plotted
        """
        plt.hist(self.intervals[trial])
        plt.xlabel("$dt$ bin")
        plt.ylabel("Abundance")
        plt.title("Trial: " + self.file_names[trial])
        plt.show()
