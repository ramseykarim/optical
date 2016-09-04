import numpy as np
import matplotlib.pyplot as plt


def show():
    plt.show()


class Stats:
    def __init__(self, unpacked):
        self.data = unpacked
    
    def mean(self, trial):
        return np.mean(self.data.intervals[trial])

    def mean_chunks(self, trial, chunk_size):
        dt = self.data.intervals[trial]
        i = np.arange(dt.size)
        means = []
        bins = []
        for j in i[0::chunk_size]:
            means.append(np.mean(dt[j:j + chunk_size]))
            bins.append((j + j + chunk_size - 1)/2)
            print j, j + chunk_size - 1, means[len(means) - 1]
        return means, bins

    def mean_progressive(self, trial, chunk_size):
        dt = self.data.intervals[trial]
        i = np.arange(dt.size)
        means = []
        bins = []
        for j in i[chunk_size::chunk_size]:
            means.append(np.mean(dt[0:j]))
            bins.append(j)
        return means, bins

    def plot_mean_chunks(self, trial, chunk_size):
        means, bins = self.mean_chunks(trial, chunk_size)
        plt.plot(bins, means, 'o', color='#005555')
        plt.xlabel("Bin")
        plt.ylabel("Mean")
        plt.title("Bin Averages with steps of " + str(chunk_size))

    def plot_mean_progressive(self, trial, chunk_size):
        means, bins = self.mean_progressive(trial, chunk_size)
        plt.plot(bins, means, 'o', color='#009999')
        plt.xlabel("Number of Intervals Averaged")
        plt.ylabel("Mean")
        plt.title("Average across progressively larger sample sizes")
