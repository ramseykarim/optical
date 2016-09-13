import numpy as np
import matplotlib.pyplot as plt
import histogram as hist
import binned as bn


class Stats:
    def __init__(self, unpacked):
        self.data = unpacked

    def mean(self, trial):
        return np.mean(self.data.intervals[trial])

    def std(self, trial):
        return np.std(self.data.intervals[trial])

    def mean_chunks(self, trial, chunk_size):
        """

        :param trial:
        :param chunk_size:
        :return:
        """
        dt = self.data.intervals[trial]
        i = np.arange(dt.size)
        means = []
        bins = []
        for j in i[0::chunk_size]:
            means.append(np.mean(dt[j:j + chunk_size]))
            bins.append((j + j + chunk_size - 1) / 2)
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

    def sdom(self, trial):
        sdoms = []
        chunks = np.arange(5, 400, 5)
        for chunk_size in chunks:
            means, bins = self.mean_chunks(trial, chunk_size)
            mu = self.mean(trial)
            means = np.array(means)
            sd_mean = np.sqrt(np.sum((means - mu) ** 2.) / (np.float(means.size) - 1))
            sdoms.append(sd_mean)
        return sdoms, chunks

    def histo(self, trial, n, gated=True, zoom_peak=False):
        def peak_test(b):
            if b < 3000:
                return True
            return False

        def always_false(b):
            return False

        dt = self.data.intervals[trial]
        dtp = dt[np.where(dt < 4000)]
        return hist.Histogram(dt if not zoom_peak else dtp, n, peak_test if gated else always_false)

    def compare_brightnesses(self):
        std_devs = np.array([])
        sample_means = np.array([])
        for run in self.data.intervals:
            dtp = run[np.where(run > 3000)]
            std_dev = np.std(dtp)
            sample_mean = np.mean(dtp)
            std_devs = np.append(std_devs, std_dev)
            sample_means = np.append(sample_means, sample_mean)
        return sample_means, std_devs

    """
    Plot procedures start here:
    """

    def plot_mean_chunks(self, trial, chunk_size):
        means, bins = self.mean_chunks(trial, chunk_size)
        plt.plot(bins, means, 'o', color='#005555')
        plt.xlabel("Bin Midpoint")
        plt.ylabel("Mean Interval (clock ticks)")
        plt.title("Bin Averages with steps of " + str(chunk_size)
                  + " from " + self.data.file_names[trial])
        plt.show()

    def plot_mean_progressive(self, trial, chunk_size):
        means, bins = self.mean_progressive(trial, chunk_size)
        plt.plot(bins, means, 'o', color='#009999')
        plt.xlabel("Number of Intervals Averaged")
        plt.ylabel("Mean Interval (clock ticks)")
        plt.title("Average across progressively larger sample sizes from "
                  + self.data.file_names[trial])
        plt.show()

    def plot_sdom_by_size(self, trial):
        sdoms, chunks = self.sdom(trial)
        plt.plot(chunks, sdoms, 'o', color='#009999')
        plt.xlabel("Number of events averaged")
        plt.ylabel("Standard deviation of the mean (ticks)")
        plt.title("SDOM versus size of averaged sample")
        plt.show()

    def plot_sdom_by_rootn(self, trial):
        sdoms, chunks = self.sdom(trial)
        std = np.std(self.data.intervals[trial])
        plt.plot((1 / np.sqrt(chunks)), sdoms, 'o', color='#000099')
        plt.plot((1 / np.sqrt(chunks)), std * (1 / np.sqrt(chunks)))
        plt.xlabel("$1/\sqrt{N}$")
        plt.ylabel("Standard deviation of the mean (ticks)")
        plt.title("SDOM versus $1/\sqrt{sample \, size}$ "
                  + self.data.file_names[trial])
        plt.show()

    def plot_histo(self, trial, n, gated=True, zoom_peak=False, log=True):
        h = self.histo(trial, n, gated=gated, zoom_peak=zoom_peak)
        plt.plot(h.bin_centers, h.bin_counts, drawstyle='steps-mid', color='blue')
        if not zoom_peak:
            mean = self.mean(trial)
            plt.plot(h.bin_centers, ((h.bw * h.dt.size / mean) * np.exp(-1. * h.bin_centers / mean)), color='green')
            std = self.std(trial)
            plt.plot(h.bin_centers, ((h.bw * h.dt.size / std) * np.exp(-1. * h.bin_centers / std)), color='red')
        plt.legend(['Data', 'exp(mean)', 'exp(std)'], loc='upper right')
        plt.xlabel("Interval Bin (ticks)")
        plt.ylabel("Frequency")
        plt.title("Histogram for " + self.data.file_names[trial])
        if log:
            plt.yscale('log')
        plt.show()

    def plot_std_means(self):
        means, stds = self.compare_brightnesses()
        plt.plot(means, stds, 'o')
        plt.plot(means, means)
        plt.xlabel("Interval sample mean (ticks)")
        plt.ylabel("Interval standard deviation (ticks)")
        plt.title("STD vs Mean for all runs")
        plt.show()

    def plot_binned_events1(self, trial):
        b = bn.Binned(self, trial)
        plt.plot(b.t1)
        plt.xlabel("Event Number")
        plt.ylabel("Clock time (ticks)")
        plt.title("Time Stamp vs Event Number")
        plt.show()

    def plot_binned_events2(self, trial):
        b = bn.Binned(self, trial)
        plt.plot(b.bin_centers, b.bin_count)
        plt.xlabel("Time Bin")
        plt.ylabel("Events per Bin")
        plt.title("Number of events in evenly spaced time bins")
        plt.show()

    def plot_bin_of_bins(self, trial):
        b = bn.Binned(self, trial)
        h = hist.Histogram(b.bin_count, b.bin_count.max(), lambda x: False)
        plt.plot(h.bin_centers, h.bin_counts, drawstyle='steps-mid')
        plt.xlabel("Counts per Bin")
        plt.ylabel("Frequency")
        plt.title("Histogram of events per time bin")
        plt.show()
