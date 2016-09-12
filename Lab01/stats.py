import numpy as np
import matplotlib.pyplot as plt


def histogram(dt, n, truth_test):
    bw = float((dt.max() - dt.min())) / (n - 1.)
    print "BW: ", bw
    bin_l = np.arange(n) * bw + dt.min()
    bin_count = np.array([])
    bin_centers = np.array([])
    for b in bin_l:
        if truth_test(b):
            continue
        count = np.where((dt >= b) & (dt < b + bw))[0].size
        bin_count = np.append(bin_count, count)
        bin_centers = np.append(bin_centers, b + 0.5 * bw)
    return bin_centers, bin_count


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
            sd_mean = np.sqrt(np.sum((means - mu)**2.)/(np.float(means.size) - 1))
            sdoms.append(sd_mean)
        return sdoms, chunks

    def histo(self, trial, n):
        def peak_test(b):
            if b < 3000:
                return True
            return False

        def always_false(b):
            return False

        dt = self.data.intervals[trial]
        dtp = dt[np.where(dt < 4000)]
        return histogram(dt, n, peak_test)

    def compare_brightnesses(self):
        std_devs = np.array([])
        sample_means = np.array([])
        for run in self.data.intervals:
            std_dev = np.std(run)
            sample_mean = np.mean(run)
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
        plt.plot((1/np.sqrt(chunks)), sdoms, 'o', color='#000099')
        plt.plot((1/np.sqrt(chunks)), std*(1/np.sqrt(chunks)))
        plt.xlabel("$1/\sqrt{N}$")
        plt.ylabel("Standard deviation of the mean (ticks)")
        plt.title("SDOM versus $1/\sqrt{sample \, size}$ "
                  + self.data.file_names[trial])
        plt.show()

    def plot_histo(self, trial, n):
        bin_centers, bin_counts = self.histo(trial, n)
        plt.plot(bin_centers, bin_counts, drawstyle='steps-mid')
        plt.yscale('log')
        plt.show()

    def plot_std_means(self):
        means, stds = self.compare_brightnesses()
        plt.plot(means, stds, '.')
        plt.show()
