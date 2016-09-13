import numpy as np


class Histogram:
    def __init__(self, dt, n, truth_test):
        self.dt = dt
        self.bw = float((dt.max() - dt.min())) / (n - 1.)
        self.bin_l = np.arange(n) * self.bw + dt.min()
        self.bin_counts = np.array([])
        self.bin_centers = np.array([])
        for b in self.bin_l:
            if truth_test(b):
                continue
            count = np.where((dt >= b) & (dt < b + self.bw))[0].size
            self.bin_counts = np.append(self.bin_counts, count)
            self.bin_centers = np.append(self.bin_centers, b + 0.5 * self.bw)
