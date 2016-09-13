import numpy as np
import stats as st


class Binned:
    def __init__(self, stat, trial):
        assert isinstance(stat, st.Stats)
        self.dt = stat.data.intervals[trial]
        self.dt = self.dt[np.where(self.dt > 3000)]
        self.t1 = np.cumsum(self.dt)
        self.bin_count = np.array([])
        self.bin_centers = np.array([])
        self.n = 1500
        self.bw = (self.t1.max() - self.t1.min()) / float(self.n - 1)
        for b in (np.arange(self.n) * self.bw + self.t1.min()):
            count = np.where((self.t1 >= b) & (self.t1 < b + self.bw))[0].size
            self.bin_count = np.append(self.bin_count, count)
            self.bin_centers = np.append(self.bin_centers, b + self.bw * 0.5)
