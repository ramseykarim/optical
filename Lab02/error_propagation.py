import numpy as np
import matplotlib.pyplot as plt
import random as rand
import wavelength_lls as lls


class ErrorPropagation:
    def __init__(self, mercury_peak_set, neon_peak_set):
        self.mercury = mercury_peak_set
        self.neon = neon_peak_set

    def bootstrap_lls(self, n):
        n_mercury = self.mercury.n_spectra - 1
        n_neon = self.neon.n_spectra - 1
        samples = []
        for i in range(n):
            mercury_sample = self.mercury[rand.randint(0, n_mercury)]
            neon_sample = self.neon[rand.randint(0, n_neon)]
            sample = np.append(mercury_sample, neon_sample)
            samples.append(sample)
        fits = []
        for sample in samples:
            fit = lls.poly_fit(lls.ALL_PEAK_ARRAY, sample)
            fits.append(fit)
        fits = np.array(fits)[:, :, 0]
        fit_avg = np.mean(fits, axis=0)
        print fit_avg


class SingleSpectrumPeakSet:
    def __init__(self, best_peak_set):
        self.best_peak_set = best_peak_set
        self.n_spectra = best_peak_set.shape[0]

    def mean(self):
        return np.mean(self.best_peak_set, axis=0)

    def std(self):
        mean = self.mean()
        print self.best_peak_set.shape
        print mean.shape
        std = np.subtract(self.best_peak_set, mean)
        # noinspection PyTypeChecker
        std = np.sum(std**2., axis=0) / (self.n_spectra - 1)
        return std

    def __getitem__(self, item):
        return self.best_peak_set[item, :]

    def __setitem__(self, key, value):
        raise RuntimeError("Do not override any spectra!")

    def __delitem__(self, key):
        raise RuntimeError("Do not override any spectra!")

    def plot_std(self):
        plt.figure(1)
        plt.plot(self.std(), '.')
        plt.yscale('log', base=10)

    def plot_mean(self):
        plt.figure(2)
        plt.plot(self.mean(), '.')
