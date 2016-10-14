import numpy as np
import matplotlib.pyplot as plt
import random as rand
import wavelength_lls as lls


class ErrorPropagation:
    def __init__(self, mercury_peak_set, neon_peak_set):
        self.mercury = mercury_peak_set
        self.neon = neon_peak_set
        self.fit = False
        self.fit_error = False

    def bootstrap_lls(self, n):
        n_mercury = self.mercury.n_spectra - 1
        n_neon = self.neon.n_spectra - 1

        # lambda_x_pairs_m = []
        # lambda_x_pairs_n = []
        # for i in range(n_mercury):
        #     lambda_x_pairs_m += zip(lls.MERCURY_PEAKS, self.mercury[i])
        # for i in range(n_neon):
        #     lambda_x_pairs_n += zip(lls.NEON_PEAKS, self.neon[i])
        # fits = []
        # for i in range(n):
        #     pairs_to_fit = []
        #     for j in range(20):
        #         pairs_to_fit.append(lambda_x_pairs_m[rand.randint(0, len(lambda_x_pairs_m) - 1)])
        #     for j in range(45):
        #         pairs_to_fit.append(lambda_x_pairs_n[rand.randint(0, len(lambda_x_pairs_n) - 1)])
        #     x = [p[0] for p in pairs_to_fit]
        #     y = [p[1] for p in pairs_to_fit]
        #     fits.append(lls.poly_fit(x, y))

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
        # noinspection PyTypeChecker
        fit_error = np.sum(np.subtract(fits, fit_avg)**2., axis=0) / (n - 1.)
        fit_error = np.sqrt(fit_error)
        self.fit = fit_avg
        self.fit_error = fit_error
        print "WL FIT: "
        for l, f in zip(['a', 'b', 'c'], self.fit):
            print l + " ", f
        print "WL FIT ERROR: "
        for l, f in zip(['a', 'b', 'c'], self.fit_error):
            print l + " ", f
        return self


class SingleSpectrumSet:
    def __init__(self, best_set):
        self.best_set = best_set
        self.n_spectra = best_set.shape[0]

    def mean(self):
        return np.mean(self.best_set, axis=0)

    def std(self):
        mean = self.mean()
        std = np.subtract(self.best_set, mean)
        # noinspection PyTypeChecker
        std = np.sum(std**2., axis=0) / (self.n_spectra - 1)
        return np.sqrt(std)

    def __getitem__(self, item):
        return self.best_set[item, :]

    def __setitem__(self, key, value):
        raise RuntimeError("Do not override any spectra!")

    def __delitem__(self, key):
        raise RuntimeError("Do not override any spectra!")

    def plot_all(self):
        plt.figure(10)
        for i in range(self.n_spectra):
            plt.plot(self.best_set[i, :], ',', color='k')

    def plot_std(self):
        plt.figure(11)
        plt.plot(self.std(), '.')
        plt.yscale('log', base=10)

    def plot_mean(self):
        plt.figure(12)
        plt.plot(self.mean(), '.')


def plot_from_pairs(pair_list):
    x = [p[0] for p in pair_list]
    print len(x), x[10]
    y = [p[1] for p in pair_list]
    print len(y), y[10]
    plt.figure(2)
    plt.plot(x, y, ',', color='k')
