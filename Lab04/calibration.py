import numpy as np
import unpacking as up
import matplotlib.pyplot as plt
import scipy.constants as cst
from line_catalogs import *
from itertools import cycle


def paint_with_colors_of_wind():
    return cycle(['blue', 'green', 'red',
                  'orange', 'maroon', 'black',
                  'purple'])


def plot_neon_line_catalog():
    line_dict = {k: v for (k, v) in zip(NEON_LINES, NEON_LINE_HEIGHTS)}
    colors = paint_with_colors_of_wind()
    for order in MATCHED_LINES:
        color = colors.next()
        for line in order:
            plt.plot([line, line], [0, line_dict[line]], color=color)


"""
Array Stuff
"""


def boolean_array(array, std_multiplier):
    thresh_hold = np.mean(array) + np.std(array) * std_multiplier
    print thresh_hold
    ones = np.where(array > thresh_hold)
    new_array = np.zeros(array.shape)
    new_array[ones] += 1
    return new_array


def integrate(array, where_list):
    integral = np.array([])
    for order in where_list:
        integral = np.concatenate([integral, simple_integrate(array, order)])
    return integral


def simple_integrate(array, where_list):
    integral = np.array([])
    for where in where_list:
        integral = np.append(integral, np.sum(array[where]))
    return integral


"""
Calibration Class
"""


class Calibration:
    def __init__(self):
        self.u = up.Unpack()

    def examine_flat(self):
        flat = np.zeros([up.DIM_1, up.DIM_2])
        flat_map = self.spec_map()
        for order in flat_map:
            for where in order:
                flat[where] += 1
        plt.imshow(flat)

    def examine_neon(self):
        neon = self.u.get_neon()
        flat_map = self.spec_map()
        for i, order in enumerate(flat_map):
            print i

    def create_full_response(self):
        """
        Remember to MULTIPLY this by the spectrum
        :return:
        """
        print "CREATING RESPONSE...",
        flat = np.zeros([up.DIM_1, up.DIM_2])
        ones = np.ones([up.DIM_1, up.DIM_2])
        flat_map = self.spec_map()
        for order in flat_map:
            for where in order:
                flat[where] += 1
        response = self.u.get_halogen() * flat + ones
        for order in flat_map:
            for where in order:
                flat[where] -= 1
        print "RESPONSE COMPLETE."
        return 1. / response

    def plot_integrate_neon(self):
        neon = self.u.get_neon()
        flat_map = self.spec_map()
        h_offset = 0
        colors = paint_with_colors_of_wind()
        for order in flat_map:
            color = colors.next()
            integral = simple_integrate(neon, order)
            x = np.arange(len(integral))
            centroids = find_centroid(integral)
            plt.plot(x + h_offset, integral,
                     '.', color=color)
            plt.axvline(centroids + h_offset, color=color)
            h_offset += len(integral)

    def plot_integrate_halogen(self):
        hal = self.u.get_halogen()
        integral = integrate(hal, self.spec_map())
        plt.plot(integral, '.')

    def plot_integrate_laser(self):
        laser = self.u.get_laser()
        integral = integrate(laser, self.spec_map())
        plt.plot(integral - 50000, '--')

    def spec_map(self):
        flat = boolean_array(self.u.get_halogen(), 1)
        y, x = flat.shape
        print y
        print x
        step_size = 30
        padding = 20
        i = -1 * step_size
        integration_map = []
        count = 0
        while i < y - step_size:
            current_order = []
            ul, ur = flat[i, padding], flat[i, x - padding]
            ll, lr = flat[i + step_size, padding], flat[i + step_size, x - padding]
            if ul + ur + ll + lr == 0:
                l, r = np.sum(flat[i:i + step_size, padding]), np.sum(flat[i:i + step_size, x - padding])
                if l > 0 and r > 0:
                    count += 1
                    for j in range(x - 1, -1, -1):
                        int_y, int_x = np.where(flat[i:i + step_size, j:j + 1] == 1)
                        integrand = (int_y + i, int_x + j)
                        current_order.append(integrand)
                    i += step_size
                    integration_map.append(current_order)
            i += step_size / 2
        print "ORDERS FOUND:", count
        return integration_map


"""
Useful Stuff
"""


def simple_centroid(array):
    weighted_sum = array * np.arange(len(array))
    return weighted_sum / np.sum(array)


def planck(wavelength_angstroms, temperature_kelvins):
    wavelength = wavelength_angstroms * 1.e-10
    first = 2. * cst.h * (cst.c ** 2.) / (wavelength ** 5.)
    exponent = cst.c * cst.h / (wavelength * cst.Boltzmann * temperature_kelvins)
    second = 1. / (np.exp(exponent) - 1.)
    return first * second


"""
Neon Stuff
"""


def find_peaks(spectrum, cutoff_f):
    comparisons = []
    comparison_function = np.frompyfunc(lambda x: x > 0, 1, 1)
    width = 3
    for number in np.arange(width * 2 + 1) - width:
        if number == 0:
            continue
        differences = spectrum - np.roll(spectrum, number)
        comparisons.append(comparison_function(differences))
    comparisons = np.array(comparisons)
    peaks = np.where(np.all(comparisons, axis=0))
    relevant_peaks = cutoff_f(spectrum, peaks)
    return relevant_peaks


def find_centroid(spectrum):
    peak_cutoff_f = neon_peak_cutoff
    centroid_cutoff_f = neon_centroid_cutoff
    peaks = find_peaks(spectrum, cutoff_f=peak_cutoff_f)
    centroids = np.array([])
    for p in peaks:
        begin_peak, end_peak = centroid_cutoff_f(spectrum, p)
        peak_range = np.arange(end_peak - begin_peak) + begin_peak
        peak_values = spectrum[begin_peak:end_peak]
        assert peak_range.size == peak_values.size
        centroid = np.sum(peak_values * peak_range) / np.sum(peak_values)
        centroids = np.append(centroids, centroid)
    return centroids


def neon_peak_cutoff(spectrum, peaks):
    return [p for p in peaks[0] if spectrum[p] > np.median(spectrum)]


def neon_centroid_cutoff(spectrum, peak):
    cutoff = np.median(spectrum)
    fwhm = ((spectrum[peak] - cutoff) / 4.) + cutoff
    begin_peak = np.where(spectrum[peak::-1] < fwhm)
    begin_peak = peak - begin_peak[0][0]
    end_peak = np.where(spectrum[peak:] < fwhm)
    end_peak = end_peak[0][0] + peak
    return begin_peak, end_peak
