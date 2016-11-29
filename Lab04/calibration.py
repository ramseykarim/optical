import numpy as np
import unpacking as up
import matplotlib.pyplot as plt
import scipy.constants as cst
from line_catalogs import *
from itertools import cycle


HALOGEN_TEMPERATURE = 3000


def plot_neon_line_catalog():
    line_dict = {k: v for (k, v) in zip(NEON_LINES, NEON_LINE_HEIGHTS)}
    colors = paint_with_colors_of_wind()
    for order in MATCHED_LINES:
        color = colors.next()
        for line in order:
            plt.plot([line, line], [0, line_dict[line]], color=color)


def paint_with_colors_of_wind():
    return cycle(['blue', 'green', 'red',
                  'orange', 'maroon', 'black',
                  'purple'])


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
        self.fit_degree = 2
        self.u = up.Unpack()
        self.halogen = self.u.get_halogen()
        self.neon = self.u.get_neon()
        self.flat_map = self.spec_map()
        self.wavelength_by_order = self.wavelength_calibrate()

    def examine_flat(self):
        flat = np.zeros([up.DIM_1, up.DIM_2])
        flat_map = self.spec_map()
        for order in flat_map:
            for where in order:
                flat[where] += 1
        plt.imshow(flat)

    def plot_integrate_neon(self):
        h_offset = 0
        colors = paint_with_colors_of_wind()
        for order in self.flat_map:
            color = colors.next()
            integral = simple_integrate(self.neon, order)
            x = np.arange(len(integral))
            centroids = find_centroid(integral)
            plt.plot(x + h_offset, integral,
                     '.', color=color)
            print color
            print len(centroids)
            print "-------"
            h_offset += len(integral)

    def plot_integrate_halogen(self):
        integral = integrate(self.halogen, self.flat_map)
        plt.plot(integral, '.')

    def plot_integrate_laser(self):
        laser = self.u.get_laser()
        integral = integrate(laser, self.flat_map)
        plt.plot(integral - 50000, '--')

    def plot_neon_fit(self):
        integrals = [simple_integrate(self.neon, order) for order in self.flat_map]
        for wlf, integral in zip(self.wavelength_by_order, integrals):
            if wlf.size:
                print "W", wlf.size
                print "N", integral.size
                plt.plot(wlf, integral, '.')

    def test_fit(self):
        n_integrals = [simple_integrate(self.neon, order) for order in self.flat_map]
        flat_neon_wl, flat_neon_spec = self.flatten_spectrum(n_integrals)
        clean_n_wl, clean_n_spec = self.flat_divide(flat_neon_wl, flat_neon_spec)
        plt.plot(clean_n_wl, clean_n_spec)

    def flat_divide(self, one_d_wl, one_d_spec):
        hwl, hint = self.flatten_spectrum([simple_integrate(self.halogen, order) for order in self.flat_map])
        cleaned = np.where(hint != 0)
        black_body = planck(hwl[cleaned], HALOGEN_TEMPERATURE)
        flat = hint[cleaned]
        clean_spectrum = one_d_spec[cleaned]
        clean_wl = one_d_wl[cleaned]
        clean_spectrum = clean_spectrum * black_body / flat
        sorted_indices = np.argsort(clean_wl)
        clean_wl = clean_wl[sorted_indices]
        clean_spectrum = clean_spectrum[sorted_indices]
        return clean_wl, clean_spectrum

    def flatten_spectrum(self, integrals):
        """
        Returns corresponding 0
        :param integrals:
        :return:
        """
        one_d_wl, one_d_spec = np.array([]), np.array([])
        last_order_index = len(self.flat_map) - 1
        for i, (wl, integral) in enumerate(zip(self.wavelength_by_order, integrals)):
            if not wl.size or i == last_order_index:
                continue
            else:
                one_d_spec = np.concatenate([one_d_spec, integral])
                one_d_wl = np.concatenate([one_d_wl, wl])
        return one_d_wl, one_d_spec

    def wavelength_calibrate(self):
        """
        This WILL match the output of INTEGRATE or a loop of SIMPLE_INTEGRATE.
        This will return empty arrays for orders in which a fit was not possible.
        :return: List of numpy arrays (empty if no fit) corresponding to the
        """
        neon = self.neon
        wavelength_set = []
        integrals = []
        pixels = np.arange(up.DIM_2)
        for i, order in enumerate(self.flat_map):
            suggestions = MATCHED_PIXELS[i]
            lines = np.array(MATCHED_LINES[i])
            if len(suggestions) < 2:
                wavelength_set.append(np.array([]))
                integrals.append(np.array([]))
                continue
            integral = simple_integrate(neon, order)
            integrals.append(integral)
            peaks = gather_centroids(integral, suggestions)
            solution = poly_fit(lines, peaks, degree=self.fit_degree)
            wavelength_set.append(eval("apply_fit_" + str(self.fit_degree) + "d")(pixels, solution))
        return wavelength_set

    def spec_map(self):
        flat = boolean_array(self.halogen, 1)
        y, x = flat.shape
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
    weighted_sum = np.sum(array * np.arange(len(array)))
    return weighted_sum / np.sum(array)


def planck(wavelength_angstroms, temperature_kelvins):
    assert isinstance(wavelength_angstroms, np.ndarray)
    wavelength = wavelength_angstroms * 1.e-10
    first = 2. * cst.h * (cst.c ** 2.) / (wavelength ** 5.)
    exponent = cst.c * cst.h / (wavelength * cst.Boltzmann * temperature_kelvins)
    second = 1. / (np.exp(exponent) - 1.)
    total = first * second
    return total / np.min(total)


"""
Neon Stuff
"""


def gather_centroids(array, peak_suggestions):
    padding = 10
    return [simple_centroid(array[p - padding:p + padding]) + p - padding for p in peak_suggestions]


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
        if not begin_peak:
            continue
        peak_range = np.arange(end_peak - begin_peak) + begin_peak
        peak_values = spectrum[begin_peak:end_peak]
        assert peak_range.size == peak_values.size
        centroid = np.sum(peak_values * peak_range) / np.sum(peak_values)
        centroids = np.append(centroids, centroid)
    return centroids


def neon_peak_cutoff(spectrum, peaks):
    return [p for p in peaks[0] if spectrum[p] > np.median(spectrum) + np.std(spectrum)]


def neon_centroid_cutoff(spectrum, peak):
    cutoff = np.median(spectrum)
    fwhm = ((spectrum[peak] - cutoff) / 4.) + cutoff
    # noinspection PyBroadException
    try:
        begin_peak = np.where(spectrum[peak::-1] < fwhm)
        begin_peak = peak - begin_peak[0][0]
        end_peak = np.where(spectrum[peak:] < fwhm)
        end_peak = end_peak[0][0] + peak
        return begin_peak, end_peak
    except:
        return False, False


def poly_fit(x, y, degree=2):
    x_list = []
    while degree >= 0:
        x_list.append(np.array(x) ** degree)
        degree -= 1
    x = np.array(x_list)
    x_t = x.copy()
    x = x.transpose()
    y = np.array([y]).transpose()
    square_matrix = np.linalg.inv(np.dot(x_t, x))
    parenthetical = np.dot(square_matrix, x_t)
    a = np.dot(parenthetical, y)
    return a


def apply_fit_1d(pixels, fit):
    m, b = fit[0], fit[1]
    wavelength_solution = (pixels - b) / m
    return wavelength_solution


def apply_fit_2d(pixels, fit):
    a, b, c = fit[0], fit[1], fit[2]
    wavelength_solution = (discriminant(a, b, c, pixels) - b) / (2. * a)
    return wavelength_solution


def discriminant(a, b, c, x):
    return np.sqrt((-4. * a * (-1. * x + c)) + b ** 2.)
