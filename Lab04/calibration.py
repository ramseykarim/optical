import numpy as np
import unpacking as up
import matplotlib.pyplot as plt
import scipy.constants as cst
from line_catalogs import *


def plot_neon_line_catalog():
    for i in range(len(NEON_LINES)):
        plt.plot([NEON_LINES[i], NEON_LINES[i]],
                 [0, NEON_LINE_HEIGHTS[i]],
                 color='g')


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

    def integrate_neon(self):
        neon = self.u.get_neon()
        flat_map = self.spec_map()
        h_offset = 0
        for order in flat_map:
            integral = simple_integrate(neon, order)
            plt.plot(np.arange(len(integral)) + h_offset, integral, '.')
            h_offset += len(integral)

    def integrate_halogen(self):
        hal = self.u.get_halogen()
        integral = integrate(hal, self.spec_map())
        plt.plot(integral, '.')

    def integrate_laser(self):
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


def centroid(array):
    weighted_sum = array * np.arange(len(array))
    return weighted_sum / np.sum(array)


def planck(wavelength_angstroms, temperature_kelvins):
    wavelength = wavelength_angstroms * 1.e-10
    first = 2. * cst.h * (cst.c ** 2.) / (wavelength ** 5.)
    exponent = cst.c * cst.h / (wavelength * cst.Boltzmann * temperature_kelvins)
    second = 1. / (np.exp(exponent) - 1.)
    return first * second
