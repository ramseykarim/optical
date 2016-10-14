import numpy as np
import matplotlib.pyplot as plt
import unpack as up
import calibration as cal
import wavelength_lls as lls
import error_propagation as ep


class SkySpectra:
    def __init__(self, unpacker, calibrator):
        self.unpacker = unpacker
        self.calibration = calibrator
        self.cutoff = 100
        self.pixel_range = np.arange(2048 - self.cutoff) + self.cutoff
        self.wavelength_range = False
        self.calibrate_wavelength()
        self.daytime_set = up.average_run(self.obtain_dark_subtract_set("sky"))[self.cutoff:]
        self.sunset_set = up.average_run(self.obtain_dark_subtract_set("sunset_better"))[self.cutoff:]
        self.day_to_sunset = self.daytime_set / self.sunset_set

    def calibrate_wavelength(self):
        self.wavelength_range = np.array([self.calibration.wavelength_calibration[i] for i in self.pixel_range])

    def obtain_wavelength_error(self, wavelength_angstroms):
        try:
            errors = []
            for i in range(len(wavelength_angstroms)):
                difference_array = np.abs(self.wavelength_range - wavelength_angstroms[i])
                pixel = self.pixel_range[np.where(difference_array == np.min(difference_array))]
                errors.append(self.obtain_pixel_error(pixel))
                return errors
        except TypeError:
            difference_array = np.abs(self.wavelength_range - wavelength_angstroms)
            pixel = self.pixel_range[np.where(difference_array == np.min(difference_array))]
            return self.obtain_pixel_error(pixel)

    def obtain_pixel_error(self, pixel):
        assert isinstance(self.calibration, cal.Calibrate)
        return self.calibration.wavelength_error(pixel)

    def obtain_dark_subtract_set(self, s_type):
        # noinspection SpellCheckingInspection
        dark = "darks_100"
        spec_set = self.unpacker.obtain(s_type)
        dark = up.average_run(self.unpacker.obtain(dark))
        spec_set = cal.dark_adjust(spec_set, dark)
        return spec_set

    def fit_power_law(self):
        x = np.log(self.wavelength_range)
        y = np.log(self.day_to_sunset)
        power, coefficient = lls.poly_fit(x, y, degree=1)
        print "COEFF: ", np.exp(coefficient)
        print "POWER: ", power
        return power, coefficient

    def plot_spectra(self):
        plt.figure(1)
#        plt.subplot(211)
        plt.errorbar(self.wavelength_range, self.daytime_set,
                     xerr=self.obtain_pixel_error(self.pixel_range), fmt=',', color='blue')
        plt.errorbar(self.wavelength_range, self.sunset_set,
                     xerr=self.obtain_pixel_error(self.pixel_range), fmt=',', color='red')
        print "THIS"
        print self.obtain_pixel_error(self.pixel_range)
        line_height = np.max(np.append(self.daytime_set, self.sunset_set))
        plt.plot([6563, 6563], [0, line_height], '--', color='green')
        plt.legend([r"6563 $\AA$", "Daytime", "Sunset"], loc='upper right')
        plt.title("Sky Spectra", fontsize=20, family='serif')
        plt.ylabel("ADU", fontsize=16, family='serif')
        # plt.subplot(212)
        # plt.plot(self.wavelength_range, self.day_to_sunset, '.', color='red')
        # plt.title("Sky Spectra -- Daytime / Sunset", fontsize=20, family='serif')
        plt.xlabel("Wavelength", fontsize=16, family='serif')
        # plt.ylabel("Ratio", fontsize=16, family='serif')

    def plot_power_law_fit(self, power, coefficient):
        plt.figure(6)
        model = np.power(self.wavelength_range, power) * np.exp(coefficient)
        plt.plot(self.wavelength_range, self.day_to_sunset, ',', color='black')
        plt.plot(self.wavelength_range, model, '-', color='blue')
        plt.xlabel("Wavelength", fontsize=16, family='serif')
        plt.ylabel("Ratio", fontsize=16, family='serif')
        plt.title("Power Law Fit", fontsize=20, family='serif')

