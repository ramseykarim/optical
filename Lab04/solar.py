import numpy as np
import matplotlib.pyplot as plt
import calibration as cal
import sys


class Sun:
    def __init__(self):
        self.calibrator = cal.Calibration()
        self.suns = self.calibrator.u.get_sun()
        self.center = 0
        self.curve = np.array([])
        self.light_curve()
        self.background_light = self.generate_light_noise()

    def light_curve(self):
        for frame in self.suns:
            self.curve = np.append(self.curve, np.sum(frame))
        self.center = np.argmax(self.curve)

    def calibrate_sun(self, sun_frame):
        wavelengths, spectrum = self.calibrator.calibrate(sun_frame)
        return wavelengths, spectrum - self.background_light

    def test_suns(self):
        center_wavelengths, center_spectrum = self.calibrate_sun(self.suns[self.center])
        plt.plot(center_wavelengths, center_spectrum, '.', color='orange')
        in_transit = np.where(self.curve > 2 * np.mean(self.curve))[0]
        shift_array = np.array([])
        strength_array = np.array([])
        for position in in_transit:
            wl, spec = self.calibrate_sun(self.suns[position])
            offset, strength = correlate_offset(center_spectrum, spec, 20)
            shift_array = np.append(shift_array, offset)
            strength_array = np.append(strength_array, strength)
        plt.figure()
        plt.plot(shift_array)
        plt.title("shift")
        plt.figure()
        plt.title("strength")
        plt.plot(strength_array)

    def generate_light_noise(self):
        background_indices = np.where(self.curve <= np.median(self.curve))
        specs = []
        for count, i in enumerate(background_indices[0]):
            sys.stdout.write("Finding background sky... {0} % \r".format((count + 1)/len(background_indices)))
            sys.stdout.flush()
            wl, spec = self.calibrator.calibrate(self.suns[i])
            specs.append(spec)
        sys.stdout.write("\nDone!\n")
        sys.stdout.flush()
        specs = np.array(specs)
        spec = np.mean(specs, axis=0)
        return spec


def correlate_offset(a_static, a_roll, max_offset):
    offset_range = np.arange(max_offset * 2 + 1) - max_offset
    return_value = np.array([])
    for offset in offset_range:
        shifted = np.roll(a_roll, offset)
        difference = a_static[max_offset:-max_offset] - shifted[max_offset:-max_offset]
        # noinspection PyTypeChecker
        return_value = np.append(return_value, np.sum(difference**2.))
    return offset_range[np.argmin(return_value)], 1./np.min(return_value)
