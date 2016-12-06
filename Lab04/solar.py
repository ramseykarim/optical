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
            self.curve = np.append(self.curve, np.mean(frame))
        self.center = np.argmax(self.curve)

    def calibrate_sun(self, sun_frame):
        wavelengths, prelim_spectrum = self.calibrator.calibrate(sun_frame)
        prelim_spec = prelim_spectrum - self.background_light
        prelim_spec -= np.median(prelim_spec)
        prelim_spec /= (np.max(prelim_spec) - np.min(prelim_spec))
        return wavelengths, prelim_spec

    def test_suns(self):
        center_wl, center_spectrum = self.calibrate_sun(self.suns[self.center])

        center_spectrum, h_alpha = clean_solar_spectrum(center_wl, center_spectrum)
        plt.figure()
        plt.plot(center_spectrum)
        h_alpha_offset = fit_h_alpha(center_spectrum)
        print "H_ALPHA_OFFSET", h_alpha_offset
        in_transit = np.where(self.curve > np.mean(self.curve))[0]
        self.center = in_transit[in_transit.size / 2]
        shift_array = []
        print "Running suns!"
        count = 0
        verb = False
        for position in in_transit:
            sys.stdout.write("Sun # {0} \r ".format(count))
            sys.stdout.flush()
            count += 1
            wl, spec = self.calibrate_sun(self.suns[position])
            spec = clean_solar_spectrum(wl, spec, h_alpha=h_alpha)
            # offset = correlate_offset(center_spectrum, spec, 2, verb=verb)
            offset = fit_h_alpha(spec, offset=h_alpha_offset)
            shift_array.append(offset)
        print "\n"
        plt.figure()
        plt.plot(shift_array, '.', color='green')
        plt.title("shift")
        plt.figure()
        plt.plot(self.curve[in_transit], '.', color='blue')
        plt.title("LC")

    def generate_light_noise(self):
        background_indices = np.where(self.curve <= np.median(self.curve))
        specs = []
        for count, i in enumerate(background_indices[0]):
            sys.stdout.write(
                "Finding background sky... {0} % \r".format(int(100. * (count + 1.) / len(background_indices[0])))
            )
            sys.stdout.flush()
            wl, spec = self.calibrator.calibrate(self.suns[i])
            specs.append(spec)
        sys.stdout.write("\nDone!\n")
        sys.stdout.flush()
        specs = np.array(specs)
        spec = np.mean(specs, axis=0)
        return spec


def correlate_offset(a_static, a_roll, max_offset, verb=False):
    result = np.correlate(a_static, a_roll, 'same')
    if verb:
        print "CENTER of CORRELATE_OFFSET"
        print result[result.size / 2 - 2:result.size / 2 + 2]
    return centroid_at_zero(result, max_offset, verb=verb)


def fit_h_alpha(array, deg=4, offset=0):
    size = array.size
    assert size >= 5
    x0 = np.arange(size) - size / 2
    fit = cal.poly_fit(x0, array, deg=deg)
    x = np.arange(x0[0], x0[size - 1], size / 1000.)
    y = cal.polynomial(x, fit)
    return x[np.where(y == np.min(y))] - offset


def centroid_at_zero(y_array, search_range, verb=False):
    zero = y_array.size / 2
    ys = y_array[zero - search_range:zero + search_range + 1]
    xs = np.arange(search_range * 2 + 1) + 1
    centroid = np.sum(ys * xs) / np.sum(ys)
    if verb:
        print "VERBOSE CENTROID_AT_ZERO"
        print "Length: ", y_array.size
        print "zero at ", zero
        print "Raw centroid: ", centroid
        print "Search range: ", search_range
        print "X_ARR"
        print xs
        print "END VERB"
    return centroid - search_range - 1


def clean_solar_spectrum(wavelengths, solar_spectrum, width=5, h_alpha=None):
    fitted_spec = cal.polynomial(wavelengths, cal.poly_fit(wavelengths, solar_spectrum, deg=2))
    fixed_spec = (solar_spectrum - fitted_spec) * np.hanning(solar_spectrum.size)
    if h_alpha is None:
        h_alpha = np.where(np.abs(fixed_spec) == np.max(np.abs(fixed_spec)))[0]
    fixed_spec = fixed_spec[h_alpha - width:h_alpha + width + 1]
    if h_alpha is None:
        return fixed_spec, h_alpha
    else:
        return fixed_spec
