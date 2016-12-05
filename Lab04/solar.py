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
        center_wavelengths, center_spectrum = self.calibrate_sun(self.suns[self.center])

        fit = np.polyfit(center_wavelengths, center_spectrum, deg=2)
        fitted_spec = center_wavelengths**2. * fit[0] + center_wavelengths * fit[1] + center_wavelengths**0. * fit[2]
        center_spectrum = (center_spectrum - fitted_spec) * np.hanning(center_spectrum.size)
        halpha = np.where(np.abs(center_spectrum) == np.max(np.abs(center_spectrum)))[0]
        print "HALPHA", halpha
        center_spectrum = center_spectrum[halpha-15:halpha+16]

        in_transit = np.where(self.curve > np.mean(self.curve))[0]
        self.center = in_transit[in_transit.size/2]
        shift_array = np.array([])
        print "Running suns!"
        count = 0
        verb = False
        for position in in_transit:
            sys.stdout.write("Sun # {0} \r ".format(count))
            sys.stdout.flush()
            count += 1
            wl, spec = self.calibrate_sun(self.suns[position])
            fit = np.polyfit(wl, spec, deg=2)
            fitted_spec = wl**2. * fit[0] + wl * fit[1] + wl**0. * fit[2]
            spec = (spec - fitted_spec) * np.hanning(spec.size)
            spec = spec[halpha-15:halpha+16]
            offset = correlate_offset(center_spectrum, spec, 2, verb=verb)
            shift_array = np.append(shift_array, offset)
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
            sys.stdout.write("Finding background sky... {0} % \r".format(((count + 1)/len(background_indices[0]))))
            sys.stdout.flush()
            wl, spec = self.calibrator.calibrate(self.suns[i])
            specs.append(spec)
        sys.stdout.write("\nDone!\n")
        sys.stdout.flush()
        specs = np.array(specs)
        spec = np.mean(specs, axis=0)
        return spec


def correlate_offset_old(a_static, a_roll, max_offset):
    offset_range = np.arange(max_offset * 2 + 1) - max_offset
    return_value = np.array([])
    for offset in offset_range:
        shifted = np.roll(a_roll, offset)
        difference = a_static[max_offset:-max_offset] - shifted[max_offset:-max_offset]
        # noinspection PyTypeChecker
        return_value = np.append(return_value, np.sum(difference**2.))
    return offset_range[np.argmin(return_value)], 1./np.min(return_value)

def correlate_offset(a_static, a_roll, max_offset, verb=False):
    result = np.correlate(a_static, a_roll, 'same')
    if verb:
        print "CENTER of CORRELATE_OFFSET"
        print result[result.size/2 - 2:result.size/2 + 2]
    return centroid_at_zero(result, max_offset, verb=verb)

def centroid_at_zero(yarray, search_range, verb=False):
    zero = yarray.size/2
    ys = yarray[zero - search_range:zero + search_range + 1]
    xs = np.arange(search_range * 2 + 1) + 1
    centroid = np.sum(ys * xs) / np.sum(ys)
    if verb:
        print "VERBOSE CENTROID_AT_ZERO"
        print "Length: ", yarray.size
        print "zero at ", zero
        print "Raw centroid: ", centroid
        print "Search range: ", search_range
        print "XARR"
        print xs
        print "END VERB"
    return centroid - search_range - 1
