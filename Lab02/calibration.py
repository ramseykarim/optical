import numpy as np
import unpack as up
import matplotlib.pyplot as plt
import error_propagation as ep
import wavelength_lls as lls


def find_peaks(spectrum, cutoff_f):
    comparisons = []
    comparison_function = np.frompyfunc(lambda x: x > 0, 1, 1)
    width = 3
    for number in np.arange(width*2 + 1) - width:
        if number == 0:
            continue
        differences = spectrum - np.roll(spectrum, number)
        comparisons.append(comparison_function(differences))
    comparisons = np.array(comparisons)
    peaks = np.where(np.all(comparisons, axis=0))
    relevant_peaks = cutoff_f(spectrum, peaks)
    return relevant_peaks


def find_centroid(spectrum, s_type):
    if s_type == "neon":
        peak_cutoff_f = neon_peak_cutoff
        centroid_cutoff_f = neon_centroid_cutoff
    elif s_type == "mercury":
        peak_cutoff_f = mercury_peak_cutoff
        centroid_cutoff_f = mercury_centroid_cutoff
    else:
        print "ya done fucked up in FIND_CENTROID"
        raise RuntimeError
    peaks = find_peaks(spectrum, cutoff_f=peak_cutoff_f)
    centroids = np.array([])
    for p in peaks:
        begin_peak, end_peak = centroid_cutoff_f(spectrum, p)
        peak_range = np.arange(end_peak - begin_peak) + begin_peak
        peak_values = spectrum[begin_peak:end_peak]
        assert peak_range.size == peak_values.size
        centroid = np.sum(peak_values * peak_range)/np.sum(peak_values)
        centroids = np.append(centroids, centroid)
    return centroids


def plot_peaks(run, peaks):
    plt.plot(run, color='k')
    m = np.max(run)
    for p in peaks:
        plt.plot([p, p], [0, m], color='g')


def neon_peak_cutoff(spectrum, peaks):
    return [p for p in peaks[0] if spectrum[p] > np.mean(spectrum)]


# noinspection PyUnresolvedReferences
def mercury_peak_cutoff(spectrum, peaks):
    peaks_list = np.copy(spectrum[peaks])
    peaks_list = peaks_list.tolist()
    peaks_list.sort()
    best_peaks = []
    for number in range(3):
        best_peaks.append(peaks_list.pop())
    peak_locations = []
    for index, point in enumerate(spectrum):
        if point in best_peaks:
            peak_locations.append(index)
    return np.array(peak_locations)


def neon_centroid_cutoff(spectrum, peak):
    cutoff = np.median(spectrum)
    fwhm = ((spectrum[peak] - cutoff) / 2.) + cutoff
    begin_peak = np.where(spectrum[peak::-1] < fwhm)
    begin_peak = peak - begin_peak[0][0]
    end_peak = np.where(spectrum[peak:] < fwhm)
    end_peak = end_peak[0][0] + peak
    return begin_peak, end_peak


# noinspection PyUnusedLocal
def mercury_centroid_cutoff(spectrum, peak):
    return peak - 20, peak + 20


def dark_adjust(spec_set, dark_average):
    new_spec_set = []
    for spectrum in spec_set:
        new_spec_set.append(spectrum - dark_average)
    return new_spec_set


def centroid_set(spectrum_list, s_type):
    print "LIST SIZE: ", spectrum_list.__len__()
    centroids = []
    count = 0
    num_centroids = []
    if s_type == "neon":
        peak_requirement = 20
    elif s_type == "mercury":
        peak_requirement = 3
    else:
        print "Ya done fucked up at PEAK_VARIATION"
        raise RuntimeError
    print "Dark subtracting...",
    for spectrum in spectrum_list:
        c = find_centroid(spectrum, s_type)
        num_centroids.append(c.size)
        if c.size != peak_requirement:
            count += 1
        centroids.append(c)
    print " done."
    print "Fuck up count: ", count
    best_locations = np.where(np.array(num_centroids) == peak_requirement)[0]
    best_centroids = [centroids[x] for x in best_locations] if s_type == "neon" else centroids
    return np.array(best_centroids)


class Calibrate:
    def __init__(self, unpacker):
        self.unpacker = unpacker
        self.dark_bias = False
        self.neon_avg = False
        self.mercury_avg = False
        self.neon_avg_centroids = False
        self.mercury_avg_centroids = False
        self.peaks = False
        self.generate_calibration_data()

    def wavelength_fit_averaged(self, degree=2):
        all_wavelengths = np.array(lls.ALL_PEAK_ARRAY)
        assert isinstance(self.peaks, np.ndarray)
        fit = lls.poly_fit(self.peaks, all_wavelengths, degree=degree)
        pixel_range = np.arange(2048)
        wavelength_solution = np.zeros(2048)
        for power, coefficient in enumerate(fit[::-1]):
            wavelength_solution += coefficient * np.power(pixel_range, power)
        return wavelength_solution

    def generate_calibration_data(self):
        m, n = "mercury", "neon"
        # noinspection SpellCheckingInspection
        md, nd = "darks_020", "darks_100"
        mercury_sets = self.unpacker.obtain(m)
        mercury_dark = up.average_run(self.unpacker.obtain(md))
        mercury_sets = dark_adjust(mercury_sets, mercury_dark)
        mercury_best = centroid_set(mercury_sets, m)
        mercury_error_propagation = ep.SingleSpectrumPeakSet(mercury_best)
        self.mercury_avg = up.average_run(mercury_sets)
        self.mercury_avg_centroids = find_centroid(self.mercury_avg, m)
        neon_sets = self.unpacker.obtain(n)
        neon_dark = up.average_run(self.unpacker.obtain(nd))
        neon_sets = dark_adjust(neon_sets, neon_dark)
        neon_best = centroid_set(neon_sets, n)
        neon_error_propagation = ep.SingleSpectrumPeakSet(neon_best)
        self.neon_avg = up.average_run(neon_sets)
        self.neon_avg_centroids = find_centroid(self.neon_avg, n)
        self.peaks = np.append(self.mercury_avg_centroids, self.neon_avg_centroids)
        e_prop = ep.ErrorPropagation(mercury_error_propagation, neon_error_propagation)
        e_prop.bootstrap_lls(5)

