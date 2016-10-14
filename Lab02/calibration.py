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
        plt.plot([p], [-50], 'o', color='r')


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
    return [spectrum - dark_average for spectrum in spec_set]


def centroid_set(spectrum_list, s_type):
    centroids = []
    count = 0
    num_centroids = []
    if s_type == "neon":
        peak_requirement = 20
    elif s_type == "mercury":
        peak_requirement = 3
    else:
        raise RuntimeError("Ya done fucked up at PEAK_VARIATION")
    for spectrum in spectrum_list:
        c = find_centroid(spectrum, s_type)
        num_centroids.append(c.size)
        if c.size != peak_requirement:
            count += 1
        centroids.append(c)
    best_locations = np.where(np.array(num_centroids) == peak_requirement)[0]
    best_centroids = [centroids[x] for x in best_locations] if s_type == "neon" else centroids
    return np.array(best_centroids)


def peak_variation(spec_set, s_type):
    if s_type != "neon" and s_type != "mercury":
        raise RuntimeError("That's not a valid type for AVERAGE_WITH_ERROR")
    best_centroids = centroid_set(spec_set, s_type)
    error_propagation = ep.SingleSpectrumSet(best_centroids)
    return error_propagation


class Calibrate:
    def __init__(self, unpacker):
        self.unpacker = unpacker
        self.dark_bias = False
        self.peaks = False
        self.averaged_cal_spectrum = False
        self.error_propagation = False
        self.generate_wavelength_calibration_data()
        self.wavelength_calibration = self.wavelength_fit_bootstrap()
        self.adu = False
        self.gain_factor = False
        self.read_noise = False
        self.gain_calibration()

    def wavelength_fit_averaged(self):
        all_wavelengths = np.array(lls.ALL_PEAK_ARRAY)
        assert isinstance(self.peaks, np.ndarray)
        fit = lls.poly_fit(all_wavelengths, self.peaks)
        return lls.apply_fit(2048, fit)

    def wavelength_fit_bootstrap(self):
        assert isinstance(self.error_propagation, ep.ErrorPropagation)
        return lls.apply_fit(2048, self.error_propagation.bootstrap_lls(500).fit)

    def obtain_dark_subtract_set(self, s_type):
        names = {"mercury": 0, "neon": 1, "ADU": 2}
        # noinspection SpellCheckingInspection
        darks = ["darks_020", "darks_100", "bias_dark_03"]
        if s_type not in names:
            raise RuntimeError("That's not a valid calibration spectrum")
        dark = darks[names[s_type]]
        spec_set = self.unpacker.obtain(s_type)
        dark = up.average_run(self.unpacker.obtain(dark))
        spec_set = dark_adjust(spec_set, dark)
        return spec_set

    def generate_wavelength_calibration_data(self):
        m, n = "mercury", "neon"
        mercury_sets = self.obtain_dark_subtract_set(m)
        mercury_error_propagation = peak_variation(mercury_sets, m)
        neon_sets = self.obtain_dark_subtract_set(n)
        neon_error_propagation = peak_variation(neon_sets, n)
        self.peaks = np.append(mercury_error_propagation.mean(), neon_error_propagation.mean())
        self.averaged_cal_spectrum = up.average_run(mercury_sets) + up.average_run(neon_sets)
        self.error_propagation = ep.ErrorPropagation(mercury_error_propagation, neon_error_propagation)

    def wavelength_error(self, pixel):
        assert isinstance(self.error_propagation, ep.ErrorPropagation)
        a, b, c = self.error_propagation.fit
        ae, be, ce = self.error_propagation.fit_error
        error_in_a = lls.lambda_da(a, b, c, pixel) * ae
        error_in_b = lls.lambda_db(a, b, c, pixel) * be
        error_in_c = lls.lambda_dc(a, b, c, pixel) * ce
        error_total = error_in_a**2. + error_in_b**2. + error_in_c**2.
        return np.sqrt(error_total)

    def gain_calibration(self):
        adu_sets = np.array(self.obtain_dark_subtract_set("ADU"))
        self.adu = ep.SingleSpectrumSet(adu_sets)
        mean = self.adu.mean()
        std = self.adu.std()
        self.gain_factor, self.read_noise = lls.poly_fit(mean, std**2., degree=1)
        print "GAIN", self.gain_factor
        print "READ NOISE", self.read_noise

    def plot_gain_cal(self):
        assert isinstance(self.adu, ep.SingleSpectrumSet)
        mean = self.adu.mean()
        std = self.adu.std()
        plt.figure(5)
        plt.plot(mean, std**2., '.', color='blue')
        model = (self.gain_factor * mean) + self.read_noise
        plt.plot(mean, model, '-', color='black')
        plt.xscale('log', base=10)
        plt.yscale('log', base=10)
        plt.xlabel("Mean (ADU)", fontsize=16, family='serif')
        plt.ylabel("Variance (ADU$^{2}$)", fontsize=16, family='serif')

