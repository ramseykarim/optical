import numpy as np
import unpack as up
import matplotlib.pyplot as plt


NEON_PEAKS = [5852.49, 5881.89, 5944.83, 5975.53,
              6030.00, 6074.34, 6096.16, 6143.06,
              6163.59, 6217.28, 6266.49, 6304.79,
              6334.43, 6382.99, 6402.25, 6506.53,
              6532.88, 6598.95, 6678.28, 6717.04]

MERCURY_PEAKS = [3650.15, 4046.56, 4358.33]


def find_peaks(spectrum, cutoff_f):
    comparisons = []
    comparison_function = np.frompyfunc(lambda x: x > 0, 1, 1)
    for number in np.arange(5) - 2:
        if number == 0:
            continue
        differences = spectrum - np.roll(spectrum, number)
        comparisons.append(comparison_function(differences))
    comparisons = np.array(comparisons)
    peaks = np.where(np.all(comparisons, axis=0))
    relevant_peaks = cutoff_f(spectrum, peaks)
    return relevant_peaks


def find_centroid(spectrum, peak_cutoff_f, cutoff_f):
    peaks = find_peaks(spectrum, cutoff_f=peak_cutoff_f)
    centroids = np.array([])
    for p in peaks:
        begin_peak, end_peak = cutoff_f(spectrum, p)
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


def mercury_centroid_cutoff(spectrum, peak):
    return peak - 20, peak + 20


class Calibrate:
    def __init__(self, unpacker):
        self.unpacker = unpacker
        self.neon = up.average_run(unpacker.obtain("neon"))
        self.neon_centroids = find_centroid(self.neon, neon_peak_cutoff, neon_centroid_cutoff)
        self.mercury = up.average_run(unpacker.obtain("mercury"))
        self.mercury_centroids = find_centroid(self.mercury, mercury_peak_cutoff, mercury_centroid_cutoff)
        self.peaks = np.append(self.mercury_centroids, self.neon_centroids)

    def wavelength_fit(self, degree=2):
        all_wavelengths = np.array(MERCURY_PEAKS + NEON_PEAKS)
        # noinspection PyTupleAssignmentBalance
        results = np.polyfit(self.peaks, all_wavelengths, degree, full=True)
        fit, residuals = results[:2]
        pixel_range = np.arange(2048)
        wavelength_solution = np.zeros(2048)
        print "SUM OF RESIDUALS FOR DEGREE ", degree, ": ", residuals[0]
        for power, coefficient in enumerate(fit[::-1]):
            wavelength_solution += coefficient * np.power(pixel_range, power)
        return wavelength_solution
