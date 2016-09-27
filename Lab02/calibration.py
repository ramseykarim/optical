import numpy as np
import unpack as up
import matplotlib.pyplot as plt


def find_peaks(spectrum):
    s_med = np.mean(spectrum)
    comparisons = []
    comparison_function = np.frompyfunc(lambda x: x > 0, 1, 1)
    for i in np.arange(5) - 2:
        if i == 0:
            continue
        differences = spectrum - np.roll(spectrum, i)
        comparisons.append(comparison_function(differences))
    comparisons = np.array(comparisons)
    peaks = np.where(np.all(comparisons, axis=0))
    return [p for p in peaks[0] if spectrum[p] > s_med]


def centroid():
    return 1


def plot_peaks(run, peaks):
    plt.plot(run, color='k')
    m = np.max(run)
    print "M ", m
    for p in peaks:
        print "P ", p
        plt.plot([p, p], [0, m], color='g')
    plt.show()


class Calibrate:
    def __init__(self, unpacker):
        self.unpacker = unpacker
        self.run = up.average_run(unpacker.obtain("neon"))
        self.peaks = find_peaks(self.run)
        print self.peaks


c = Calibrate(up.Unpack())
plot_peaks(c.run, c.peaks)
