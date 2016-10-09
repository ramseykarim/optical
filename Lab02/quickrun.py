import unpack as up
import calibration as cal
import numpy as np
import matplotlib.pyplot as plt
import wavelength_lls as lls
import sys

c = cal.Calibrate(up.Unpack())
all_wavelengths = lls.ALL_PEAK_ARRAY
pixels = np.arange(2048)
plt.show()
sys.exit(0)

deg2 = c.wavelength_fit_averaged(degree=2)
deg2comp = all_wavelengths - np.array([deg2[int(x)] for x in c.peaks])
print "Residuals: ", np.sum(deg2comp**2.)

plt.figure(1)
cal.plot_peaks(c.neon_avg, c.neon_avg_centroids)
cal.plot_peaks(c.mercury_avg, c.mercury_avg_centroids)
plt.xlabel("Pixel Number")
plt.ylabel("Intensity")
plt.title("Measured Spectra for Ne and Hg with peaks")

plt.figure(2)

plt.subplot(211)
plt.plot(c.neon_avg_centroids, lls.NEON_PEAKS, 'o', color='blue')
plt.plot(c.mercury_avg_centroids, lls.MERCURY_PEAKS, 'o', color='blue')
plt.plot(pixels, deg2, '-')
plt.xlabel("Pixel")
plt.ylabel("Wavelength (A)")
plt.title("Wavelength Solution to Spectrometer")

plt.subplot(212)
plt.plot(c.peaks, deg2comp, 'o', color='green')
plt.xlabel("Pixel")
plt.ylabel("Difference (A)")
plt.title("2nd Degree fit comparison")

plt.tight_layout()

plt.show()
